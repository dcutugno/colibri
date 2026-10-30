#include "backend_cuda_dsv4.h"
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#if !defined(COLI_DSV4_NO_TC)
#include <cublasLt.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
#include <vector>
#include <mutex>
#ifndef _WIN32
#include <unistd.h>
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#ifdef COLI_DSV4_NCCL
#include <nccl.h>
#endif
#ifdef COLI_DSV4_DEEPGEMM
#include <cuda.h>
#include <deep_gemm/impls/sm120_fp8_fp4_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm120_split_k_reduce.cuh>
#include <deep_gemm/impls/sm120_bf16_gemm.cuh>
#endif

/* The cuBLASLt MXFP8 tensor-core path (tc_plan/tc_fp4_matvec, enabled at
 * runtime by DSV4_CUDA_TC=1, off by default) needs the cuBLASLt >= 12.8
 * block-scaling APIs (CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 / *_SCALE_MODE).
 * Older toolkits (e.g. Jetson JetPack with CUDA 12.6) lack them; build with
 * -DCOLI_DSV4_NO_TC to compile without that path (generic DLL/object only —
 * the DeepGEMM sm_120a flavour needs 12.8+ anyway). Without it nothing here
 * touches cuBLASLt, so the Windows DLLs the release ships (cuda-dsv4-dll
 * DSV4_TC=0) need no cublasLt64_12.dll beside them (#2047). */
#if defined(COLI_DSV4_NO_TC)
#define COLI_DSV4_TC 0
#else
#define COLI_DSV4_TC 1
#endif
struct Dsv4CudaTensor { void *w,*bf16_cache; uint8_t *scale;int *dg_scale; int O,I,fmt,device,own_w,own_scale,own_dg_scale; long long bytes; void *block,*block2; long long blockpad; /* block mirror: the owner's weight-run and scale-run allocations (dsv4_cuda_mirror_alloc) */
    cudaEvent_t ev;int evwait; /* a prefetched mirror's completion on the prefetch stream (mirror_point with S<0); the expert group waits for it once */ };
static bool run_fp8_attn(const Dsv4CudaTensor*t,const float*x,float*y,int groups,cudaStream_t st);   /* below, by mv_fp8_dec */
struct Dsv4CudaActivation { float *data; long long elements; int device; };
struct Dsv4CudaGraph { cudaGraph_t graph;cudaGraphExec_t exec;int device; };
struct Dsv4CudaKvCache { float *kv,*compressed,*comp_value,*comp_score,*rope_cos,*rope_sin,*comp_cos,*comp_sin;int *compressed_len;int device,window,dim,max_tokens,max_compressed,rope_pairs; };
struct MvDesc { const uint8_t *w,*scale; const float *x; float *y; };
struct ExpertPtr { const uint8_t *w,*scale; };
struct Dsv4CudaExpertSet { ExpertPtr *table;int64_t *hash;Dsv4CudaTensor *sg,*su,*sd;uint8_t *bank_fc1,*bank_fc1_scale,*bank_fc2,*bank_fc2_scale;int *bank_fc1_dg_scale,*bank_fc2_dg_scale;int count,device,H,I,vocab,topk; };
#if COLI_DSV4_TC
struct TcPlan {int O,I,ok;cublasLtMatmulDesc_t op;cublasLtMatrixLayout_t a,b,c,d;cublasLtMatmulHeuristicResult_t algo;};
#endif
struct Dev { void *stg;size_t stgcap; int id,sms; cudaStream_t stream,aux,pre; cudaEvent_t ready,fork,join;float *dx,*dy,*p1,*p2,*p3,*p4,*aux1,*aux2,*aux3; size_t xcap,ycap,p1cap,p2cap,p3cap,p4cap,aux1cap,aux2cap,aux3cap; float *hx,*hy;
    /* decode expert group: the upload fence event, pinned staging for its
     * small copies (x, descriptors + weights, y), DSV4_MOE_KPROF sums */
    cudaEvent_t upfence; void *grp_hparams,*grp_params; float *hx_pin,*hy_pin; size_t grp_paramscap,grp_hcap; int grp_pinned;
    double kp_dev,kp_host; int kp_n;
#ifdef COLI_DSV4_NCCL
    ncclComm_t ep_comm;
#endif
    int *decode_state;
    MvDesc *mvdesc;size_t mvdesccap;float *expert_weights;size_t expert_weightscap;int *expert_ids;size_t expert_idcap;
    int *emaj;size_t emajcap;float *emv1,*emv2,*emv4;size_t emv1cap,emv2cap,emv4cap;unsigned long long *emcount; /* expert-major MoE */
    float *mmv;size_t mmvcap;unsigned long long *mmvcount; /* DSV4_MM_WARP_VERIFY */
#if COLI_DSV4_TC
    cublasLtHandle_t lt;TcPlan plans[2];
#endif
    void *tc_workspace;uint8_t *tcw,*tcs,*tcx,*tcxs;size_t tcwcap,tcscap,tcxcap,tcxscap;
#ifdef COLI_DSV4_DEEPGEMM
    uint8_t *dga1,*dga2;int *dgsfa1,*dgsfa2,*dglayout,*dgrowmap;__nv_bfloat16 *dgmm1,*dgmm2;float *dgref,*dgdensews;size_t dga1cap,dga2cap,dgsfa1cap,dgsfa2cap,dglayoutcap,dgrowmapcap,dgmm1cap,dgmm2cap,dgrefcap,dgdensewscap;
#endif
};
static Dev g[16];static int ng;
static Dev *ctx(int d){for(int i=0;i<ng;i++)if(g[i].id==d)return g+i;return nullptr;}
static int ok(cudaError_t e,const char *s){if(e==cudaSuccess)return 1;fprintf(stderr,"[DSV4 CUDA] %s: %s\n",s,cudaGetErrorString(e));return 0;}
#if COLI_DSV4_TC
static int bok(cublasStatus_t e,const char*s){if(e==CUBLAS_STATUS_SUCCESS)return 1;fprintf(stderr,"[DSV4 CUDA] %s: cuBLAS status %d\n",s,(int)e);return 0;}
#endif
#ifdef COLI_DSV4_NCCL
static int nok(ncclResult_t e,const char*s){if(e==ncclSuccess)return 1;fprintf(stderr,"[DSV4 NCCL] %s: %s\n",s,ncclGetErrorString(e));return 0;}
#endif
/* For cudaMalloc results: a failed allocation is recoverable (callers fall
 * back or skip work), but the runtime also keeps it as its last error, and the
 * next launch check (cudaGetLastError) would report that OOM against a kernel
 * that never touched the missing buffer. Read it here. */
static int alloc_ok(cudaError_t e,const char *s){if(e==cudaSuccess)return 1;(void)cudaGetLastError();return ok(e,s);}
static int buf(void **p,size_t *cap,size_t n){if(*cap>=n)return 1;if(*p)cudaFree(*p);*p=nullptr;*cap=0;if(!alloc_ok(cudaMalloc(p,n),"buffer allocation"))return 0;*cap=n;return 1;}
/* DSV4_PIN_STAGE (1): a per-device page-locked staging block for a call
 * that ends with a stream sync (the prefill chain: its seven inputs as one
 * staged copy, its output download). A cudaMemcpyAsync from pageable memory
 * blocks the host per call on WDDM; a pinned copy queues. The caller's
 * sync retires the copies, so the block needs no event. NULL = pageable. */
static int pin_stage_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_PIN_STAGE");on=!(e&&*e=='0');}return on;}
static void *stage_get(Dev*c,size_t n){
    if(!pin_stage_on())return nullptr;
    if(c->stgcap<n){
        size_t cap=c->stgcap?c->stgcap:((size_t)1<<20);while(cap<n)cap*=2;
        if(c->stg)cudaFreeHost(c->stg);
        c->stg=nullptr;c->stgcap=0;
        if(!alloc_ok(cudaHostAlloc(&c->stg,cap,cudaHostAllocPortable),"pinned staging"))return nullptr;
        c->stgcap=cap;
    }
    return c->stg;
}


__constant__ float e8_table[256];
__global__ void set_decode_state(int*state,int token,int position){if(!threadIdx.x){state[0]=token;state[1]=position;}}
__device__ float e4m3(uint8_t b){int sign=b>>7,e=(b>>3)&15,m=b&7;float v;
    if(!e)v=ldexpf((float)m,-9);else if(e==15)v=m==7?NAN:ldexpf(1.f+m/8.f,8);else v=ldexpf(1.f+m/8.f,e-7);
    return sign?-v:v;}
__device__ float e8(uint8_t b){return e8_table[b];}
__device__ float f4(uint8_t q){const float a[8]={0,.5f,1,1.5f,2,3,4,6};return q&8?-a[q&7]:a[q];}
__global__ void mm_bf16_batch(const __nv_bfloat16*w,const float*x,float*y,int O,int I,int T);
template<int F> __global__ void mv(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups){
    int o=blockIdx.x;if(o>=O)return;float sum=0;
    const float *xi=x+(long long)(o/(O/groups))*I;for(int i=threadIdx.x;i<I;i+=blockDim.x){float v;
        if constexpr(F==8||F==9){v=e4m3(w[(long long)o*I+i])*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
        else {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}
        sum+=xi[i]*v;
    }
    __shared__ float r[256];r[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)r[threadIdx.x]+=r[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[o]=r[0];
}
template<int F,int R> __global__ void mv_rows(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups){
    int base=blockIdx.x*R;float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){const float *xi=x+(long long)(o/(O/groups))*I;float v;
            if constexpr(F==8||F==9){v=e4m3(w[(long long)o*I+i])*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
            else {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}sum[r]+=xi[i]*v;}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)y[base+r]=s[r][0];}
}
/* DSV4_MV_WARP (1): the one-token fp8 matvec as one warp per row. mv<F>
 * has thread j of its 256-thread block sum partial j (i = j, j+256, ... in
 * order) and then fold the 256 partials as a tree (128, 64, ..., 1). Here
 * lane l holds partials 8l..8l+7: eight consecutive bytes of the row, one
 * 8-byte load per lane and 256 contiguous bytes per warp per step, x as two
 * float4 (mv<F> loads one byte per thread). The tree's 128/64/32/16/8 steps
 * are shuffles over lanes 16/8/4/2/1 apart, its 4/2/1 steps in-lane adds:
 * every output has mv<F>'s operands in mv<F>'s order (bit-identical;
 * DSV4_MV_WARP_VERIFY runs both and counts differing words). I % 256 == 0
 * and 8-byte aligned rows. Eight rows per 256-thread block; e4m3 through a
 * shared table of exactly e4m3()'s values. */
template<int F> __global__ void __launch_bounds__(256) mv_warp(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int groups){
    __shared__ float lut[256];
    lut[threadIdx.x]=e4m3((uint8_t)threadIdx.x);__syncthreads();
    int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
    const float *xi=x+(long long)(o/(O/groups))*I;
    const uint8_t *wr=w+(long long)o*I;
    const uint8_t *scr=sc+(long long)(o/128)*((I+127)/128);
    float p[8];
    #pragma unroll
    for(int k=0;k<8;k++)p[k]=0.f;
    for(int c=0;c<I;c+=256){
        int i0=c+8*lane;
        uint2 wv=*(const uint2*)(wr+i0);
        float4 xa=*(const float4*)(xi+i0),xb=*(const float4*)(xi+i0+4);
        float es=e8(scr[i0/128]);
        unsigned b[8]={wv.x&255u,(wv.x>>8)&255u,(wv.x>>16)&255u,wv.x>>24,wv.y&255u,(wv.y>>8)&255u,(wv.y>>16)&255u,wv.y>>24};
        float xs[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int k=0;k<8;k++){float v=lut[b[k]]*es;if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));p[k]+=xs[k]*v;}
    }
    #pragma unroll
    for(int k=0;k<8;k++){
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],16);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],8);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],4);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],2);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],1);
    }
    if(!lane){float v0=p[0]+p[4],v1=p[1]+p[5],v2=p[2]+p[6],v3=p[3]+p[7];v0=v0+v2;v1=v1+v3;y[o]=v0+v1;}
}
/* The bf16 one (mm_bf16_batch with a batch of one, mv_bf16: the same
 * partials and tree): lane l's eight bf16 as one 16-byte load. */
__global__ void __launch_bounds__(256) mv_bf16_warp(const __nv_bfloat16 *w,const float *x,float *y,int O,int I){
    int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
    const __nv_bfloat16 *wr=w+(long long)o*I;
    float p[8];
    #pragma unroll
    for(int k=0;k<8;k++)p[k]=0.f;
    for(int c=0;c<I;c+=256){
        int i0=c+8*lane;
        uint4 wv=*(const uint4*)(wr+i0);
        float4 xa=*(const float4*)(x+i0),xb=*(const float4*)(x+i0+4);
        unsigned u[4]={wv.x,wv.y,wv.z,wv.w};
        float xs[8]={xa.x,xa.y,xa.z,xa.w,xb.x,xb.y,xb.z,xb.w};
        #pragma unroll
        for(int k=0;k<8;k++){float v=__uint_as_float((k&1)?(u[k>>1]&0xffff0000u):(u[k>>1]<<16));p[k]+=xs[k]*v;}
    }
    #pragma unroll
    for(int k=0;k<8;k++){
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],16);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],8);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],4);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],2);
        p[k]+=__shfl_down_sync(0xffffffffu,p[k],1);
    }
    if(!lane){float v0=p[0]+p[4],v1=p[1]+p[5],v2=p[2]+p[6],v3=p[3]+p[7];v0=v0+v2;v1=v1+v3;y[o]=v0+v1;}
}
static int mv_warp_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_MV_WARP");on=!(e&&*e=='0');}return on;}
static int mv_warp_verify(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_MV_WARP_VERIFY");on=e&&*e&&*e!='0';}return on;}
static bool mv_warp_fits(const void*w,const float*x,int I,int walign){return I>0&&I%256==0&&!((uintptr_t)x&15)&&!((uintptr_t)w&(walign-1));}
__global__ void mv_warp_diff_kernel(const float*a,const float*b,int n,unsigned*cnt){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&__float_as_uint(a[i])!=__float_as_uint(b[i]))atomicAdd(cnt,1u);}
/* DSV4_MV_WARP_VERIFY: `got` (the warp kernel's output, O floats) against
 * `ref` (the block kernel's), counted on the device, reported per call and
 * in total; synchronous. */
static void mv_warp_report(const char*tag,const float*ref,const float*got,int O,int I,cudaStream_t st){
    static unsigned*dc=nullptr;static unsigned long long calls=0,bad_calls=0,words=0;
    if(!dc){if(cudaMalloc((void**)&dc,sizeof(unsigned))!=cudaSuccess){dc=nullptr;return;}}
    cudaMemsetAsync(dc,0,sizeof(unsigned),st);
    mv_warp_diff_kernel<<<(O+255)/256,256,0,st>>>(ref,got,O,dc);
    unsigned n=0;cudaMemcpyAsync(&n,dc,sizeof(n),cudaMemcpyDeviceToHost,st);cudaStreamSynchronize(st);
    calls++;words+=n;if(n)bad_calls++;
    fprintf(stderr,"mvwverify %s O=%d I=%d mismatches=%u (calls=%llu bad=%llu words=%llu)\n",tag,O,I,n,calls,bad_calls,words);
}
static float*mv_warp_scratch(size_t n){static float*p=nullptr;static size_t cap=0;if(cap<n){if(p)cudaFree(p);p=nullptr;cap=0;if(cudaMalloc((void**)&p,n*4)!=cudaSuccess){p=nullptr;return nullptr;}cap=n;}return p;}
/* bf16 rows for one token: the warp kernel when it fits, else
 * mm_bf16_batch with a batch of one (what the callers launched). */
static void launch_bf16_one(const __nv_bfloat16*w,const float*x,float*y,int O,int I,cudaStream_t st){
    if(mv_warp_on()&&mv_warp_fits(w,x,I,16)){
        if(mv_warp_verify()){float*t=mv_warp_scratch((size_t)O);if(t){
            mm_bf16_batch<<<dim3(O,1),256,0,st>>>(w,x,y,O,I,1);
            mv_bf16_warp<<<(O+7)/8,256,0,st>>>(w,x,t,O,I);
            mv_warp_report("bf16",y,t,O,I,st);return;}}
        mv_bf16_warp<<<(O+7)/8,256,0,st>>>(w,x,y,O,I);
    }else mm_bf16_batch<<<dim3(O,1),256,0,st>>>(w,x,y,O,I,1);
}
template<int R> __global__ void mv_fp4_grouped(const MvDesc *desc,int count,int O,int I){
    int matrix=blockIdx.y,base=blockIdx.x*R;if(matrix>=count)return;MvDesc d=desc[matrix];float sum[R]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){
        #pragma unroll
        for(int r=0;r<R;r++){int o=base+r;if(o<O){uint8_t p=d.w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;
            sum[r]+=d.x[i]*f4(q)*e8(d.scale[(long long)o*(I/32)+i/32]);}}
    }
    __shared__ float s[R][256];
    #pragma unroll
    for(int r=0;r<R;r++)s[r][threadIdx.x]=sum[r];__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int r=0;r<R;r++)s[r][threadIdx.x]+=s[r][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){
        #pragma unroll
        for(int r=0;r<R;r++)if(base+r<O)d.y[base+r]=s[r][0];}
}
#ifndef COLI_DSV4_DEEPGEMM /* generic batched MoE only (DeepGEMM builds use dg_moe_batch_contiguous) */
/* Expert-major fp4 matvec for the generic batched MoE (DSV4_MOE_EXPERT_MAJOR).
 * mv_fp4_grouped runs one matvec per route, so an expert picked by k tokens of
 * the chunk has its weights streamed k times, and each 256-thread block spends
 * most of its time in the shared-memory tree. Here one WARP owns one row of
 * one bank expert and walks every route that picked it, TB at a time. Lane l
 * holds the tree's partials j = l + 32q (q = 0..7): partial j sums
 * i = j, j+256, ... in order, exactly like thread j of mv_fp4_grouped; the
 * tree's 128/64/32 steps pair partials inside a lane and 16..1 are shuffles,
 * so every output has the same operands in the same order: bit-identical
 * (DSV4_MOE_EXPERT_MAJOR_VERIFY checks it at run time). No shared memory, no
 * barriers. Rows below `split` go to y0, the rest to y1 (FC1 = gate, up). */
template<int TB> __global__ void __launch_bounds__(256) mv_fp4_expert_major(const uint8_t*bank_w,const uint8_t*bank_s,size_t wstride,size_t sstride,
        const int*experts,const int*offsets,const int*routes,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I){
    int slot=blockIdx.y,lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);
    if(o>=O)return;
    int e=experts[slot],begin=offsets[slot],end=offsets[slot+1];
    const uint8_t*w=bank_w+(size_t)e*wstride+(size_t)o*(I/2);const uint8_t*sc=bank_s+(size_t)e*sstride+(size_t)o*(I/32);
    for(int g=begin;g<end;g+=TB){
        int n=end-g<TB?end-g:TB;const float*xs[TB];float p[TB][8];
        #pragma unroll
        for(int t=0;t<TB;t++){xs[t]=t<n?x+(long long)(routes[g+t]/xdiv)*xstride:x;
            #pragma unroll
            for(int q=0;q<8;q++)p[t][q]=0.f;}
        for(int c0=0;c0<I/32;c0+=8){
            #pragma unroll
            for(int q=0;q<8;q++){int i=lane+32*(c0+q);
                float xv[TB];
                #pragma unroll
                for(int t=0;t<TB;t++)xv[t]=t<n?xs[t][i]:0.f;
                uint8_t pk=w[i>>1],qq=(i&1)?pk>>4:pk&15;float f=f4(qq),es=e8(sc[i/32]);
                #pragma unroll
                for(int t=0;t<TB;t++)if(t<n)p[t][q]+=xv[t]*f*es;
            }
        }
        #pragma unroll
        for(int t=0;t<TB;t++){
            float v0=p[t][0]+p[t][4],v1=p[t][1]+p[t][5],v2=p[t][2]+p[t][6],v3=p[t][3]+p[t][7];
            v0=v0+v2;v1=v1+v3;float v=v0+v1;
            #pragma unroll
            for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
            if(!lane&&t<n){long long route=routes[g+t];
                if(o<split)y0[route*ystride+o]=v;else y1[route*ystride+(o-split)]=v;}
        }
    }
}
static void run_fp4_expert_major(int tb,const uint8_t*w,const uint8_t*sc,size_t wstride,size_t sstride,
        const int*ex,const int*off,const int*rt,int nex,const float*x,long long xstride,int xdiv,
        float*y0,float*y1,int split,long long ystride,int O,int I,cudaStream_t stream){
    dim3 grid((O+7)/8,nex);
    if(tb==3)mv_fp4_expert_major<3><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
    else if(tb==2)mv_fp4_expert_major<2><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
    else if(tb==6)mv_fp4_expert_major<6><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
    else mv_fp4_expert_major<4><<<grid,256,0,stream>>>(w,sc,wstride,sstride,ex,off,rt,x,xstride,xdiv,y0,y1,split,ystride,O,I);
}
__global__ void count_bit_mismatch(const float*a,const float*b,long long n,unsigned long long*count){
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n&&__float_as_uint(a[i])!=__float_as_uint(b[i]))atomicAdd(count,1ULL);}
#endif
template<int F> static void run_mv(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I,int groups,cudaStream_t stream){
    static int rows=-1;if(rows<0){const char *p=getenv(F==4?"DSV4_CUDA_ROWS":"DSV4_CUDA_FP8_ROWS");rows=p?atoi(p):(F==4?4:1);}
    if(rows==32)mv_rows<F,32><<<(O+31)/32,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==16)mv_rows<F,16><<<(O+15)/16,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==8)mv_rows<F,8><<<(O+7)/8,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else if(rows==4)mv_rows<F,4><<<(O+3)/4,256,0,stream>>>(w,sc,x,y,O,I,groups);
    else{
        if constexpr(F==8||F==9){
            if(mv_warp_on()&&mv_warp_fits(w,x,I,8)){
                if(mv_warp_verify()){float*t=mv_warp_scratch((size_t)O);if(t){
                    mv<F><<<O,256,0,stream>>>(w,sc,x,y,O,I,groups);
                    mv_warp<F><<<(O+7)/8,256,0,stream>>>(w,sc,x,t,O,I,groups);
                    mv_warp_report(F==9?"fp8b":"fp8",y,t,O,I,stream);return;}}
                mv_warp<F><<<(O+7)/8,256,0,stream>>>(w,sc,x,y,O,I,groups);return;
            }
        }
        mv<F><<<O,256,0,stream>>>(w,sc,x,y,O,I,groups);
    }
}
static void run_fp4_grouped(const MvDesc *desc,int count,int O,int I,cudaStream_t stream){
    static int rows=-1;if(rows<0){const char *p=getenv("DSV4_CUDA_FP4_ROWS");rows=p?atoi(p):2;}
    if(rows==8)mv_fp4_grouped<8><<<dim3((O+7)/8,count),256,0,stream>>>(desc,count,O,I);
    else if(rows==4)mv_fp4_grouped<4><<<dim3((O+3)/4,count),256,0,stream>>>(desc,count,O,I);
    else if(rows==2)mv_fp4_grouped<2><<<dim3((O+1)/2,count),256,0,stream>>>(desc,count,O,I);
    else mv_fp4_grouped<1><<<dim3(O,count),256,0,stream>>>(desc,count,O,I);
}
/* Prefill batch GEMM without DeepGEMM: one block per output row and 16-token
 * tile, so each dequantized weight value is reused across the tile and the
 * weight matrix is read ceil(T/16) times per call instead of T times (what a
 * matvec-per-token loop would cost). Numerics match mv<F> exactly: same
 * e4m3/e8 dequant, same bf16 round-trip for fmt 9. x is tokens-major
 * [T][I], y is tokens-major [T][O] — the layout dg_dense_batch_dispatch and
 * the engine's coli_fp8_matmul_batch_ref both use. */
template<int F> __global__ void mm_batch(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int T){
    int o=blockIdx.x;if(o>=O)return;int tbase=blockIdx.y*16,tcount=T-tbase<16?T-tbase:16;
    float sum[16]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){float v;
        if constexpr(F==8||F==9){v=e4m3(w[(long long)o*I+i])*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));}
        else {uint8_t p=w[((long long)o*I+i)>>1],q=(i&1)?p>>4:p&15;v=f4(q)*e8(sc[(long long)o*(I/32)+i/32]);}
        #pragma unroll
        for(int t=0;t<16;t++)if(t<tcount)sum[t]+=x[(long long)(tbase+t)*I+i]*v;
    }
    __shared__ float s[16][256];
    #pragma unroll
    for(int t=0;t<16;t++)s[t][threadIdx.x]=sum[t];
    __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int t=0;t<16;t++)s[t][threadIdx.x]+=s[t][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x)for(int t=0;t<tcount;t++)y[(long long)(tbase+t)*O+o]=s[t][0];
}
#ifndef COLI_DSV4_DEEPGEMM /* run_mm_batch's callers are all generic-build paths */
/* Warp-per-row mm_batch for fp8 / bf16-rounded fp8 weights (DSV4_MM_WARP).
 * One warp owns one output row and TB tokens; lane l holds the 256-wide
 * tree's partials j = l + 32q, each summing i = j, j+256, ... in mm_batch's
 * order, and the tree's 128/64/32 steps are in-lane adds, 16..1 shuffles:
 * every output has mm_batch's operands in mm_batch's order (bit-identical;
 * DSV4_MM_WARP_VERIFY counts mismatches). e4m3 decodes through a shared
 * 256-entry table holding exactly e4m3()'s values. */
template<int F,int TB> __global__ void __launch_bounds__(256) mm_batch_w(const uint8_t *w,const uint8_t *sc,const float *x,float *y,int O,int I,int T){
    __shared__ float lut[256];
    lut[threadIdx.x]=e4m3((uint8_t)threadIdx.x);__syncthreads();
    int lane=threadIdx.x&31,o=blockIdx.x*8+(threadIdx.x>>5);if(o>=O)return;
    int tbase=blockIdx.y*TB,tcount=T-tbase<TB?T-tbase:TB;
    float p[TB][8];
    #pragma unroll
    for(int t=0;t<TB;t++)
        #pragma unroll
        for(int q=0;q<8;q++)p[t][q]=0.f;
    for(int c0=0;c0<I/32;c0+=8){
        #pragma unroll
        for(int q=0;q<8;q++){int i=lane+32*(c0+q);
            float v=lut[w[(long long)o*I+i]]*e8(sc[(o/128)*((I+127)/128)+i/128]);if constexpr(F==9)v=__bfloat162float(__float2bfloat16(v));
            #pragma unroll
            for(int t=0;t<TB;t++)if(t<tcount)p[t][q]+=x[(long long)(tbase+t)*I+i]*v;
        }
    }
    #pragma unroll
    for(int t=0;t<TB;t++){
        float v0=p[t][0]+p[t][4],v1=p[t][1]+p[t][5],v2=p[t][2]+p[t][6],v3=p[t][3]+p[t][7];
        v0=v0+v2;v1=v1+v3;float v=v0+v1;
        #pragma unroll
        for(int m=16;m;m>>=1)v+=__shfl_down_sync(0xffffffffu,v,m);
        if(!lane&&t<tcount)y[(long long)(tbase+t)*O+o]=v;
    }
}
static void run_mm_batch(Dsv4CudaTensor*t,const float*x,float*y,int T,cudaStream_t stream){
    /* DSV4_MM_WARP, for fp8 shapes whose I is a multiple of 256: unset, empty
     * or "auto" = warp kernel for short rows only (I <= 1024, e.g. attn.wq_b,
     * where mm_batch spends most of each block in its tree); 1 (any non-zero
     * number) = all of them; 0 = never. Kernel time against mm_batch on sm_86:
     * ~0.4-0.55x at I = 1024, ~0.8x at I = 2048 (shared w2, 4096x2048), ~1.1x
     * at I = 4096, ~1.65x at I = 8192; auto stays at I <= 1024 until I = 2048
     * is checked in the engine.
     * DSV4_MM_WARP_VERIFY=1 also runs mm_batch, keeps ITS results and counts
     * mismatching bits; over DSV4_MM_WARP=0 it checks the auto shapes (said
     * once at startup). */
    static int mw=-1,mw_env=1,mwv=0;static unsigned long long vcalls=0,vbad=0;
    if(mw<0){const char*e=getenv("DSV4_MM_WARP");mw_env=!e||!*e||!strcmp(e,"auto")?1:atoi(e)!=0?2:0;mw=mw_env;
        e=getenv("DSV4_MM_WARP_VERIFY");mwv=e&&atoi(e)!=0;
        if(mwv&&!mw){mw=1;fprintf(stderr,"[DSV4 CUDA] DSV4_MM_WARP_VERIFY=1 overrides DSV4_MM_WARP=0: the warp kernel runs beside mm_batch on the auto shapes, mm_batch's results are kept\n");}}
    if(mw&&(t->fmt==8||t->fmt==9)&&t->I%256==0&&(mw==2||t->I<=1024)){
        size_t n=(size_t)T*t->O;Dev*c=mwv?ctx(t->device):nullptr;
        /* verify buffers are per device; without them verify switches itself
         * off for good and DSV4_MM_WARP's own setting applies again (an
         * overridden 0 sends this call to mm_batch below) */
        if(mwv&&(!c||!buf((void**)&c->mmv,&c->mmvcap,n*sizeof(float))||
                 (!c->mmvcount&&!alloc_ok(cudaMalloc(&c->mmvcount,sizeof(*c->mmvcount)),"mm-warp verify counter")))){
            mwv=0;mw=mw_env;fprintf(stderr,"[DSV4 CUDA] DSV4_MM_WARP_VERIFY off: no memory for its buffers\n");}
        if(mw){
            dim3 wg((t->O+7)/8,(T+7)/8);float*dst=mwv?c->mmv:y;
            if(t->fmt==9)mm_batch_w<9,8><<<wg,256,0,stream>>>((uint8_t*)t->w,t->scale,x,dst,t->O,t->I,T);
            else mm_batch_w<8,8><<<wg,256,0,stream>>>((uint8_t*)t->w,t->scale,x,dst,t->O,t->I,T);
            if(!mwv)return;
            dim3 grid(t->O,(T+15)/16);
            if(t->fmt==9)mm_batch<9><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
            else mm_batch<8><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
            unsigned long long bad=0;cudaMemsetAsync(c->mmvcount,0,sizeof(*c->mmvcount),stream);
            count_bit_mismatch<<<(unsigned)((n+255)/256),256,0,stream>>>(y,c->mmv,(long long)n,c->mmvcount);
            cudaMemcpyAsync(&bad,c->mmvcount,sizeof(bad),cudaMemcpyDeviceToHost,stream);cudaStreamSynchronize(stream);
            vcalls++;vbad+=bad;
            if(bad||vcalls%500==0)fprintf(stderr,"mm-warp verify calls=%llu mismatches=%llu (this call %llu of %zu, %dx%d T=%d fmt=%d)\n",
                                          vcalls,vbad,bad,n,t->O,t->I,T,t->fmt);
            return;
        }
    }
    dim3 grid(t->O,(T+15)/16);
    if(t->fmt==4)mm_batch<4><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
    else if(t->fmt==9)mm_batch<9><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
    else mm_batch<8><<<grid,256,0,stream>>>((uint8_t*)t->w,t->scale,x,y,t->O,t->I,T);
}
#endif
__device__ int tc_sf_off(int outer,int inner,int sf_inner_dim){int base=((inner/4)*4+(outer/128)*sf_inner_dim)*128,o=outer&127;return base+(o%32)*16+(o/32)*4+(inner&3);}
__device__ uint8_t e4m3_code(float x){int best=0;float bd=INFINITY;for(int v=0;v<255;v++){float z=e4m3((uint8_t)v),d=fabsf(z-x);if(d<bd||(d==bd&&!(v&1)&&(best&1))){bd=d;best=v;}}return (uint8_t)best;}
__global__ void tc_expand(const uint8_t*q4,const uint8_t*src,uint8_t*q8,uint8_t*dst,int O,int I){
    const uint8_t lut[16]={0x00,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c,0x80,0xb0,0xb8,0xbc,0xc0,0xc4,0xc8,0xcc};long long n=(long long)O*I;
    for(long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;p<n;p+=(long long)gridDim.x*blockDim.x){uint8_t b=q4[p>>1];q8[p]=lut[(p&1)?b>>4:b&15];}
    int blocks=O*(I/32),sid=(I/32+3)&~3;for(int p=blockIdx.x*blockDim.x+threadIdx.x;p<blocks;p+=gridDim.x*blockDim.x){int o=p/(I/32),k=p%(I/32);dst[tc_sf_off(o,k,sid)]=src[p];}}
__global__ void tc_quant(const float*x,uint8_t*q,uint8_t*scale,int I){int b=blockIdx.x,i=b*128+threadIdx.x;__shared__ float mx[128];float v=i<I?fabsf(x[i]):0;mx[threadIdx.x]=v;__syncthreads();
    for(int n=64;n;n>>=1){if(threadIdx.x<n&&mx[threadIdx.x+n]>mx[threadIdx.x])mx[threadIdx.x]=mx[threadIdx.x+n];__syncthreads();}float raw=fmaxf(mx[0],1e-4f)/448.f;int e;frexpf(raw,&e);float s=ldexpf(1.f,e-1);if(s<raw)s*=2;int se;frexpf(s,&se);
    if(i<I)q[i]=e4m3_code(fmaxf(-448.f,fminf(448.f,x[i]/s)));if(threadIdx.x<4){int k=b*4+threadIdx.x;scale[tc_sf_off(0,k,(I/32+3)&~3)]=(uint8_t)(se-1+127);}}
__device__ __forceinline__ int compression_ready(const int *state,int ratio);
__global__ void mv_bf16(const __nv_bfloat16 *w,const float *x,float *y,int O,int I){int o=blockIdx.x;if(o>=O)return;float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x)sum+=x[i]*__bfloat162float(w[(long long)o*I+i]);
    __shared__ float r[256];r[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)r[threadIdx.x]+=r[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[o]=r[0];}
__global__ void mv_f32(const float *w,const float *x,float *y,int O,int I){int o=blockIdx.x;if(o>=O)return;float sum=0;
    for(int i=threadIdx.x;i<I;i+=blockDim.x)sum+=x[i]*w[(long long)o*I+i];
    __shared__ float r[256];r[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)r[threadIdx.x]+=r[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[o]=r[0];}
__global__ void mhc_fn_split(const float*w,const float*x,float*part,int N,int I,int parts){int p=blockIdx.x,o=blockIdx.y;if(p>=parts||o>=N)return;int begin=(long long)I*p/parts,end=(long long)I*(p+1)/parts;float sum=0,ss=0;
    for(int i=begin+threadIdx.x;i<end;i+=blockDim.x){float v=x[i];sum+=w[(long long)o*I+i]*v;if(!o)ss+=v*v;}__shared__ float r[256],q[256];r[threadIdx.x]=sum;q[threadIdx.x]=ss;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){r[threadIdx.x]+=r[threadIdx.x+n];q[threadIdx.x]+=q[threadIdx.x+n];}__syncthreads();}if(!threadIdx.x){part[o*parts+p]=r[0];if(!o)part[N*parts+p]=q[0];}}
__global__ void mhc_fn_finish(float*out,const float*part,const float*scale,const float*base,int N,int M,int I,float eps,int parts){__shared__ float inv;if(!threadIdx.x){float ss=0;for(int p=0;p<parts;p++)ss+=part[N*parts+p];inv=rsqrtf(ss/I+eps);}__syncthreads();
    int n=threadIdx.x;if(n<N){float v=0;for(int p=0;p<parts;p++)v+=part[n*parts+p];int group=n<M?0:n<2*M?1:2;out[n]=v*inv*scale[group]+base[n];}}
__global__ void mhc_post_fn_split(const float*w,const float*x,const float*residual,const float*state,float*out,float*part,int N,int M,int H,int parts){
    int p=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5;if(p>=parts)return;int I=M*H,begin=(long long)I*p/parts,end=(long long)I*(p+1)/parts;
    float acc[24]={},ss=0;const float*post=state,*comb=state+M;
    for(int i=begin+tid;i<end;i+=blockDim.x){int j=i/H,h=i-j*H;float v=post[j]*x[h];
        #pragma unroll
        for(int k=0;k<4;k++)v+=comb[k*M+j]*residual[(long long)k*H+h];
        v=__bfloat162float(__float2bfloat16(v));out[i]=v;ss+=v*v;
        #pragma unroll
        for(int n=0;n<24;n++)acc[n]+=w[(long long)n*I+i]*v;}
    __shared__ float warp_sum[25][8];
    #pragma unroll
    for(int n=0;n<24;n++){float v=acc[n];for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)warp_sum[n][warp]=v;}
    for(int d=16;d;d>>=1)ss+=__shfl_down_sync(0xffffffff,ss,d);if(!lane)warp_sum[24][warp]=ss;__syncthreads();
    if(!warp){if(lane<25){float v=0;for(int q=0;q<8;q++)v+=warp_sum[lane][q];if(lane<24)part[lane*parts+p]=v;else part[24*parts+p]=v;}}
}
__global__ void mhc_fn_split_batch(const float*w,const float*x,float*part,int tokens,int N,int I,int parts){int t=blockIdx.z,p=blockIdx.x,o=blockIdx.y;if(t>=tokens||p>=parts||o>=N)return;const float*xt=x+(long long)t*I;float*out=part+(long long)t*(N+1)*parts;int begin=(long long)I*p/parts,end=(long long)I*(p+1)/parts;float sum=0,ss=0;
    for(int i=begin+threadIdx.x;i<end;i+=blockDim.x){float v=xt[i];sum+=w[(long long)o*I+i]*v;if(!o)ss+=v*v;}__shared__ float r[256],q[256];r[threadIdx.x]=sum;q[threadIdx.x]=ss;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n){r[threadIdx.x]+=r[threadIdx.x+n];q[threadIdx.x]+=q[threadIdx.x+n];}__syncthreads();}if(!threadIdx.x){out[o*parts+p]=r[0];if(!o)out[N*parts+p]=q[0];}}
__global__ void mhc_fn_finish_batch(float*mix,const float*part,const float*scale,const float*base,int N,int I,float eps,int parts){int t=blockIdx.x;const float*src=part+(long long)t*(N+1)*parts;float*dst=mix+(long long)t*N;__shared__ float inv;if(!threadIdx.x){float ss=0;for(int p=0;p<parts;p++)ss+=src[N*parts+p];inv=rsqrtf(ss/I+eps);}__syncthreads();int n=threadIdx.x;if(n<N){float v=0;for(int p=0;p<parts;p++)v+=src[n*parts+p];dst[n]=v*inv*scale[n<4?0:n<8?1:2]+base[n];}}
__global__ void mhc_coeff_batch(float*post,float*comb,const float*mix){int t=blockIdx.x,lane=threadIdx.x&31;const float*m=mix+(long long)t*24;if(lane<4)post[(long long)t*4+lane]=(1.f/(1.f+expf(-m[4+lane])))*2.f;float v=lane<16?m[8+lane]:-INFINITY,mx=v;mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));v=lane<16?expf(v-mx):0.f;float sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v=v/sum+1e-6f;for(int it=0;it<20;it++){if(it){sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v/=sum+1e-6f;}sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,4);sum+=__shfl_xor_sync(0xffffffff,sum,8);if(lane<16)v/=sum+1e-6f;}if(lane<16)comb[(long long)t*16+lane]=v;}
__global__ void mhc_input_norm_batch(float*out,const float*residual,const float*mix,const float*w,int H){int t=blockIdx.x;const float*pre=mix+(long long)t*24,*r=residual+(long long)t*4*H;float*dst=out+(long long)t*H;__shared__ float ws[8],inv;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int h=threadIdx.x;h<H;h+=blockDim.x){float v=0;for(int i=0;i<4;i++)v+=(1.f/(1.f+expf(-pre[i]))+1e-6f)*r[(long long)i*H+h];v=__bfloat162float(__float2bfloat16(v));dst[h]=v;ss+=v*v;}for(int d=16;d;d>>=1)ss+=__shfl_down_sync(0xffffffff,ss,d);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);if(!lane)inv=rsqrtf(v/H+1e-6f);}__syncthreads();for(int h=threadIdx.x;h<H;h+=blockDim.x)dst[h]=__bfloat162float(__float2bfloat16(dst[h]*inv*w[h]));}
__global__ void mhc_input_batch(float*out,const float*residual,const float*mix,int H){int t=blockIdx.x;const float*pre=mix+(long long)t*24,*r=residual+(long long)t*4*H;float*dst=out+(long long)t*H;for(int h=threadIdx.x;h<H;h+=blockDim.x){float v=0;for(int i=0;i<4;i++)v+=(1.f/(1.f+expf(-pre[i]))+1e-6f)*r[(long long)i*H+h];dst[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void mhc_output_batch(float*out,const float*x,const float*residual,const float*post,const float*comb,int H){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;int t=p/(4LL*H),j=(p/H)&3,h=p%H;float v=post[(long long)t*4+j]*x[(long long)t*H+h];for(int k=0;k<4;k++)v+=comb[(long long)t*16+k*4+j]*residual[((long long)t*4+k)*H+h];out[p]=__bfloat162float(__float2bfloat16(v));}
__global__ void expert_act(float *a,const float *g,const float *u,float weight,float limit,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
    float gg=fminf(__bfloat162float(__float2bfloat16(g[i])),limit);
    float uu=fmaxf(-limit,fminf(__bfloat162float(__float2bfloat16(u[i])),limit));
    a[i]=__bfloat162float(__float2bfloat16(weight*(gg/(1.f+expf(-gg)))*uu));}}
__global__ void expert_act_grouped(float *a,const float *g,const float *u,const float *weight,float limit,int rows,int n){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<rows*n){int row=p/n;float gg=fminf(__bfloat162float(__float2bfloat16(g[p])),limit);
        float uu=fmaxf(-limit,fminf(__bfloat162float(__float2bfloat16(u[p])),limit));
        a[p]=__bfloat162float(__float2bfloat16(weight[row]*(gg/(1.f+expf(-gg)))*uu));}}
__device__ float e4m3_round(float x){int best=0;float bd=INFINITY;for(int v=0;v<255;v++){float q=e4m3((uint8_t)v);if(isnan(q))continue;
    float d=fabsf(q-x);if(d<bd||(d==bd&&!(v&1)&&(best&1))){bd=d;best=v;}}return e4m3((uint8_t)best);}
__global__ void fp8_sim(float *x,int rows,int n){int row=blockIdx.x/((n+127)/128),b=blockIdx.x%((n+127)/128),i=b*128+threadIdx.x;if(row>=rows)return;
    float v=i<n?fabsf(x[(long long)row*n+i]):0.f;__shared__ float mx[128];mx[threadIdx.x]=v;__syncthreads();
    for(int k=64;k;k>>=1){if(threadIdx.x<k&&mx[threadIdx.x+k]>mx[threadIdx.x])mx[threadIdx.x]=mx[threadIdx.x+k];__syncthreads();}
    if(i<n){float m=fmaxf(mx[0],1e-4f),raw=m/448.f;int e;frexpf(raw,&e);float scale=ldexpf(1.f,e-1);if(scale<raw)scale*=2.f;
        float q=fmaxf(-448.f,fminf(448.f,x[(long long)row*n+i]/scale));x[(long long)row*n+i]=__bfloat162float(__float2bfloat16(e4m3_round(q)*scale));}}
__global__ void fp8_sim64(float *x,int n){int i=blockIdx.x*64+threadIdx.x;__shared__ float mx[64];float v=i<n?fabsf(x[i]):0;mx[threadIdx.x]=v;__syncthreads();
    for(int k=32;k;k>>=1){if(threadIdx.x<k&&mx[threadIdx.x+k]>mx[threadIdx.x])mx[threadIdx.x]=mx[threadIdx.x+k];__syncthreads();}if(i<n){float m=fmaxf(mx[0],1e-4f),raw=m/448.f;int e;frexpf(raw,&e);float scale=ldexpf(1.f,e-1);uint8_t q=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[i]/scale)),__NV_SATFINITE,__NV_E4M3);x[i]=__bfloat162float(__float2bfloat16(e4m3(q)*scale));}}
/* Batched router (generic CUDA; used by both builds). */
__device__ __forceinline__ bool route_better(float,unsigned,float,unsigned);__device__ __forceinline__ float route_prob(float);
__global__ void route_logits_batch(const float*w,const float*x,float*y,int tokens,int E,int H){int e=blockIdx.x% E,t=blockIdx.x/E;float sum=0;const float*in=x+(long long)t*H;for(int h=threadIdx.x;h<H;h+=blockDim.x)sum+=w[(long long)e*H+h]*in[h];__shared__ float s[256];s[threadIdx.x]=sum;__syncthreads();for(int n=128;n;n>>=1){if(threadIdx.x<n)s[threadIdx.x]+=s[threadIdx.x+n];__syncthreads();}if(!threadIdx.x)y[(long long)t*E+e]=s[0];(void)tokens;}
__global__ void route_top6_batch(int*ids,float*weights,const float*bias,const float*logits,int tokens,float scale){int t=blockIdx.x,lane=threadIdx.x;if(t>=tokens||lane>=32)return;const float*src=logits+(long long)t*256;float prob[8],score[8];for(int j=0;j<8;j++){int e=lane+j*32;prob[j]=route_prob(src[e]);score[j]=prob[j]+(bias?bias[e]:0.f);}float outp[6]={};unsigned outi[6]={};for(int k=0;k<6;k++){float bs=-INFINITY,bp=0;unsigned bi=UINT_MAX;for(int j=0;j<8;j++){unsigned e=lane+j*32;if(route_better(score[j],e,bs,bi)){bs=score[j];bp=prob[j];bi=e;}}for(int m=16;m;m>>=1){float os=__shfl_xor_sync(0xffffffff,bs,m),op=__shfl_xor_sync(0xffffffff,bp,m);unsigned oi=__shfl_xor_sync(0xffffffff,bi,m);if(route_better(os,oi,bs,bi)){bs=os;bp=op;bi=oi;}}for(int j=0;j<8;j++)if((unsigned)(lane+j*32)==bi)score[j]=-INFINITY;if(!lane){outp[k]=bp;outi[k]=bi;}}if(!lane){float sum=0;for(int k=0;k<6;k++)sum+=outp[k];sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++){ids[(long long)t*6+k]=(int)outi[k];weights[(long long)t*6+k]=outp[k]/sum*scale;}}}
#ifdef COLI_DSV4_DEEPGEMM
__global__ void dg_pack_weight_scale(int*out,const uint8_t*in,int expert,int N,int groups){
    int p=blockIdx.x*blockDim.x+threadIdx.x,total=N*(groups/4);if(p>=total)return;int n=p/(groups/4),kg=p%(groups/4);uint32_t v=0;
    for(int b=0;b<4;b++)v|=(uint32_t)in[(long long)n*groups+kg*4+b]<<(8*b);out[((long long)expert*(groups/4)+kg)*N+n]=(int)v;
}
__global__ void dg_pack_dense_scale(int*out,const uint8_t*in,int N,int groups){int p=blockIdx.x*blockDim.x+threadIdx.x,total=N*(groups/4);if(p>=total)return;
    int n=p/(groups/4),kg=p%(groups/4);uint32_t v=0;for(int b=0;b<4;b++)v|=(uint32_t)in[(long long)(n/128)*groups+kg*4+b]<<(8*b);out[(long long)kg*N+n]=(int)v;}
__global__ void dg_pack_batched_scale(int*out,const uint8_t*in,int N,int groups,int batches){int p=blockIdx.x*blockDim.x+threadIdx.x,per=N*(groups/4),total=batches*per;if(p>=total)return;
    int batch=p/per,r=p%per,n=r/(groups/4),kg=r%(groups/4);uint32_t v=0;for(int b=0;b<4;b++)v|=(uint32_t)in[(long long)(batch*N+n)/128*groups+kg*4+b]<<(8*b);out[((long long)batch*(groups/4)+kg)*N+n]=(int)v;}
__device__ __forceinline__ uint8_t dg_scale_code(float m){float raw=fmaxf(m,1e-4f)/448.f;int e;frexpf(raw,&e);float s=ldexpf(1.f,e-1);if(s<raw){s*=2.f;e++;}return (uint8_t)(e-1+127);}
__device__ __forceinline__ float dg_scale_value(uint8_t code){return ldexpf(1.f,(int)code-127);}
__global__ void dg_quant_dense(const float*x,uint8_t*q,int*sfa,int K){int pack=blockIdx.x;if(pack>=K/512)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[pack*4]=(int)codes;}
__global__ void dg_quant_dense_batch(const float*x,uint8_t*q,int*sfa,int rows,int aligned_rows,int K){int row=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(row>=rows)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[(long long)row*K+k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[(long long)row*K+k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[(long long)pack*aligned_rows+row]=(int)codes;}
__global__ void dg_quant_batched8(const float*x,uint8_t*q,int*sfa,int K){int batch=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(batch>=8)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[(long long)batch*K+k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)batch*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[(long long)batch*K+k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[((long long)batch*(K/512)+pack)*4]=(int)codes;}
__global__ void dg_bf16_to_float(float*out,const __nv_bfloat16*in,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(in[i]);}
__global__ void dg_float_to_bf16(__nv_bfloat16*out,const float*in,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__float2bfloat16(in[i]);}
__global__ void dg_argmax_bf16(const __nv_bfloat16*x,int n,int*out_id,float*out_value){__shared__ float value[256];__shared__ int id[256];float best=-INFINITY;int bi=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float v=__bfloat162float(x[i]);if(v>best||(v==best&&i<bi)){best=v;bi=i;}}value[threadIdx.x]=best;id[threadIdx.x]=bi;__syncthreads();
    for(int k=128;k;k>>=1){if(threadIdx.x<k){float v=value[threadIdx.x+k];int i=id[threadIdx.x+k];if(v>value[threadIdx.x]||(v==value[threadIdx.x]&&i<id[threadIdx.x])){value[threadIdx.x]=v;id[threadIdx.x]=i;}}__syncthreads();}if(!threadIdx.x){*out_id=id[0];*out_value=value[0];}}
__global__ void dg_layout6(int*out,const int*ids,int rows,int align){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows)out[i]=ids[i/align];}
__global__ void dg_layout6_ep(int*out,const int*ids,int rows,int align,int base){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows){int id=ids[i/align]-base;out[i]=id>=0&&id<128?id:-1;}}
__global__ void dg_quant_replicate(const float*x,uint8_t*q,int*sfa,int K){int row=blockIdx.x;if(row>=768)return;__shared__ float mx[128];uint32_t pack=0;int groups=K/128;
    for(int g=0;g<groups;g++){int k=g*128+threadIdx.x;float v=fabsf(x[k]);mx[threadIdx.x]=v;__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);
        if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*768+row]=(int)pack;pack=0;}}__syncthreads();}
}
__global__ void dg_quant_one(const float*x,uint8_t*q,int*sfa,int K,int rows){int pack=blockIdx.x;if(pack>=K/512)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[(long long)pack*rows]=(int)codes;}
__global__ void dg_replicate_rows(uint8_t*q,int*sfa,int K,int rows){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)rows*K;if(p<n){int k=p%K;q[p]=q[k];}
    int z=(int)p,total=rows*(K/512);if(z<total){int row=z/(K/512),pack=z%(K/512);sfa[(long long)pack*rows+row]=sfa[(long long)pack*rows];}}
__global__ void dg_swiglu_quant(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit){int row=blockIdx.x;if(row>=768)return;__shared__ float mx[128],value[128];uint32_t pack=0;int expert=row/128;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16(weight[expert]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();
        for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);
        q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);
        if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*768+row]=(int)pack;pack=0;}}__syncthreads();}
}
__global__ void dg_swiglu_quant_unique(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit,int rows,int align){int expert=blockIdx.x;if(expert>=6)return;int row=expert*align;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16(weight[expert]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_swiglu_quant_unique_ep(const __nv_bfloat16*in,uint8_t*q,int*sfa,const int*ids,int base,float limit,int rows,int align){int slot=blockIdx.x;if(slot>=6)return;int id=ids[slot]-base;if(id<0||id>=128)return;int row=slot*align;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_swiglu_quant_unique_tp2(const __nv_bfloat16*in,uint8_t*q,int*sfa,float limit,int rows,int align){int slot=blockIdx.x;if(slot>=6)return;int row=slot*align;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<8;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*2048+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*2048+1024+k]),limit));
        float v=__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*1024+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_replicate_expert_rows(uint8_t*q,int*sfa,int K,int rows,int align){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)rows*K;if(p<n){int row=p/K,k=p%K,base=(row/align)*align;q[p]=q[(long long)base*K+k];}
    int z=(int)p,total=rows*(K/512);if(z<total){int row=z/(K/512),pack=z%(K/512),base=(row/align)*align;sfa[(long long)pack*rows+row]=sfa[(long long)pack*rows+base];}}
__global__ void dg_reduce6(float*out,const __nv_bfloat16*parts,int align){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++)v+=__bfloat162float(parts[(long long)e*align*4096+h]);out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_reduce6_ep(float*out,const __nv_bfloat16*parts,const int*ids,const float*weights,int base,int align){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++){int id=ids[e]-base;if(id>=0&&id<128)v+=weights[e]*__bfloat162float(parts[(long long)e*align*4096+h]);}out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_mask6(int*mask,const int*ids,int groups){int i=threadIdx.x;if(i<groups)mask[i]=0;__syncthreads();if(i<6&&ids[i]>=0)mask[ids[i]]=1;}
__global__ void dg_quant_masked(const float*x,uint8_t*q,int*sfa,const int*ids,int K,int maxm){int slot=blockIdx.x/(K/512),pack=blockIdx.x%(K/512),id=ids[slot];if(id<0)return;__shared__ float mx[128];uint32_t codes=0;
    for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;mx[threadIdx.x]=fabsf(x[k]);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[((long long)id*maxm)*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,x[k]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}
    if(!threadIdx.x)sfa[((long long)id*(K/512)+pack)*maxm]=(int)codes;}
__global__ void dg_swiglu_quant_masked(const __nv_bfloat16*in,uint8_t*q,int*sfa,const int*ids,const float*weight,float limit,int maxm){int slot=blockIdx.x,id=ids[slot];if(id<0)return;long long row=(long long)id*maxm;__shared__ float mx[128],value[128];uint32_t pack=0;
    for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[row*4096+k]),limit);float up=fmaxf(-limit,fminf(__bfloat162float(in[row*4096+2048+k]),limit));
        float v=__bfloat162float(__float2bfloat16(weight[slot]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}
        uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[((long long)id*4+g/4)*maxm]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_reduce6_masked(float*out,const __nv_bfloat16*parts,const int*ids,int maxm){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++)if(ids[e]>=0)v+=__bfloat162float(parts[((long long)ids[e]*maxm)*4096+h]);out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_swiglu_quant_rows(const __nv_bfloat16*in,uint8_t*q,int*sfa,float limit,int rows){int row=blockIdx.x,warp=threadIdx.x>>5,lane=threadIdx.x&31;__shared__ uint8_t code[16];
    for(int phase=0;phase<2;phase++){int group=phase*8+warp,base=group*128;float value[4],mx=0;
        #pragma unroll
        for(int j=0;j<4;j++){int k=base+lane+j*32;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit),up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));value[j]=__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up));mx=fmaxf(mx,fabsf(value[j]));}
        for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));uint8_t c=dg_scale_code(mx),cc=__shfl_sync(0xffffffff,c,0);float scale=dg_scale_value(cc);if(!lane)code[group]=cc;
        #pragma unroll
        for(int j=0;j<4;j++){int k=base+lane+j*32;q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[j]/scale)),__NV_SATFINITE,__NV_E4M3);}}
    __syncthreads();if(threadIdx.x<4){int g=threadIdx.x*4;sfa[(long long)threadIdx.x*rows+row]=(int)((uint32_t)code[g]|((uint32_t)code[g+1]<<8)|((uint32_t)code[g+2]<<16)|((uint32_t)code[g+3]<<24));}}
__global__ void dg_reduce6_weighted(float*out,const __nv_bfloat16*parts,const float*weight,int align){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<4096){float v=0;for(int e=0;e<6;e++)v+=weight[e]*__bfloat162float(parts[(long long)e*align*4096+h]);out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void route_hash_batch(int*ids,const int64_t*map,const int*tokens,int rows){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<rows*6)ids[p]=(int)map[(long long)tokens[p/6]*6+p%6];}
__global__ void route_fixed_weights_batch(float*weights,const int*ids,const float*logits,int tokens,float scale){int t=blockIdx.x,k=threadIdx.x;__shared__ float p[6];if(t<tokens&&k<6)p[k]=route_prob(logits[(long long)t*256+ids[(long long)t*6+k]]);__syncthreads();if(t<tokens&&!k){float sum=0;for(int i=0;i<6;i++)sum+=p[i];sum=fmaxf(sum,6.103515625e-5f);for(int i=0;i<6;i++)weights[(long long)t*6+i]=p[i]/sum*scale;}}
__global__ void dg_layout_routes(int*out,const int*ids,int rows,int align){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows)out[i]=ids[i/align];}
__global__ void dg_quant_routes(const float*x,uint8_t*q,int*sfa,int tokens,int K,int rows,int align){int route=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(route>=tokens*6)return;int token=route/6,row=route*align;__shared__ float mx[128];uint32_t codes=0;for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;float v=x[(long long)token*K+k];mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,v/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}if(!threadIdx.x)sfa[(long long)pack*rows+row]=(int)codes;}
__global__ void dg_swiglu_quant_routes(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit,int routes,int rows,int align){int route=blockIdx.x;if(route>=routes)return;int row=route*align;__shared__ float mx[128],value[128];uint32_t pack=0;for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit),up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));float v=__bfloat162float(__float2bfloat16(weight[route]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
/* EXPERT-GROUPED variants: routes are laid out contiguously per expert (host
 * builds rowmap[route] -> row and layout[row] -> expert, groups 64-aligned,
 * pad rows zeroed). Each route's row computes exactly what its old 64x
 * replicated row did, so results are bitwise the replicated layout with
 * ~4x fewer GEMM rows. */
__global__ void dg_quant_routes_map(const float*x,uint8_t*q,int*sfa,int tokens,int K,int rows,const int*rowmap){int route=blockIdx.x/(K/512),pack=blockIdx.x%(K/512);if(route>=tokens*6)return;int token=route/6,row=rowmap[route];__shared__ float mx[128];uint32_t codes=0;for(int b=0;b<4;b++){int k=(pack*4+b)*128+threadIdx.x;float v=x[(long long)token*K+k];mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*K+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,v/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x)codes|=(uint32_t)code<<(8*b);__syncthreads();}if(!threadIdx.x)sfa[(long long)pack*rows+row]=(int)codes;}
__global__ void dg_swiglu_quant_routes_map(const __nv_bfloat16*in,uint8_t*q,int*sfa,const float*weight,float limit,int routes,int rows,const int*rowmap){int route=blockIdx.x;if(route>=routes)return;int row=rowmap[route];__shared__ float mx[128],value[128];uint32_t pack=0;for(int g=0;g<16;g++){int k=g*128+threadIdx.x;float gate=fminf(__bfloat162float(in[(long long)row*4096+k]),limit),up=fmaxf(-limit,fminf(__bfloat162float(in[(long long)row*4096+2048+k]),limit));float v=__bfloat162float(__float2bfloat16(weight[route]*__bfloat162float(__float2bfloat16((gate/(1.f+expf(-gate)))*up))));value[threadIdx.x]=v;mx[threadIdx.x]=fabsf(v);__syncthreads();for(int n=64;n;n>>=1){if(threadIdx.x<n)mx[threadIdx.x]=fmaxf(mx[threadIdx.x],mx[threadIdx.x+n]);__syncthreads();}uint8_t code=dg_scale_code(mx[0]);float scale=dg_scale_value(code);q[(long long)row*2048+k]=__nv_cvt_float_to_fp8(fmaxf(-448.f,fminf(448.f,value[threadIdx.x]/scale)),__NV_SATFINITE,__NV_E4M3);if(!threadIdx.x){pack|=(uint32_t)code<<(8*(g&3));if((g&3)==3){sfa[(long long)(g/4)*rows+row]=(int)pack;pack=0;}}__syncthreads();}}
__global__ void dg_reduce_routes_map(float*out,const __nv_bfloat16*parts,int tokens,const int*rowmap){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*4096;if(p<n){int t=p/4096,h=p%4096;float v=0;for(int k=0;k<6;k++)v+=__bfloat162float(parts[(long long)rowmap[t*6+k]*4096+h]);out[p]=__bfloat162float(__float2bfloat16(v));}}
__global__ void dg_reduce_routes(float*out,const __nv_bfloat16*parts,int tokens,int align){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*4096;if(p<n){int t=p/4096,h=p%4096;float v=0;for(int k=0;k<6;k++)v+=__bfloat162float(parts[((long long)(t*6+k)*align)*4096+h]);out[p]=__bfloat162float(__float2bfloat16(v));}}
__global__ void swiglu_rows(float*out,const float*gate,const float*up,float limit,int rows,int I){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)rows*I;if(p<n){float g=fminf(gate[p],limit),u=fmaxf(-limit,fminf(up[p],limit));out[p]=__bfloat162float(__float2bfloat16((g/(1.f+expf(-g)))*u));}}
__global__ void add_rows(float*out,const float*x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(__float2bfloat16(out[i]+x[i]));}
__global__ void dg_compare(const float*ref,const float*got,float*stats,int n){__shared__ float ma[256],sa[256];float mx=0,sum=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x){float d=fabsf(ref[i]-got[i]);mx=fmaxf(mx,d);sum+=d;}ma[threadIdx.x]=mx;sa[threadIdx.x]=sum;__syncthreads();
    for(int k=128;k;k>>=1){if(threadIdx.x<k){ma[threadIdx.x]=fmaxf(ma[threadIdx.x],ma[threadIdx.x+k]);sa[threadIdx.x]+=sa[threadIdx.x+k];}__syncthreads();}
    if(!threadIdx.x){stats[0]=ma[0];stats[1]=sa[0]/n;}}
__global__ void dg_compare_fc1(const float*gate,const float*up,const __nv_bfloat16*got,float*stats){__shared__ float ma[256],sa[256];float mx=0,sum=0;
    for(int i=threadIdx.x;i<6*4096;i+=blockDim.x){int e=i/4096,n=i%4096;float ref=n<2048?gate[(long long)e*2048+n]:up[(long long)e*2048+n-2048];float d=fabsf(ref-__bfloat162float(got[(long long)e*128*4096+n]));mx=fmaxf(mx,d);sum+=d;}
    ma[threadIdx.x]=mx;sa[threadIdx.x]=sum;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k){ma[threadIdx.x]=fmaxf(ma[threadIdx.x],ma[threadIdx.x+k]);sa[threadIdx.x]+=sa[threadIdx.x+k];}__syncthreads();}if(!threadIdx.x){stats[0]=ma[0];stats[1]=sa[0]/(6*4096);}}
__global__ void dg_compare_fc2(const float*ref,const __nv_bfloat16*got,float*stats){__shared__ float ma[256],sa[256];float mx=0,sum=0;
    for(int i=threadIdx.x;i<6*4096;i+=blockDim.x){int e=i/4096,n=i%4096;float d=fabsf(ref[i]-__bfloat162float(got[(long long)e*128*4096+n]));mx=fmaxf(mx,d);sum+=d;}
    ma[threadIdx.x]=mx;sa[threadIdx.x]=sum;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k){ma[threadIdx.x]=fmaxf(ma[threadIdx.x],ma[threadIdx.x+k]);sa[threadIdx.x]+=sa[threadIdx.x+k];}__syncthreads();}if(!threadIdx.x){stats[0]=ma[0];stats[1]=sa[0]/(6*4096);}}

static int dg_driver(CUresult e,const char*what){if(e==CUDA_SUCCESS)return 1;const char*msg=nullptr;cuGetErrorString(e,&msg);fprintf(stderr,"[DSV4 DeepGEMM] %s: %s\n",what,msg?msg:"CUDA driver error");return 0;}
static int dg_map(CUtensorMap*map,CUtensorMapDataType type,void*ptr,unsigned long long inner,unsigned long long outer,unsigned long long stride,unsigned box_inner,unsigned box_outer,CUtensorMapSwizzle swizzle){
    const cuuint64_t dims[2]={inner,outer},strides[1]={stride};const cuuint32_t box[2]={box_inner,box_outer},elem[2]={1,1};
    CUresult e=cuTensorMapEncodeTiled(map,type,2,ptr,dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(e!=CUDA_SUCCESS)fprintf(stderr,"[DSV4 DeepGEMM] bad TMA type=%d ptr=%p dims=%llux%llu stride=%llu box=%ux%u swizzle=%d\n",(int)type,ptr,inner,outer,stride,box_inner,box_outer,(int)swizzle);
    return dg_driver(e,"encode TMA map");
}
static int dg_map_batched(CUtensorMap*map,CUtensorMapDataType type,void*ptr,unsigned long long inner,unsigned long long outer,unsigned long long batches,unsigned long long outer_stride,unsigned long long batch_stride,unsigned box_inner,unsigned box_outer,CUtensorMapSwizzle swizzle){
    const cuuint64_t dims[3]={inner,outer,batches},strides[2]={outer_stride,batch_stride};const cuuint32_t box[3]={box_inner,box_outer,1},elem[3]={1,1,1};
    return dg_driver(cuTensorMapEncodeTiled(map,type,3,ptr,dims,strides,box,elem,CU_TENSOR_MAP_INTERLEAVE_NONE,swizzle,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),"encode batched TMA map");
}
using DgOutput=cutlass::bfloat16_t;using DgEpilogue=deep_gemm::epilogue::transform::EpilogueIdentity;
static int dg_dense_2048x4096(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){constexpr int O=2048,I=4096;
    if(!t||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,I)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,(I/512)*4*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||!buf((void**)&c->dgdensews,&c->dgdensewscap,(size_t)4*O*sizeof(float)))return 0;
    dg_quant_dense<<<I/512,128,0,c->stream>>>(x,c->dga1,c->dgsfa1,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,64,64,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,1,I,64,16,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,I/512,16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,16,64,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,88320),"dense DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,88320,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,c->dgdensews,O,1,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    auto reduce=deep_gemm::sm120_split_k_reduce_impl<DgOutput,4>;reduce<<<(O+255)/256,256,0,c->stream>>>((DgOutput*)c->dgmm1,c->dgdensews,O,1,1,O);
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);
    static int dumped;if(!dumped++&&getenv("DSV4_CUDA_DENSE_DUMP")){struct Item{const char*p;const void*d;size_t n;}items[]={{"/tmp/dsv4-dense-x.f32",x,I*4},{"/tmp/dsv4-dense-q.fp8",c->dga1,I},{"/tmp/dsv4-dense-sfa.bin",c->dgsfa1,(I/512)*4*sizeof(int)},{"/tmp/dsv4-dense-sfb.bin",t->dg_scale,(size_t)O*(I/512)*sizeof(int)},{"/tmp/dsv4-dense-out.f32",y,O*4}};
        if(!ok(cudaStreamSynchronize(c->stream),"dense dump sync"))return 0;for(unsigned z=0;z<sizeof(items)/sizeof(*items);z++){void*h=malloc(items[z].n);FILE*f;if(!h||!ok(cudaMemcpy(h,items[z].d,items[z].n,cudaMemcpyDeviceToHost),"dense dump copy")||!(f=fopen(items[z].p,"wb"))||fwrite(h,1,items[z].n,f)!=items[z].n){free(h);return 0;}fclose(f);free(h);}}
    return ok(cudaGetLastError(),"dense DeepGEMM launch");}
static int dg_dense_4096x2048(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){constexpr int O=4096,I=2048;
    if(!t||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,I)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,(I/512)*4*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||!buf((void**)&c->dgdensews,&c->dgdensewscap,(size_t)4*O*sizeof(float)))return 0;
    dg_quant_dense<<<I/512,128,0,c->stream>>>(x,c->dga1,c->dgsfa1,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,64,64,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,1,I,64,16,CU_TENSOR_MAP_SWIZZLE_64B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,I/512,16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,16,64,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,64,16,64,64,64,0,16,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,64,4>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,88320),"dense DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,88320,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,c->dgdensews,O,1,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    auto reduce=deep_gemm::sm120_split_k_reduce_impl<DgOutput,4>;reduce<<<(O+255)/256,256,0,c->stream>>>((DgOutput*)c->dgmm1,c->dgdensews,O,1,1,O);
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);return ok(cudaGetLastError(),"dense DeepGEMM launch");}
template<int O,int I,int BM,int BK,int SW,int STAGES,int EPI,int SPLIT,bool QUANT=true>
static int dg_dense_attention(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){
    constexpr int SMEM=SW==128?95872:98816;
    constexpr CUtensorMapSwizzle SWIZZLE=SW==128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B;
    if(!t||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,I)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,(I/512)*4*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||
       (SPLIT>1&&!buf((void**)&c->dgdensews,&c->dgdensewscap,(size_t)SPLIT*O*sizeof(float)))){fprintf(stderr,"[DSV4 DG] attention buffer/shape failed %dx%d\n",O,I);return 0;}
    if constexpr(QUANT)dg_quant_dense<<<I/512,128,0,c->stream>>>(x,c->dga1,c->dgsfa1,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,BK,BM,SWIZZLE)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,1,I,BK,16,SWIZZLE)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,BM,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,I/512,16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,16,BM,CU_TENSOR_MAP_SWIZZLE_NONE)){fprintf(stderr,"[DSV4 DG] attention TMA map failed %dx%d\n",O,I);return 0;}
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,BM,16,BK,SW,SW,0,STAGES,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,SPLIT>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,I,128,128,1,BM,16,BK,SW,SW,0,STAGES,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,SPLIT>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM),"attention DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,SMEM,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,SPLIT>1?c->dgdensews:nullptr,O,1,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if constexpr(SPLIT>1){auto reduce=deep_gemm::sm120_split_k_reduce_impl<DgOutput,SPLIT>;reduce<<<(O+255)/256,256,0,c->stream>>>((DgOutput*)c->dgmm1,c->dgdensews,O,1,1,O);}
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);return ok(cudaGetLastError(),"attention DeepGEMM launch");
}
template<int O,int I,int BM,int BN,int BK,int SW,int STAGES,int EPI>
static int dg_dense_batch(Dev*c,Dsv4CudaTensor*t,const float*x,float*y,int tokens){
    constexpr int SMEM=88320;int aligned_tokens=(tokens+BN-1)/BN*BN;constexpr CUtensorMapSwizzle SWIZZLE=SW==128?CU_TENSOR_MAP_SWIZZLE_128B:CU_TENSOR_MAP_SWIZZLE_64B;
    if(!t||tokens<1||t->O!=O||t->I!=I||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,(size_t)tokens*I)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,(size_t)aligned_tokens*(I/512)*sizeof(int))||
       !buf((void**)&c->dgmm1,&c->dgmm1cap,(size_t)tokens*O*sizeof(__nv_bfloat16)))return 0;
    dg_quant_dense_batch<<<tokens*(I/512),128,0,c->stream>>>(x,c->dga1,c->dgsfa1,tokens,aligned_tokens,I);CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,I,O,I,BK,BM,SWIZZLE)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,I,tokens,I,BK,BN,SWIZZLE)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,O,I/512,(unsigned long long)O*4,BM,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,aligned_tokens,I/512,(unsigned long long)aligned_tokens*4,BN,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,tokens,(unsigned long long)O*2,16,BM,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,0,I,128,128,1,BM,BN,BK,SW,SW,0,STAGES,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,0,I,128,128,1,BM,BN,BK,SW,SW,0,STAGES,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,false,false,false,true,false,EPI,1>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM),"batched dense DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,SMEM,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,nullptr,O,tokens,I,1,O,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    dg_bf16_to_float<<<((long long)tokens*O+255)/256,256,0,c->stream>>>(y,c->dgmm1,tokens*O);return ok(cudaGetLastError(),"batched dense DeepGEMM launch");
}
static int dg_dense_batch_dispatch(Dev*c,Dsv4CudaTensor*t,const float*x,float*y,int tokens){
    if(t->O==1536&&t->I==4096)return dg_dense_batch<1536,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==1024&&t->I==4096)return dg_dense_batch<1024,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==512&&t->I==4096)return dg_dense_batch<512,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==2048&&t->I==4096)return dg_dense_batch<2048,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==4096)return dg_dense_batch<4096,4096,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==8192)return dg_dense_batch<4096,8192,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==1024)return dg_dense_batch<4096,1024,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==4096&&t->I==2048)return dg_dense_batch<4096,2048,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==16384&&t->I==1024)return dg_dense_batch<16384,1024,64,16,64,64,16,64>(c,t,x,y,tokens);
    if(t->O==32768&&t->I==1024)return dg_dense_batch<32768,1024,64,16,64,64,16,64>(c,t,x,y,tokens);
    return 0;
}
static int dg_attention_kv_reuse(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){(void)x;return t&&t->O==512&&t->I==4096&&dg_dense_attention<512,4096,64,128,128,9,64,4,false>(c,t,x,y);}
static int dg_attention_projection(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){
    int good=0;
    if(t->I==4096&&(t->O==1536||t->O==1024||t->O==512)){
        good=t->O==1536?dg_dense_attention<1536,4096,64,128,128,9,64,4>(c,t,x,y):
             t->O==1024?dg_dense_attention<1024,4096,64,128,128,9,64,4>(c,t,x,y):
                         dg_dense_attention<512,4096,64,128,128,9,64,4>(c,t,x,y);
    }else if(t->O==32768&&t->I==1024)good=dg_dense_attention<32768,1024,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==16384&&t->I==1024)good=dg_dense_attention<16384,1024,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==4096&&t->I==8192)good=dg_dense_attention<4096,8192,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==4096&&t->I==4096)good=dg_dense_attention<4096,4096,128,64,64,10,128,1>(c,t,x,y);
    else if(t->O==4096&&t->I==1024)good=dg_dense_attention<4096,1024,128,64,64,10,128,1>(c,t,x,y);
    if(!good)return 0;
    static int checks;
    if(checks<4&&getenv("DSV4_CUDA_ATTN_DG_AB")&&atoi(getenv("DSV4_CUDA_ATTN_DG_AB"))){
        checks++;float hs[2];
        if(!buf((void**)&c->dgref,&c->dgrefcap,(size_t)t->I*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(size_t)t->O*sizeof(float))||
           !ok(cudaMemcpyAsync(c->dgref,x,(size_t)t->I*sizeof(float),cudaMemcpyDeviceToDevice,c->stream),"attention A/B input copy"))return 0;
        fp8_sim<<<(t->I+127)/128,128,0,c->stream>>>(c->dgref,1,t->I);
        run_mv<8>((uint8_t*)t->w,t->scale,c->dgref,c->dy,t->O,t->I,1,c->stream);
        dg_compare<<<1,256,0,c->stream>>>(c->dy,y,c->dgref,t->O);
        if(!ok(cudaMemcpyAsync(hs,c->dgref,sizeof(hs),cudaMemcpyDeviceToHost,c->stream),"attention A/B stats")||
           !ok(cudaStreamSynchronize(c->stream),"attention A/B sync"))return 0;
        fprintf(stderr,"[DSV4 attention DeepGEMM A/B] %dx%d max=%.6g mae=%.6g\n",t->O,t->I,hs[0],hs[1]);
    }
    return 1;
}
template<int G> static int dg_attention_wo_a_grouped(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){constexpr int N=1024,K=4096,O=G*N;
    if(!t||t->fmt!=9||t->O!=O||t->I!=K||!t->dg_scale||!buf((void**)&c->dga1,&c->dga1cap,(size_t)G*K)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,(size_t)G*(K/512)*4*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,(size_t)O*sizeof(__nv_bfloat16)))return 0;
    dg_quant_batched8<<<G*(K/512),128,0,c->stream>>>(x,c->dga1,c->dgsfa1,K);CUtensorMap a,b,sa,sb,d;
    if(!dg_map_batched(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,t->w,K,N,G,K,(unsigned long long)K*N,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map_batched(&b,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,K,1,G,K,K,128,16,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,t->dg_scale,N,G*(K/512),(unsigned long long)N*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,4,G*(K/512),16,16,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,1,(unsigned long long)N*2,16,64,CU_TENSOR_MAP_SWIZZLE_NONE))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,K,128,128,G,64,16,128,128,128,0,9,128,256,170,deep_gemm::GemmType::Batched,false,DgOutput,DgEpilogue,false,false,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,1,K,128,128,G,64,16,128,128,128,0,9,128,256,84,deep_gemm::GemmType::Batched,false,DgOutput,DgEpilogue,false,false,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,95872),"wo_a DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,95872,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)t->w,(__nv_fp8_e4m3*)c->dga1,nullptr,nullptr,nullptr,N,1,K,1,N,N,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    dg_bf16_to_float<<<(O+255)/256,256,0,c->stream>>>(y,c->dgmm1,O);return ok(cudaGetLastError(),"wo_a DeepGEMM launch");
}
static int dg_attention_wo_a(Dev*c,Dsv4CudaTensor*t,const float*x,float*y){
    return t&&t->O==8192?dg_attention_wo_a_grouped<8>(c,t,x,y):t&&t->O==4096?dg_attention_wo_a_grouped<4>(c,t,x,y):0;
}
template<int G> static int dg_moe(Dev*c,Dsv4CudaExpertSet*set,const float*weight,float limit,float*out){constexpr int M=16,N=4096;
    size_t a1=(size_t)G*M*4096,a2=(size_t)G*M*2048,s1=(size_t)G*M*8*4,s2=(size_t)G*M*4*4,mm=(size_t)G*M*N*2;
    if(!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||
       !buf((void**)&c->dglayout,&c->dglayoutcap,G*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,mm)||!buf((void**)&c->dgmm2,&c->dgmm2cap,mm))return 0;
    dg_mask6<<<1,256,0,c->stream>>>(c->dglayout,c->expert_ids,G);dg_quant_masked<<<6*8,128,0,c->stream>>>(c->dx,c->dga1,c->dgsfa1,c->expert_ids,4096,M);
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,4096,(unsigned long long)G*M,4096,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,4096,(unsigned long long)G*N,2048,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,(unsigned long long)G*8,(unsigned long long)M*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N,(unsigned long long)G*32,16384,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,(unsigned long long)G*M,(unsigned long long)N*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"DeepGEMM FC1 shared memory"))return 0;
    fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N,4096,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    int debug=getenv("DSV4_CUDA_DG_PROFILE")&&atoi(getenv("DSV4_CUDA_DG_PROFILE"));if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM FC1 sync"))return 0;
    dg_swiglu_quant_masked<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,c->expert_ids,weight,limit,M);
    if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM SwiGLU sync"))return 0;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,2048,(unsigned long long)G*M,2048,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,2048,(unsigned long long)G*N,1024,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,(unsigned long long)G*4,(unsigned long long)M*4,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,N,(unsigned long long)G*16,16384,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,N,(unsigned long long)G*M,(unsigned long long)N*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedMasked,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"DeepGEMM FC2 shared memory"))return 0;
    fc2<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,N,2048,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM FC2 sync"))return 0;dg_reduce6_masked<<<16,256,0,c->stream>>>(out,c->dgmm2,c->expert_ids,M);if(debug&&!ok(cudaStreamSynchronize(c->stream),"DeepGEMM reduce sync"))return 0;return ok(cudaGetLastError(),"DeepGEMM routed MoE launch");
}
template<int G> static int dg_moe_contiguous_fast(Dev*c,Dsv4CudaExpertSet*set,const float*weight,float limit,float*out,int ep_base=-1){constexpr int M=384,N=4096,A=64;
    size_t a1=(size_t)M*4096,a2=(size_t)M*2048,s1=(size_t)M*8*4,s2=(size_t)M*4*4,mm=(size_t)M*N*2;
    if(!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||
       !buf((void**)&c->dglayout,&c->dglayoutcap,M*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,mm)||!buf((void**)&c->dgmm2,&c->dgmm2cap,mm))return 0;
    if(ep_base>=0)dg_layout6_ep<<<3,256,0,c->stream>>>(c->dglayout,c->expert_ids,M,A,ep_base);else dg_layout6<<<3,256,0,c->stream>>>(c->dglayout,c->expert_ids,M,A);
    dg_quant_one<<<8,128,0,c->stream>>>(c->dx,c->dga1,c->dgsfa1,4096,M);dg_replicate_rows<<<((long long)M*4096+255)/256,256,0,c->stream>>>(c->dga1,c->dgsfa1,4096,M);
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,4096,M,4096,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,4096,(unsigned long long)G*N,2048,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,8,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N,(unsigned long long)G*32,16384,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"DeepGEMM FC1 shared memory"))return 0;
    fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N,4096,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(ep_base>=0)dg_swiglu_quant_unique_ep<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,c->expert_ids,ep_base,limit,M,A);
    else dg_swiglu_quant_unique<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,weight,limit,M,A);
    dg_replicate_expert_rows<<<((long long)M*2048+255)/256,256,0,c->stream>>>(c->dga2,c->dgsfa2,2048,M,A);
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,2048,M,2048,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,2048,(unsigned long long)G*N,1024,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,4,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,N,(unsigned long long)G*16,16384,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,2,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,128,128,128,128,128,2,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,101376),"DeepGEMM FC2 shared memory"))return 0;
    fc2<<<(c->sms>=170?170:84),384,101376,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,N,2048,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(ep_base>=0)dg_reduce6_ep<<<16,256,0,c->stream>>>(out,c->dgmm2,c->expert_ids,weight,ep_base,A);else dg_reduce6<<<16,256,0,c->stream>>>(out,c->dgmm2,A);
    return ok(cudaGetLastError(),"DeepGEMM routed MoE fast launch");
}
/* ids/weight are DEVICE pointers (uploaded by the caller); host_ids is the
 * host copy used to build the expert-grouped layout (nullptr = legacy
 * replicated layout, also DSV4_CUDA_MOE_GROUPED=0). */
static int dg_moe_batch_contiguous(Dev*c,Dsv4CudaExpertSet*set,const float*input,const int*ids,const float*weight,int tokens,float limit,float*out,const int*host_ids=nullptr){constexpr int N=4096,A=64,G=256;int routes=tokens*6;
    static int grouped=-1;if(grouped<0){const char*e=getenv("DSV4_CUDA_MOE_GROUPED");grouped=!(e&&*e=='0');}
    int M=routes*A;
    std::vector<int> hrow,hlayout;
    if(host_ids&&grouped){
        /* Stable bucket by expert: rows per expert padded to A. */
        int cnt[G]={0};for(int r=0;r<routes;r++){int e=host_ids[r];if(e<0||e>=G)return 0;cnt[e]++;}
        int base[G];int m=0;for(int e=0;e<G;e++){base[e]=m;if(cnt[e])m+=(cnt[e]+A-1)/A*A;}
        M=m;hrow.resize(routes);hlayout.assign(M,0);
        int fill[G]={0};for(int r=0;r<routes;r++){int e=host_ids[r];hrow[r]=base[e]+fill[e]++;}
        for(int e=0;e<G;e++)if(cnt[e]){int g=(cnt[e]+A-1)/A*A;for(int i=0;i<g;i++)hlayout[base[e]+i]=e;}
    }
    size_t a1=(size_t)M*4096,a2=(size_t)M*2048,s1=(size_t)M*8*4,s2=(size_t)M*4*4,mm=(size_t)M*N*2;
    if(set->count!=G||set->H!=4096||set->I!=2048||!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||!buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||!buf((void**)&c->dglayout,&c->dglayoutcap,(size_t)M*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,mm)||!buf((void**)&c->dgmm2,&c->dgmm2cap,mm))return 0;
    bool use_map=!hrow.empty();
    if(use_map){
        if(!buf((void**)&c->dgrowmap,&c->dgrowmapcap,(size_t)routes*sizeof(int))||
           !ok(cudaMemcpyAsync(c->dgrowmap,hrow.data(),(size_t)routes*sizeof(int),cudaMemcpyHostToDevice,c->stream),"grouped MoE rowmap upload")||
           !ok(cudaMemcpyAsync(c->dglayout,hlayout.data(),(size_t)M*sizeof(int),cudaMemcpyHostToDevice,c->stream),"grouped MoE layout upload")||
           !ok(cudaMemsetAsync(c->dga1,0,a1,c->stream),"grouped MoE A1 clear")||!ok(cudaMemsetAsync(c->dgsfa1,0,s1,c->stream),"grouped MoE S1 clear")||
           !ok(cudaMemsetAsync(c->dga2,0,a2,c->stream),"grouped MoE A2 clear")||!ok(cudaMemsetAsync(c->dgsfa2,0,s2,c->stream),"grouped MoE S2 clear"))return 0;
        dg_quant_routes_map<<<routes*8,128,0,c->stream>>>(input,c->dga1,c->dgsfa1,tokens,4096,M,c->dgrowmap);
    }else{
        dg_layout_routes<<<(M+255)/256,256,0,c->stream>>>(c->dglayout,ids,M,A);dg_quant_routes<<<routes*8,128,0,c->stream>>>(input,c->dga1,c->dgsfa1,tokens,4096,M,A);dg_replicate_expert_rows<<<((long long)M*4096+255)/256,256,0,c->stream>>>(c->dga1,c->dgsfa1,4096,M,A);
    }
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,4096,M,4096,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,4096,(unsigned long long)G*N,2048,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,8,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N,(unsigned long long)G*32,16384,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,4096,128,32,G,64,64,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"batched MoE FC1 shared memory"))return 0;fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N,4096,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(use_map)dg_swiglu_quant_routes_map<<<routes,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,weight,limit,routes,M,c->dgrowmap);
    else{dg_swiglu_quant_routes<<<routes,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,weight,limit,routes,M,A);dg_replicate_expert_rows<<<((long long)M*2048+255)/256,256,0,c->stream>>>(c->dga2,c->dgsfa2,2048,M,A);}
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,2048,M,2048,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,2048,(unsigned long long)G*N,1024,128,64,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,4,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,N,(unsigned long long)G*16,16384,64,1,CU_TENSOR_MAP_SWIZZLE_NONE)||!dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,N,M,(unsigned long long)N*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,64,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N,2048,128,32,G,64,64,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"batched MoE FC2 shared memory"))return 0;fc2<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,N,2048,N,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);if(use_map)dg_reduce_routes_map<<<((long long)tokens*4096+255)/256,256,0,c->stream>>>(out,c->dgmm2,tokens,c->dgrowmap);else dg_reduce_routes<<<((long long)tokens*4096+255)/256,256,0,c->stream>>>(out,c->dgmm2,tokens,A);return ok(cudaGetLastError(),"batched contiguous MoE launch");
}
static int dg_moe_tp2(Dev*c,Dsv4CudaExpertSet*set,const float*weight,float limit,float*out){constexpr int G=256,M=384,A=64,H=4096,J=1024,N1=2048;
    int trace=getenv("DSV4_CUDA_TP2_TRACE")&&atoi(getenv("DSV4_CUDA_TP2_TRACE"));
    size_t a1=(size_t)M*H,a2=(size_t)M*J,s1=(size_t)M*(H/512)*4,s2=(size_t)M*(J/512)*4;
    if(!set||set->count!=G||set->H!=H||set->I!=J||!buf((void**)&c->dga1,&c->dga1cap,a1)||!buf((void**)&c->dga2,&c->dga2cap,a2)||
       !buf((void**)&c->dgsfa1,&c->dgsfa1cap,s1)||!buf((void**)&c->dgsfa2,&c->dgsfa2cap,s2)||
       !buf((void**)&c->dglayout,&c->dglayoutcap,M*sizeof(int))||!buf((void**)&c->dgmm1,&c->dgmm1cap,(size_t)M*N1*2)||
       !buf((void**)&c->dgmm2,&c->dgmm2cap,(size_t)M*H*2))return 0;
    dg_layout6<<<3,256,0,c->stream>>>(c->dglayout,c->expert_ids,M,A);
    dg_quant_one<<<H/512,128,0,c->stream>>>(c->dx,c->dga1,c->dgsfa1,H,M);
    dg_replicate_rows<<<((long long)M*H+255)/256,256,0,c->stream>>>(c->dga1,c->dgsfa1,H,M);
    CUtensorMap a,b,sa,sb,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga1,H,M,H,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc1,H,(unsigned long long)G*N1,H/2,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa1,M,H/512,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc1_dg_scale,N1,(unsigned long long)G*(H/128),(unsigned long long)N1*4,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,N1,M,(unsigned long long)N1*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc1=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N1,H,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,N1,H,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc1,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"TP2 DeepGEMM FC1 shared memory"))return 0;
    fc1<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(__nv_fp8_e4m3*)c->dga1,(__nv_fp8_e4m3*)set->bank_fc1,c->dglayout,nullptr,nullptr,M,N1,H,N1,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(trace){fprintf(stderr,"[DSV4 TP2] device %d FC1 launched\n",c->id);fflush(stderr);if(!ok(cudaStreamSynchronize(c->stream),"TP2 FC1 trace sync"))return 0;fprintf(stderr,"[DSV4 TP2] device %d FC1 done\n",c->id);fflush(stderr);}
    dg_swiglu_quant_unique_tp2<<<6,128,0,c->stream>>>(c->dgmm1,c->dga2,c->dgsfa2,limit,M,A);
    dg_replicate_expert_rows<<<((long long)M*J+255)/256,256,0,c->stream>>>(c->dga2,c->dgsfa2,J,M,A);
    if(trace){if(!ok(cudaStreamSynchronize(c->stream),"TP2 SwiGLU trace sync"))return 0;fprintf(stderr,"[DSV4 TP2] device %d SwiGLU done\n",c->id);fflush(stderr);}
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_UINT8,c->dga2,J,M,J,128,A,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,set->bank_fc2,J,(unsigned long long)G*H,J/2,128,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&sa,CU_TENSOR_MAP_DATA_TYPE_INT32,c->dgsfa2,M,J/512,(unsigned long long)M*4,A,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&sb,CU_TENSOR_MAP_DATA_TYPE_INT32,set->bank_fc2_dg_scale,H,(unsigned long long)G*(J/128),(unsigned long long)H*4,128,1,CU_TENSOR_MAP_SWIZZLE_NONE)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm2,H,M,(unsigned long long)H*2,64,A,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto fc2=c->sms>=170?deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,H,J,128,32,G,64,128,128,128,128,128,3,128,256,170,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>:deep_gemm::sm120_fp8_fp4_gemm_1d1d_impl<0,H,J,128,32,G,64,128,128,128,128,128,3,128,256,84,deep_gemm::GemmType::MGroupedContiguous,false,DgOutput,DgEpilogue,false,true,false,true,false,64,1>;
    if(!ok(cudaFuncSetAttribute(fc2,cudaFuncAttributeMaxDynamicSharedMemorySize,92672),"TP2 DeepGEMM FC2 shared memory"))return 0;
    fc2<<<(c->sms>=170?170:84),384,92672,c->stream>>>((DgOutput*)c->dgmm2,nullptr,(__nv_fp8_e4m3*)c->dga2,(__nv_fp8_e4m3*)set->bank_fc2,c->dglayout,nullptr,nullptr,M,H,J,H,0,0,deep_gemm::TmaParamPad48{},a,b,sa,sb,d);
    if(trace){fprintf(stderr,"[DSV4 TP2] device %d FC2 launched\n",c->id);fflush(stderr);if(!ok(cudaStreamSynchronize(c->stream),"TP2 FC2 trace sync"))return 0;fprintf(stderr,"[DSV4 TP2] device %d FC2 done\n",c->id);fflush(stderr);}
    dg_reduce6_weighted<<<16,256,0,c->stream>>>(out,c->dgmm2,weight,A);
    return ok(cudaGetLastError(),"TP2 DeepGEMM routed MoE launch");
}
static int dg_dump_once(Dev*c){static int dumped;if(dumped++||!(getenv("DSV4_CUDA_DG_DUMP")&&atoi(getenv("DSV4_CUDA_DG_DUMP"))))return 1;
    struct Item{const char*path;const void*device;size_t bytes;}items[]={
        {"/tmp/dsv4-dg-x.f32",c->dx,4096*4},{"/tmp/dsv4-dg-a1.fp8",c->dga1,(size_t)768*4096},
        {"/tmp/dsv4-dg-sfa1.bin",c->dgsfa1,(size_t)768*8*4},{"/tmp/dsv4-dg-layout.i32",c->dglayout,768*4},
        {"/tmp/dsv4-dg-fc1.bf16",c->dgmm1,(size_t)768*4096*2},{"/tmp/dsv4-dg-ids.i32",c->expert_ids,6*4},
        {"/tmp/dsv4-dg-weights.f32",c->expert_weights,6*4}};
    if(!ok(cudaStreamSynchronize(c->stream),"DeepGEMM dump sync"))return 0;
    for(unsigned i=0;i<sizeof(items)/sizeof(*items);i++){void*h=malloc(items[i].bytes);FILE*f=nullptr;if(!h||!ok(cudaMemcpy(h,items[i].device,items[i].bytes,cudaMemcpyDeviceToHost),"DeepGEMM dump copy")||!(f=fopen(items[i].path,"wb"))||fwrite(h,1,items[i].bytes,f)!=items[i].bytes){if(f)fclose(f);free(h);fprintf(stderr,"[DSV4 DeepGEMM] dump failed: %s\n",items[i].path);return 0;}fclose(f);free(h);}
    fprintf(stderr,"[DSV4 DeepGEMM] dumped first FC1 fixture to /tmp/dsv4-dg-*\n");return 1;}
#endif
__global__ void expert_reduce(float *y,const float *parts,int count,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){float v=0;
    for(int e=0;e<count;e++)v+=__bfloat162float(__float2bfloat16(parts[(long long)e*n+i]));y[i]=v;}}
/* Batched siblings of expert_reduce / expert_act(weight=1) for the generic
 * batched MoE: parts hold `count` rows per token ([t*count+e][n]). */
__global__ void expert_reduce_rows(float *y,const float *parts,int tokens,int count,int n){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(p<(long long)tokens*n){int t=p/n,i=p%n;float v=0;for(int e=0;e<count;e++)v+=__bfloat162float(__float2bfloat16(parts[((long long)t*count+e)*n+i]));y[p]=v;}}
__global__ void expert_act_rows1(float *a,const float *g,const float *u,float limit,long long n){long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){
    float gg=fminf(__bfloat162float(__float2bfloat16(g[i])),limit);
    float uu=fmaxf(-limit,fminf(__bfloat162float(__float2bfloat16(u[i])),limit));
    a[i]=__bfloat162float(__float2bfloat16((gg/(1.f+expf(-gg)))*uu));}}
__global__ void moe_combine(float *routed,const float *shared,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)
    routed[i]=__bfloat162float(__float2bfloat16(routed[i]+__bfloat162float(__float2bfloat16(shared[i]))));}
__global__ void ep2_localize(int *ids,float *weights,int base){int k=threadIdx.x;if(k<6){int id=ids[k]-base;if(id<0||id>=128){ids[k]=-1;weights[k]=0.f;}else ids[k]=id;}}
__global__ void sort_routes6(int*ids,float*weights){if(threadIdx.x||blockIdx.x)return;for(int i=1;i<6;i++){int id=ids[i],j=i-1;float w=weights[i];while(j>=0&&ids[j]>id){ids[j+1]=ids[j];weights[j+1]=weights[j];j--;}ids[j+1]=id;weights[j+1]=w;}}
__global__ void add_peer(float *out,const float *peer,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__bfloat162float(__float2bfloat16(out[i]+peer[i]));}
__global__ void bf16_round(float *x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=__bfloat162float(__float2bfloat16(x[i]));}
__device__ __forceinline__ float warp_sum_f32(float v){for(int d=16;d;d>>=1)v+=__shfl_down_sync(0xffffffff,v,d);return v;}
__global__ void rmsnorm_seq(float *x,const __nv_bfloat16 *w,int n,float eps){
    if(blockDim.x==1){double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];float inv=rsqrtf((float)(ss/n)+eps);for(int i=0;i<n;i++)x[i]=__bfloat162float(__float2bfloat16(x[i]*inv*__bfloat162float(w[i])));return;}
    __shared__ float ws[8],inv;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int i=threadIdx.x;i<n;i+=blockDim.x)ss+=x[i]*x[i];ss=warp_sum_f32(ss);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;v=warp_sum_f32(v);if(!lane)inv=rsqrtf(v/n+eps);}__syncthreads();for(int i=threadIdx.x;i<n;i+=blockDim.x)x[i]=__bfloat162float(__float2bfloat16(x[i]*inv*__bfloat162float(w[i])));
}
__global__ void rmsnorm_f32(float *out,const float *x,const float *w,int n,float eps){
    if(blockDim.x==1){double ss=0;for(int i=0;i<n;i++)ss+=(double)x[i]*x[i];float inv=rsqrtf((float)(ss/n)+eps);for(int i=0;i<n;i++)out[i]=__bfloat162float(__float2bfloat16(x[i]*inv*w[i]));return;}
    __shared__ float ws[8],inv;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;for(int i=threadIdx.x;i<n;i+=blockDim.x)ss+=x[i]*x[i];ss=warp_sum_f32(ss);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;v=warp_sum_f32(v);if(!lane)inv=rsqrtf(v/n+eps);}__syncthreads();for(int i=threadIdx.x;i<n;i+=blockDim.x)out[i]=__bfloat162float(__float2bfloat16(x[i]*inv*w[i]));
}
__global__ void head_rmsnorm(float *q,int heads,int dim,float eps){int h=blockIdx.x;if(h>=heads)return;float *x=q+(long long)h*dim;__shared__ float sum[256],inv;float ss=0;
    for(int d=threadIdx.x;d<dim;d+=blockDim.x)ss+=x[d]*x[d];sum[threadIdx.x]=ss;__syncthreads();for(int n=blockDim.x/2;n;n>>=1){if(threadIdx.x<n)sum[threadIdx.x]+=sum[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)inv=rsqrtf(sum[0]/dim+eps);__syncthreads();for(int d=threadIdx.x;d<dim;d+=blockDim.x)x[d]=__bfloat162float(__float2bfloat16(x[d]*inv));}
__global__ void attention_one(float *context,const float *q,const float *kv,const float *sink,int heads,int dim){int h=blockIdx.x;if(h>=heads)return;__shared__ float a,den;
    if(!threadIdx.x){float score=0;for(int d=0;d<dim;d++)score+=q[(long long)h*dim+d]*kv[d];score/=sqrtf((float)dim);float mx=fmaxf(score,sink[h]);a=expf(score-mx);den=a+expf(sink[h]-mx);}__syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x)context[(long long)h*dim+d]=__bfloat162float(__float2bfloat16(a*kv[d]/den));}
__global__ void rope_heads(float *x,int heads,int dim,int rope,int pos,int inverse,const float *cs,const float *sn){int p=blockIdx.x*blockDim.x+threadIdx.x,pairs=rope/2;if(p<heads*pairs){int h=p/pairs,k=p%pairs,off=h*dim+dim-rope+2*k;float s=inverse?-sn[(long long)pos*pairs+k]:sn[(long long)pos*pairs+k],u=x[off],v=x[off+1];x[off]=__bfloat162float(__float2bfloat16(u*cs[(long long)pos*pairs+k]-v*s));x[off+1]=__bfloat162float(__float2bfloat16(u*s+v*cs[(long long)pos*pairs+k]));}}
__global__ void cache_store(float *cache,const float *kv,int slot,int dim){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim)cache[(long long)slot*dim+d]=kv[d];}
__global__ void fill_value(float *x,long long n,float v){long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)x[i]=v;}
__global__ void compress_store(float *values,float *scores,const float *v,const float *g,const float *ape,int slot,int phase,int O){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<O){values[(long long)slot*O+d]=v[d];scores[(long long)slot*O+d]=g[d]+ape[(long long)phase*O+d];}}
__global__ void compress_pool(float *out,const float *values,const float *scores,int ratio,int overlap,int dim){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim){int O=(1+overlap)*dim,items=overlap?2*ratio:ratio;float mx=-INFINITY;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d;mx=fmaxf(mx,scores[(long long)j*O+col]);}float den=0,sum=0;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d,aoff=j*O+col;float a=expf(scores[aoff]-mx);den+=a;sum+=a*values[aoff];}out[d]=sum/den;}}
__global__ void attention_window_kernel(float *context,const float *q,const float *cache,const float *compressed,const float *sink,int heads,int dim,int window,int pos,int cn){
    int h=blockIdx.x;if(h>=heads)return;extern __shared__ float score[];__shared__ float mx,den;int start=pos-window+1;if(start<0)start=0;int nw=pos-start+1;
    if(!threadIdx.x){mx=sink[h];for(int t=start;t<=pos;t++){float z=0,*v=(float*)cache+(long long)(t%window)*dim;for(int d=0;d<dim;d++)z+=q[(long long)h*dim+d]*v[d];z/=sqrtf((float)dim);score[t-start]=z;mx=fmaxf(mx,z);}for(int i=0;i<cn;i++){float z=0,*v=(float*)compressed+(long long)i*dim;for(int d=0;d<dim;d++)z+=q[(long long)h*dim+d]*v[d];z/=sqrtf((float)dim);score[nw+i]=z;mx=fmaxf(mx,z);}den=expf(sink[h]-mx);for(int i=0;i<nw+cn;i++){float a=expf(score[i]-mx);score[i]=a;den+=a;}}
    __syncthreads();for(int d=threadIdx.x;d<dim;d+=blockDim.x){float sum=0;for(int t=start;t<=pos;t++)sum+=score[t-start]*cache[((long long)(t%window))*dim+d];for(int i=0;i<cn;i++)sum+=score[nw+i]*compressed[(long long)i*dim+d];context[(long long)h*dim+d]=__bfloat162float(__float2bfloat16(sum/den));}}
/* Prefill batched sparse window attention.  vals is a linear slab of KV rows:
 * [0, comp_base) window rows in position order, [comp_base, ...) the
 * compressed rows.  meta[3t] = first window row for token t, meta[3t+1] = its
 * window row count, meta[3t+2] = its visible compressed prefix.  Numerics
 * mirror coli_v4_sparse_attention_ref: fp32 scores, max over scores only
 * (sink joins the denominator, not the max), probabilities rounded to bf16
 * before the value sum, output rounded to bf16 after the division. */
/* ---- persistent per-layer KV cache for batched prefill attention ----
 * The engine rebuilt and re-uploaded the linear KV slab every chunk-layer
 * (~2 GB PCIe per 826-token prefill). Instead: a per-layer device RING of
 * window rows (absolute position % window) appended once per chunk, plus an
 * append-only compressed-prefix buffer. The engine tracks validity by exact
 * position continuity and reseeds on any mismatch, so no invalidation hooks
 * are needed here. */
#define DSV4_KV_CACHE_LAYERS 64
static float *g_kv_ring[DSV4_KV_CACHE_LAYERS];
static int g_kv_ring_rows[DSV4_KV_CACHE_LAYERS];
static float *g_kv_comp[DSV4_KV_CACHE_LAYERS];
static int g_kv_comp_cap[DSV4_KV_CACHE_LAYERS];

/* Which GPUs this build's kernels can run on. The DeepGEMM build carries
 * sm_120a-only block-scaled MMA kernels. The generic build is a portable fat
 * binary of plain CUDA kernels (fp32 compute from fp8/fp4 decode); the engine's
 * loader tries the DeepGEMM DLL first and falls back to the generic one when
 * this says no.
 *
 * The generic floor is sm_60, not sm_80. Nothing in these kernels needs Ampere:
 * bf16 is storage only (every use is a __bfloat162float / __float2bfloat16
 * convert, the arithmetic is fp32) and fp8 weights are decoded through a
 * constant lookup table rather than fp8 tensor cores. The warp primitives used
 * here (__shfl_*_sync, __syncwarp) are sm_30-era and impose no floor of their
 * own.
 *
 * sm_60 is chosen because Pascal is the oldest architecture this has actually
 * been validated on -- sm_61 on a GTX 1080 Ti, matching the CPU reference to
 * 2^-26. Maxwell would very likely work and is deliberately NOT claimed here,
 * because nobody has run it.
 *
 * This is a RUNTIME gate and is separate from what the build actually contains:
 * a binary compiled with CUDA_ARCH=portable still has no sm_61 or sm_75 code in
 * it. Use CUDA_ARCH=portable-pre-ampere for those cards. Saying yes here for a
 * card the build has no cubin for surfaces as a "no kernel image is available"
 * launch failure, not a wrong answer. */
extern "C" int dsv4_cuda_backend_arch_ok(int device){
    cudaDeviceProp prop;
    if(cudaGetDeviceProperties(&prop,device)!=cudaSuccess){cudaGetLastError();return 0;}
#ifdef COLI_DSV4_DEEPGEMM
    return prop.major==12;
#else
    return prop.major>=6;
#endif
}
extern "C" const char *dsv4_cuda_backend_name(void){
#ifdef COLI_DSV4_DEEPGEMM
    return "deepgemm-sm120";
#else
    return "generic";
#endif
}

extern "C" long long dsv4_cuda_mem_free_mb(int device){
    if(!ok(cudaSetDevice(device),"select meminfo device"))return -1;
    size_t free_b=0,total_b=0;
    if(!ok(cudaMemGetInfo(&free_b,&total_b),"device meminfo"))return -1;
    return (long long)(free_b>>20);
}

/* 1 when the device shares its memory with the host (GB10 / DGX Spark, Jetson,
 * any integrated part): cudaMemGetInfo then reports the system's free memory,
 * not headroom on a card, so a VRAM reserve compared against it says nothing
 * about whether one more mirror fits (#1538). */
/* Number of CUDA devices the runtime sees (-1 on error): the engine looks
 * for a second card to host more decode expert mirrors. */
extern "C" int dsv4_cuda_device_count(void){
    int n=0;if(cudaGetDeviceCount(&n)!=cudaSuccess){cudaGetLastError();return -1;}return n;
}

extern "C" int dsv4_cuda_device_unified(int device){
    int integrated=0,host_tables=0;
    if(cudaDeviceGetAttribute(&integrated,cudaDevAttrIntegrated,device)!=cudaSuccess){cudaGetLastError();integrated=0;}
    if(cudaDeviceGetAttribute(&host_tables,cudaDevAttrPageableMemoryAccessUsesHostPageTables,device)!=cudaSuccess){cudaGetLastError();host_tables=0;}
    return (integrated||host_tables)?1:0;
}

extern "C" int dsv4_cuda_kv_ring_append(int device,int layer,const float*rows,
        int start_pos,int count,int window,int dim){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!rows||start_pos<0||count<1||
       window<1||dim<1||!ok(cudaSetDevice(device),"select kv ring device"))return 0;
    if(g_kv_ring[layer]&&g_kv_ring_rows[layer]!=window){cudaFree(g_kv_ring[layer]);g_kv_ring[layer]=NULL;}
    if(!g_kv_ring[layer]){
        if(!alloc_ok(cudaMalloc((void**)&g_kv_ring[layer],(size_t)window*dim*4),"kv ring allocation"))return 0;
        g_kv_ring_rows[layer]=window;
    }
    int done=0;
    while(done<count){
        int slot=(start_pos+done)%window;
        int run=count-done;if(run>window-slot)run=window-slot;
        if(!ok(cudaMemcpyAsync(g_kv_ring[layer]+(size_t)slot*dim,rows+(size_t)done*dim,
                               (size_t)run*dim*4,cudaMemcpyHostToDevice,c->stream),"kv ring append"))return 0;
        done+=run;
    }
    return 1;
}

extern "C" int dsv4_cuda_kv_comp_append(int device,int layer,const float*rows,
        int start_idx,int count,int dim){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!rows||start_idx<0||count<1||dim<1||
       !ok(cudaSetDevice(device),"select kv comp device"))return 0;
    int need=start_idx+count;
    if(g_kv_comp_cap[layer]<need){
        int cap=g_kv_comp_cap[layer]?g_kv_comp_cap[layer]:512;
        while(cap<need)cap*=2;
        float*grown=NULL;
        if(!alloc_ok(cudaMalloc((void**)&grown,(size_t)cap*dim*4),"kv comp allocation"))return 0;
        if(g_kv_comp[layer]&&start_idx&&
           !ok(cudaMemcpyAsync(grown,g_kv_comp[layer],(size_t)start_idx*dim*4,
                               cudaMemcpyDeviceToDevice,c->stream),"kv comp grow copy")){cudaFree(grown);return 0;}
        if(g_kv_comp[layer]){cudaStreamSynchronize(c->stream);cudaFree(g_kv_comp[layer]);}
        g_kv_comp[layer]=grown;g_kv_comp_cap[layer]=cap;
    }
    return ok(cudaMemcpyAsync(g_kv_comp[layer]+(size_t)start_idx*dim,rows,
                              (size_t)count*dim*4,cudaMemcpyHostToDevice,c->stream),"kv comp append");
}

/* Window rows at absolute position >= chunk_start come from the chunk buffer,
 * not the ring: with the ring indexed position%window, appending the chunk
 * before attention would let its newest rows overwrite the oldest window rows
 * that early chunk tokens still need (only bites once start+batch > window).
 * The engine appends the chunk to the ring AFTER attention instead. */
__device__ __forceinline__ const float*cached_row(const float*win,const float*chunk,
        int ap,int chunk_start,int window,int dim){
    return ap>=chunk_start?chunk+(long long)(ap-chunk_start)*dim
                          :win+(long long)(ap%window)*dim;
}

/* DSV4_SATTN_V2 (1): the cached sparse attention rewritten for latency —
 * the same arithmetic in the same order (each lane's per-row dot, the
 * warp shuffle, the denominator's serial sum, the row-ordered weighted
 * sum), only the loads overlapped: four rows per warp step for the
 * scores, the selection indices and eight rows prefetched into registers
 * for the weighted sum, the exponentials by every thread with the max as
 * a block reduction. IDX: rows past the window come from the indexer's
 * selection. Measured before: 0.39 ms per 640-row layer, one thread's
 * serial softmax and a dependent index load before each row load. */
template<int IDX,int HPB> __global__ void __launch_bounds__(512) sparse_attn_cached_v2(float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,
        int selstride,int abs_base,int chunk_start,int window,int heads,int dim,float scale){
    int t=blockIdx.y,h0=blockIdx.x*HPB,nh=heads-h0;if(nh>HPB)nh=HPB;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const int*ts=IDX?sel+(long long)t*selstride:nullptr;
    const float*qb=q+((long long)t*heads+h0)*dim;
    int qo[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++)qo[k]=(k<nh?k:0)*dim;
    extern __shared__ float score[];   /* [HPB][total] */
    __shared__ float red[HPB][32];
    __shared__ float den[HPB];
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,nw=blockDim.x>>5;
    for(int item=warp*4;item<total;item+=nw*4){
        int nr=total-item;if(nr>4)nr=4;
        const float*v[4];
        #pragma unroll
        for(int r=0;r<4;r++){int it=item+(r<nr?r:0);
            v[r]=it<wn?cached_row(win,chunk,abs_base+wo+it,chunk_start,window,dim)
                      :comp+(long long)(IDX?ts[it-wn]:(it-wn))*dim;}
        float z[HPB][4];
        #pragma unroll
        for(int k=0;k<HPB;k++){
            #pragma unroll
            for(int r=0;r<4;r++)z[k][r]=0.f;}
        for(int d=lane;d<dim;d+=32){float vd[4];
            #pragma unroll
            for(int r=0;r<4;r++)vd[r]=r<nr?v[r][d]:0.f;
            #pragma unroll
            for(int k=0;k<HPB;k++){float qd=qb[qo[k]+d];
                #pragma unroll
                for(int r=0;r<4;r++)if(r<nr)z[k][r]+=qd*vd[r];}}
        #pragma unroll
        for(int k=0;k<HPB;k++){
            #pragma unroll
            for(int r=0;r<4;r++){float zz=z[k][r];
                for(int n=16;n;n>>=1)zz+=__shfl_down_sync(0xffffffff,zz,n);
                if(!lane&&r<nr&&k<nh)score[k*total+item+r]=zz*scale;}}
    }
    __syncthreads();
    float m[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++){float lm=-INFINITY;
        if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)lm=fmaxf(lm,score[k*total+i]);
        for(int n=16;n;n>>=1)lm=fmaxf(lm,__shfl_xor_sync(0xffffffff,lm,n));
        if(!lane)red[k][warp]=lm;}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<HPB;k++){float mm=red[k][0];for(int w=1;w<nw;w++)mm=fmaxf(mm,red[k][w]);m[k]=mm;}
    #pragma unroll
    for(int k=0;k<HPB;k++)if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)score[k*total+i]=expf(score[k*total+i]-m[k]);
    __syncthreads();
    if(threadIdx.x<HPB){int k=threadIdx.x;if(k<nh){float d0=expf(sink[h0+k]-m[k]);const float*sc=score+k*total;for(int i=0;i<total;i++)d0+=sc[i];den[k]=d0;}}
    __syncthreads();
    #pragma unroll
    for(int k=0;k<HPB;k++)if(k<nh)for(int i=threadIdx.x;i<total;i+=blockDim.x)score[k*total+i]=__bfloat162float(__float2bfloat16(score[k*total+i]));
    __syncthreads();
    float dn[HPB];
    #pragma unroll
    for(int k=0;k<HPB;k++)dn[k]=den[k];
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum[HPB];
        #pragma unroll
        for(int k=0;k<HPB;k++)sum[k]=0.f;
        int i=0;
        for(;i+16<=wn;i+=16){float v[16];
            #pragma unroll
            for(int j=0;j<16;j++)v[j]=cached_row(win,chunk,abs_base+wo+i+j,chunk_start,window,dim)[d];
            #pragma unroll
            for(int k=0;k<HPB;k++){
                #pragma unroll
                for(int j=0;j<16;j++)sum[k]+=score[k*total+i+j]*v[j];}}
        for(;i<wn;i++){float vv=cached_row(win,chunk,abs_base+wo+i,chunk_start,window,dim)[d];
            #pragma unroll
            for(int k=0;k<HPB;k++)sum[k]+=score[k*total+i]*vv;}
        i=0;
        for(;i+16<=cn;i+=16){int ix[16];float v[16];
            #pragma unroll
            for(int j=0;j<16;j++)ix[j]=IDX?ts[i+j]:i+j;
            #pragma unroll
            for(int j=0;j<16;j++)v[j]=comp[(long long)ix[j]*dim+d];
            #pragma unroll
            for(int k=0;k<HPB;k++){
                #pragma unroll
                for(int j=0;j<16;j++)sum[k]+=score[k*total+wn+i+j]*v[j];}}
        for(;i<cn;i++){int ix=IDX?ts[i]:i;float vv=comp[(long long)ix*dim+d];
            #pragma unroll
            for(int k=0;k<HPB;k++)sum[k]+=score[k*total+wn+i]*vv;}
        #pragma unroll
        for(int k=0;k<HPB;k++)if(k<nh)out[((long long)t*heads+h0+k)*dim+d]=__bfloat162float(__float2bfloat16(sum[k]/dn[k]));
    }
}
static int sattn_v2_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_SATTN_V2");on=!e||!*e||*e!='0';}return on;}
/* DSV4_SATTN_HPB (4): heads per block; sm1 = the one-head score row bytes
 * (max_total * 4). Four heads when their score rows fit the 48 KB default,
 * else one. */
static int sattn_hpb(void){static int n=-1;if(n<0){const char*e=getenv("DSV4_SATTN_HPB");n=e&&*e?atoi(e):4;if(n!=4)n=1;}return n;}
/* DSV4_SATTN_HPB_PF (16): heads per block for the multi-token launches
 * (prefill chunks), where the token count fills the card and the key rows
 * are shared across the heads; 16, 8 or 4 (else the one-token rule) */
static int sattn_hpb_pf(void){static int n=-1;if(n<0){const char*e=getenv("DSV4_SATTN_HPB_PF");n=e&&*e?atoi(e):16;if(n!=16&&n!=8)n=4;}return n;}
template<int IDX> static void launch_sattn_v2(int heads,int tokens,size_t sm1,cudaStream_t st,float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,int selstride,int abs_base,
        int chunk_start,int window,int dim,float scale){
    if(tokens>1&&sattn_hpb_pf()==16&&sm1*16<=48*1024)
        sparse_attn_cached_v2<IDX,16><<<dim3((heads+15)/16,tokens),512,sm1*16,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
    else if(tokens>1&&sattn_hpb_pf()>=8&&sm1*8<=48*1024)
        sparse_attn_cached_v2<IDX,8><<<dim3((heads+7)/8,tokens),512,sm1*8,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
    else if(sattn_hpb()==4&&sm1>=1024&&sm1*4<=48*1024)   /* four heads from 256 rows up: fewer blocks under-fill the card on short layers */
        sparse_attn_cached_v2<IDX,4><<<dim3((heads+3)/4,tokens),512,sm1*4,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
    else
        sparse_attn_cached_v2<IDX,1><<<dim3(heads,tokens),512,sm1,st>>>(out,q,win,chunk,comp,sink,meta,sel,selstride,abs_base,chunk_start,window,heads,dim,scale);
}

__global__ void sparse_attn_batch_cached_kernel(float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,int abs_base,
        int chunk_start,int window,int heads,int dim,float scale){
    int t=blockIdx.y,h=blockIdx.x;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const float*qh=q+((long long)t*heads+h)*dim;
    extern __shared__ float score[];
    __shared__ float den;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=item<wn?cached_row(win,chunk,abs_base+wo+item,chunk_start,window,dim)
                              :comp+(long long)(item-wn)*dim;
        float z=0;for(int d=lane;d<dim;d+=32)z+=qh[d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);
        if(!lane)score[item]=z*scale;
    }
    __syncthreads();
    if(!threadIdx.x){float m=-INFINITY;for(int i=0;i<total;i++)m=fmaxf(m,score[i]);
        float d0=expf(sink[h]-m);
        for(int i=0;i<total;i++){float a=expf(score[i]-m);d0+=a;score[i]=__bfloat162float(__float2bfloat16(a));}
        den=d0;}
    __syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum=0;
        for(int i=0;i<wn;i++)sum+=score[i]*cached_row(win,chunk,abs_base+wo+i,chunk_start,window,dim)[d];
        for(int i=0;i<cn;i++)sum+=score[wn+i]*comp[(long long)i*dim+d];
        out[((long long)t*heads+h)*dim+d]=__bfloat162float(__float2bfloat16(sum/den));
    }
}

extern "C" int dsv4_cuda_sparse_attn_batch_cached(int device,int layer,const float*q,
        const float*chunk,int chunk_start,const float*sinks,const int*meta,int abs_base,
        int comp_limit,int heads,int dim,int tokens,float scale,float*out){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_kv_ring[layer]||!q||!chunk||!sinks||
       !meta||!out||abs_base<0||chunk_start<0||heads<1||dim<1||tokens<1||!(scale>0.f))return 0;
    if(comp_limit>0&&(!g_kv_comp[layer]||g_kv_comp_cap[layer]<comp_limit))return 0;
    int window=g_kv_ring_rows[layer],max_total=0;
    for(int t=0;t<tokens;t++){
        int wn=meta[3*t+1],cn=meta[3*t+2],tt=wn+cn;
        if(meta[3*t]<0||wn<1||wn>window||cn<0||cn>comp_limit)return 0;
        if(abs_base+meta[3*t]+wn>chunk_start+tokens)return 0;
        if(tt>max_total)max_total=tt;
    }
    size_t qb=(size_t)tokens*heads*dim*4,cb=(size_t)tokens*dim*4,sb=(size_t)heads*4,
           mb=(size_t)tokens*3*sizeof(int),sm=(size_t)max_total*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select cached sparse attention device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p4,&c->p4cap,qb)||
       !buf((void**)&c->p3,&c->p3cap,cb)||
       !buf((void**)&c->aux1,&c->aux1cap,sb)||!buf((void**)&c->aux2,&c->aux2cap,mb))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"cached attention query upload")||
       !ok(cudaMemcpyAsync(c->p3,chunk,cb,cudaMemcpyHostToDevice,c->stream),"cached attention chunk upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"cached attention sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"cached attention meta upload"))return 0;
    if(sattn_v2_on())launch_sattn_v2<0>(heads,tokens,sm,c->stream,
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,nullptr,0,
        abs_base,chunk_start,window,dim,scale);
    else sparse_attn_batch_cached_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,
        abs_base,chunk_start,window,heads,dim,scale);
    return ok(cudaGetLastError(),"cached sparse attention launch")&&
           ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"cached attention download")&&
           ok(cudaStreamSynchronize(c->stream),"cached sparse attention sync");
}

/* Indexed variant: past index_topk the indexer selects a strict subset of the
 * compressed prefix, so meta[3t+2] becomes the SELECTED count and sel holds
 * each token's compressed ordinals (score order — the same order the CPU
 * reference accumulates, keeping fp32 sums bitwise identical). */
__global__ void sparse_attn_batch_cached_idx_kernel(float*out,const float*q,const float*win,
        const float*chunk,const float*comp,const float*sink,const int*meta,const int*sel,
        int selstride,int abs_base,int chunk_start,int window,int heads,int dim,float scale){
    int t=blockIdx.y,h=blockIdx.x;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const int*ts=sel+(long long)t*selstride;
    const float*qh=q+((long long)t*heads+h)*dim;
    extern __shared__ float score[];
    __shared__ float den;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=item<wn?cached_row(win,chunk,abs_base+wo+item,chunk_start,window,dim)
                              :comp+(long long)ts[item-wn]*dim;
        float z=0;for(int d=lane;d<dim;d+=32)z+=qh[d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);
        if(!lane)score[item]=z*scale;
    }
    __syncthreads();
    if(!threadIdx.x){float m=-INFINITY;for(int i=0;i<total;i++)m=fmaxf(m,score[i]);
        float d0=expf(sink[h]-m);
        for(int i=0;i<total;i++){float a=expf(score[i]-m);d0+=a;score[i]=__bfloat162float(__float2bfloat16(a));}
        den=d0;}
    __syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum=0;
        for(int i=0;i<wn;i++)sum+=score[i]*cached_row(win,chunk,abs_base+wo+i,chunk_start,window,dim)[d];
        for(int i=0;i<cn;i++)sum+=score[wn+i]*comp[(long long)ts[i]*dim+d];
        out[((long long)t*heads+h)*dim+d]=__bfloat162float(__float2bfloat16(sum/den));
    }
}


extern "C" int dsv4_cuda_sparse_attn_batch_cached_idx(int device,int layer,const float*q,
        const float*chunk,int chunk_start,const float*sinks,const int*meta,const int*sel,
        int selstride,int abs_base,int comp_limit,int heads,int dim,int tokens,
        float scale,float*out){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_kv_ring[layer]||!q||!chunk||!sinks||
       !meta||!sel||selstride<1||!out||abs_base<0||chunk_start<0||heads<1||dim<1||
       tokens<1||!(scale>0.f))return 0;
    if(comp_limit>0&&(!g_kv_comp[layer]||g_kv_comp_cap[layer]<comp_limit))return 0;
    int window=g_kv_ring_rows[layer],max_total=0;
    for(int t=0;t<tokens;t++){
        int wn=meta[3*t+1],cn=meta[3*t+2],tt=wn+cn;
        if(meta[3*t]<0||wn<1||wn>window||cn<0||cn>selstride||cn>comp_limit)return 0;
        if(abs_base+meta[3*t]+wn>chunk_start+tokens)return 0;
        for(int i=0;i<cn;i++){int o=sel[(long long)t*selstride+i];if(o<0||o>=comp_limit)return 0;}
        if(tt>max_total)max_total=tt;
    }
    size_t qb=(size_t)tokens*heads*dim*4,cb=(size_t)tokens*dim*4,sb=(size_t)heads*4,
           mb=(size_t)tokens*3*sizeof(int),ib=(size_t)tokens*selstride*sizeof(int),
           sm=(size_t)max_total*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select cached idx sparse attention device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p4,&c->p4cap,qb)||
       !buf((void**)&c->p3,&c->p3cap,cb)||
       !buf((void**)&c->aux1,&c->aux1cap,sb)||!buf((void**)&c->aux2,&c->aux2cap,mb)||
       !buf((void**)&c->aux3,&c->aux3cap,ib))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"cached idx attention query upload")||
       !ok(cudaMemcpyAsync(c->p3,chunk,cb,cudaMemcpyHostToDevice,c->stream),"cached idx attention chunk upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"cached idx attention sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"cached idx attention meta upload")||
       !ok(cudaMemcpyAsync(c->aux3,sel,ib,cudaMemcpyHostToDevice,c->stream),"cached idx attention index upload"))return 0;
    if(sattn_v2_on())launch_sattn_v2<1>(heads,tokens,sm,c->stream,
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,
        selstride,abs_base,chunk_start,window,dim,scale);
    else sparse_attn_batch_cached_idx_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(
        c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,
        selstride,abs_base,chunk_start,window,heads,dim,scale);
    return ok(cudaGetLastError(),"cached idx sparse attention launch")&&
           ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"cached idx attention download")&&
           ok(cudaStreamSynchronize(c->stream),"cached idx sparse attention sync");
}

/* Batched indexer scoring for prefill. One thread per (token, candidate):
 * score = sum_h relu(q_h . k) * w_h with the SAME sequential fp32 order as the
 * CPU reference (dims inner, heads outer, separately rounded mul/add) so the
 * host-side top-k sort sees bitwise-identical scores. queries [tokens][heads*dim]
 * (already hadamard/fp4-rounded), keys [count][dim], head_w [tokens][heads],
 * counts[t] = candidates visible to token t. Candidates >= counts[t] get 0. */
__global__ void indexer_score_kernel(float*scores,const float*queries,const float*keys,
        const float*head_w,const int*counts,int heads,int dim,int count){
    int t=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
    extern __shared__ float qs[];
    int qn=heads*dim;
    for(int i=threadIdx.x;i<qn;i+=blockDim.x)qs[i]=queries[(long long)t*qn+i];
    __syncthreads();
    if(c>=count)return;
    float score=0.f;
    if(c<counts[t]){
        const float*k=keys+(long long)c*dim;
        for(int h=0;h<heads;h++){
            const float*q=qs+h*dim;
            float dot=0.f;
            /* The CPU build (MinGW GCC, -O3 -march=native) does NOT contract
             * these into FMAs — measured — so keep mul and add separately
             * rounded here too. */
            for(int i=0;i<dim;i++)dot=__fadd_rn(dot,__fmul_rn(q[i],k[i]));
            score=__fadd_rn(score,__fmul_rn(fmaxf(dot,0.f),head_w[(long long)t*heads+h]));
        }
    }
    scores[(long long)t*count+c]=score;
}

/* DSV4_IDX_COAL twin of indexer_score_kernel: keys_t [dim][kstride], a
 * warp's 32 candidates in one line per i. The same mul/add sequence. */
__global__ void indexer_score_coal_kernel(float*scores,const float*queries,const float*keys_t,int kstride,
        const float*head_w,const int*counts,int heads,int dim,int count){
    int t=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
    extern __shared__ float qs[];
    int qn=heads*dim;
    for(int i=threadIdx.x;i<qn;i+=blockDim.x)qs[i]=queries[(long long)t*qn+i];
    __syncthreads();
    if(c>=count)return;
    float score=0.f;
    if(c<counts[t]){
        const float*k=keys_t+c;
        for(int h=0;h<heads;h++){
            const float*q=qs+h*dim;
            float dot=0.f;
            for(int i=0;i<dim;i++)dot=__fadd_rn(dot,__fmul_rn(q[i],k[(long long)i*kstride]));
            score=__fadd_rn(score,__fmul_rn(fmaxf(dot,0.f),head_w[(long long)t*heads+h]));
        }
    }
    scores[(long long)t*count+c]=score;
}

extern "C" int dsv4_cuda_indexer_score_batch(int device,const float*queries,const float*keys,
        const float*head_w,const int*counts,int tokens,int heads,int dim,int count,
        float*scores){
    Dev*c=ctx(device);
    if(!c||!queries||!keys||!head_w||!counts||!scores||tokens<1||heads<1||dim<1||count<1)return 0;
    size_t qb=(size_t)tokens*heads*dim*4,kb=(size_t)count*dim*4,wb=(size_t)tokens*heads*4,
           cb=(size_t)tokens*sizeof(int),sb=(size_t)tokens*count*4,sm=(size_t)heads*dim*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select indexer score device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p3,&c->p3cap,kb)||
       !buf((void**)&c->p4,&c->p4cap,sb)||!buf((void**)&c->aux1,&c->aux1cap,wb)||
       !buf((void**)&c->aux2,&c->aux2cap,cb))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,queries,qb,cudaMemcpyHostToDevice,c->stream),"indexer query upload")||
       !ok(cudaMemcpyAsync(c->p3,keys,kb,cudaMemcpyHostToDevice,c->stream),"indexer key upload")||
       !ok(cudaMemcpyAsync(c->aux1,head_w,wb,cudaMemcpyHostToDevice,c->stream),"indexer head weight upload")||
       !ok(cudaMemcpyAsync(c->aux2,counts,cb,cudaMemcpyHostToDevice,c->stream),"indexer count upload"))return 0;
    if(sm>48*1024&&cudaFuncSetAttribute(indexer_score_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess)return 0;
    indexer_score_kernel<<<dim3((count+127)/128,tokens),128,sm,c->stream>>>(
        c->p4,c->p2,c->p3,c->aux1,(int*)c->aux2,heads,dim,count);
    return ok(cudaGetLastError(),"indexer score launch")&&
           ok(cudaMemcpyAsync(scores,c->p4,sb,cudaMemcpyDeviceToHost,c->stream),"indexer score download")&&
           ok(cudaStreamSynchronize(c->stream),"indexer score sync");
}

static int g_idx_coal=-1;
static int idx_coal_on(void){if(g_idx_coal<0){const char*e=getenv("DSV4_IDX_COAL");g_idx_coal=!(e&&*e=='0');}return g_idx_coal;}
/* src [rows][cols] -> dst [cols][dstride] (dstride >= rows) through a padded tile */
__global__ void transpose_f32_kernel(float*dst,const float*src,int rows,int cols,int dstride){
    __shared__ float tile[32][33];
    int bx=blockIdx.x*32,by=blockIdx.y*32;
    int x=bx+threadIdx.x,y=by+threadIdx.y;
    for(int j=0;j<32;j+=8){int yy=y+j;if(x<cols&&yy<rows)tile[threadIdx.y+j][threadIdx.x]=src[(long long)yy*cols+x];}
    __syncthreads();
    x=by+threadIdx.x;y=bx+threadIdx.y;
    for(int j=0;j<32;j+=8){int yy=y+j;if(x<rows&&yy<cols)dst[(long long)yy*dstride+x]=tile[threadIdx.x][threadIdx.y+j];}
}
static void transpose_f32(float*dst,const float*src,int rows,int cols,int dstride,cudaStream_t st){
    transpose_f32_kernel<<<dim3((cols+31)/32,(rows+31)/32),dim3(32,8),0,st>>>(dst,src,rows,cols,dstride);
}

/* Exact replica of the CPU reference matmul_fp8 (quant.h): per output row,
 * per 128-column block a sequential fp32 accumulate of e4m3(w)*x, then a
 * DOUBLE accumulate of block*scale across blocks, cast to float. Separately
 * rounded mul/add (the CPU build does not contract to FMA), so results are
 * bitwise the reference. x is the fp8-qdq'd activation (done on the host).
 * One thread per (row, token); tokens are the fast dimension so the 64
 * threads of a token block broadcast-read the same weight bytes. */
__device__ __forceinline__ float e4m3_dev(uint8_t b){
    int s=b>>7,e=(b>>3)&15,m=b&7;
    float v;
    if(e==15&&m==7)v=__int_as_float(0x7fc00000);          /* nan */
    else if(e==0)v=ldexpf((float)m/8.f,-6);
    else v=ldexpf(1.f+(float)m/8.f,e-7);
    return s?-v:v;
}
__global__ void fp8_ref_matmul_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const uint8_t*wr=w+(long long)o*cols;
    const float*xs=x+(long long)t*cols;
    int nblk=(cols+127)/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    double a=0.0;
    for(int bi=0;bi*128<cols;bi++){
        int base=bi*128,blen=cols-base<128?cols-base:128;
        float acc=0.f;
        for(int i=base;i<base+blen;i++)acc=__fadd_rn(acc,__fmul_rn(e4m3_dev(wr[i]),xs[i]));
        a=__dadd_rn(a,__dmul_rn((double)acc,(double)scl[bi]));
    }
    y[(long long)t*rows+o]=__double2float_rn(a);
}

/* rows8-PACKED replica: the resident CPU layout stores 8 consecutive rows'
 * bytes per column ((tile*cols+col)*8 + r) and the AVX2 reference kernel
 * accumulates sum += (x*w)*scale in fp32 sequentially over ALL columns (no
 * per-block double). Same order here, so bitwise the AVX2 path. */
__global__ void fp8_ref_matmul_rows8_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const float*xs=x+(long long)t*cols;
    int nblk=cols/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    long long tile=o>>3;int r=o&7;
    float sum=0.f;
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        for(int i=base;i<base+128;i++){
            float v=e4m3_dev(w[((tile*cols)+i)*8+r]);
            sum=__fadd_rn(sum,__fmul_rn(__fmul_rn(xs[i],v),sc));
        }
    }
    y[(long long)t*rows+o]=sum;
}

/* DSV4_IDX_COAL twins of the two replicas: x_t [cols][tokens], a warp's
 * 32 tokens in one line per column. The same per-thread sequence. */
__global__ void fp8_ref_matmul_xt_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x_t,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const uint8_t*wr=w+(long long)o*cols;
    const float*xs=x_t+t;
    int nblk=(cols+127)/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    double a=0.0;
    for(int bi=0;bi*128<cols;bi++){
        int base=bi*128,blen=cols-base<128?cols-base:128;
        float acc=0.f;
        for(int i=base;i<base+blen;i++)acc=__fadd_rn(acc,__fmul_rn(e4m3_dev(wr[i]),xs[(long long)i*tokens]));
        a=__dadd_rn(a,__dmul_rn((double)acc,(double)scl[bi]));
    }
    y[(long long)t*rows+o]=__double2float_rn(a);
}
__global__ void fp8_ref_matmul_rows8_xt_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x_t,int rows,int cols,int tokens){
    int t=threadIdx.x,o=blockIdx.x;
    if(t>=tokens||o>=rows)return;
    const float*xs=x_t+t;
    int nblk=cols/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    long long tile=o>>3;int r=o&7;
    float sum=0.f;
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        for(int i=base;i<base+128;i++){
            float v=e4m3_dev(w[((tile*cols)+i)*8+r]);
            sum=__fadd_rn(sum,__fmul_rn(__fmul_rn(xs[(long long)i*tokens],v),sc));
        }
    }
    y[(long long)t*rows+o]=sum;
}
/* DSV4_IDX_TILE (1): the packed replica a tile (8 rows) per block: one
 * 8-byte load per column gives the tile's weight bytes, the activation
 * load is shared by the 8 rows' sums. Each (row, token) sum in its order. */
__global__ void fp8_ref_matmul_rows8_tile_kernel(float*y,const uint8_t*w,const float*bscale,
        const float*x_t,int rows,int cols,int tokens){
    int t=threadIdx.x;long long tile=blockIdx.x;
    if(t>=tokens||tile*8>=rows)return;
    const float*xs=x_t+t;
    int nblk=cols/128;
    const float*scl=bscale+(long long)((tile*8)/128)*nblk;
    const unsigned long long*wt=(const unsigned long long*)(w+tile*cols*8);
    float sum[8];
    #pragma unroll
    for(int r=0;r<8;r++)sum[r]=0.f;
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        for(int i=base;i<base+128;i++){
            unsigned long long w8=wt[i];
            float xv=xs[(long long)i*tokens];
            #pragma unroll
            for(int r=0;r<8;r++){
                float v=e4m3_dev((uint8_t)(w8>>(8*r)));
                sum[r]=__fadd_rn(sum[r],__fmul_rn(__fmul_rn(xv,v),sc));
            }
        }
    }
    #pragma unroll
    for(int r=0;r<8;r++)y[(long long)t*rows+tile*8+r]=sum[r];
}
static int idx_tile_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_IDX_TILE");on=!(e&&*e=='0');}return on;}
/* the replica over x [tokens][cols] on the stream: transposed into x_t when
 * DSV4_IDX_COAL (and the scratch) allows, the row-major kernels otherwise */
static int fp8_ref_matmul_launch(Dev*c,float*y,const uint8_t*w,const float*bscale,const float*x,float*x_t,
        int rows,int cols,int packed_rows8,int tokens){
    int threads=tokens<32?32:((tokens+31)/32)*32;
    if(x_t){
        transpose_f32(x_t,x,tokens,cols,tokens,c->stream);
        if(packed_rows8&&idx_tile_on()&&rows%8==0)
            fp8_ref_matmul_rows8_tile_kernel<<<rows/8,threads,0,c->stream>>>(y,w,bscale,x_t,rows,cols,tokens);
        else if(packed_rows8)
            fp8_ref_matmul_rows8_xt_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x_t,rows,cols,tokens);
        else
            fp8_ref_matmul_xt_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x_t,rows,cols,tokens);
    }else if(packed_rows8)
        fp8_ref_matmul_rows8_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x,rows,cols,tokens);
    else
        fp8_ref_matmul_kernel<<<rows,threads,0,c->stream>>>(y,w,bscale,x,rows,cols,tokens);
    return ok(cudaGetLastError(),"fp8 ref matmul launch");
}

extern "C" int dsv4_cuda_fp8_ref_matmul(int device,const uint8_t*w,const float*bscale,
        int rows,int cols,int packed_rows8,const float*x,int tokens,float*y){
    Dev*c=ctx(device);
    if(!c||!w||!bscale||!x||!y||rows<1||cols<1||cols%128||tokens<1||tokens>1024||
       (packed_rows8&&rows%8))return 0;
    size_t wb=(size_t)rows*cols,sb=(size_t)((rows+127)/128)*(cols/128)*4,
           xb=(size_t)tokens*cols*4,yb=(size_t)tokens*rows*4;
    if(!ok(cudaSetDevice(device),"select fp8 ref matmul device")||
       !buf((void**)&c->p1,&c->p1cap,wb)||!buf((void**)&c->aux1,&c->aux1cap,sb)||
       !buf((void**)&c->p2,&c->p2cap,xb)||!buf((void**)&c->p4,&c->p4cap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->p1,w,wb,cudaMemcpyHostToDevice,c->stream),"fp8 ref weight upload")||
       !ok(cudaMemcpyAsync(c->aux1,bscale,sb,cudaMemcpyHostToDevice,c->stream),"fp8 ref scale upload")||
       !ok(cudaMemcpyAsync(c->p2,x,xb,cudaMemcpyHostToDevice,c->stream),"fp8 ref activation upload"))return 0;
    float*xt=(idx_coal_on()&&buf((void**)&c->p3,&c->p3cap,xb))?c->p3:nullptr;
    return fp8_ref_matmul_launch(c,c->p4,(const uint8_t*)c->p1,c->aux1,c->p2,xt,rows,cols,packed_rows8,tokens)&&
           ok(cudaMemcpyAsync(y,c->p4,yb,cudaMemcpyDeviceToHost,c->stream),"fp8 ref matmul download")&&
           ok(cudaStreamSynchronize(c->stream),"fp8 ref matmul sync");
}

/* ---- Device-resident indexer selection (V4_IDX_FUSED) ----------------------
 * Per layer the card keeps the indexer's wq_b bytes and f32 block scales in
 * the CPU reference layout, the bf16 head-weight rows, and the compressed keys
 * (append-only; re-seeded when the owner changes or the count regresses). One
 * round trip per selection: projection replica -> query prep (bf16 round,
 * RoPE, hadamard, fp4 round trip) -> head weights -> scoring; the host keeps
 * the sort. The arithmetic mirrors the GCC build of the reference, which fuses
 * the head-weight accumulate and the cosine products of the RoPE into FMAs
 * (read off the binary) and nothing else, so the scores are bitwise the
 * CPU's and the top-k boundary cannot move. */
static uint8_t *g_idx_w[DSV4_KV_CACHE_LAYERS];
static float *g_idx_s[DSV4_KV_CACHE_LAYERS];
static uint16_t *g_idx_hw[DSV4_KV_CACHE_LAYERS];
static const void *g_idx_w_src[DSV4_KV_CACHE_LAYERS];
static int g_idx_rows[DSV4_KV_CACHE_LAYERS],g_idx_cols[DSV4_KV_CACHE_LAYERS],g_idx_packed[DSV4_KV_CACHE_LAYERS],
           g_idx_heads[DSV4_KV_CACHE_LAYERS],g_idx_hidden[DSV4_KV_CACHE_LAYERS],g_idx_dev[DSV4_KV_CACHE_LAYERS];
static float *g_idx_keys[DSV4_KV_CACHE_LAYERS];
static float *g_idx_keys_t[DSV4_KV_CACHE_LAYERS];   /* DSV4_IDX_COAL: [dim][cap] copy, refreshed on append */
static int g_idx_keys_cap[DSV4_KV_CACHE_LAYERS],g_idx_keys_n[DSV4_KV_CACHE_LAYERS],g_idx_keys_dim[DSV4_KV_CACHE_LAYERS];
static unsigned long long g_idx_keys_owner[DSV4_KV_CACHE_LAYERS];
static void *g_idx_stage;static size_t g_idx_stage_cap;

__device__ __forceinline__ float bf16_round_dev(float v){
    unsigned b=__float_as_uint(v);
    if((b&0x7f800000u)!=0x7f800000u){unsigned tie=(b>>16)&1u;b+=0x7fffu+tie;}
    return __uint_as_float(b&0xffff0000u);
}
__constant__ float c_idx_e2m1[16]={0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f,0.f,-0.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f};

/* Query prep in place, grid (heads, tokens), block = dim threads, dim floats of
 * shared memory: bf16 round, RoPE on the last rope_dim (rope_fma 1 = the
 * cosine product fused, the GCC build's vfmaddsub; 2 = the sine product fused;
 * 0 = none), hadamard butterflies + scale + bf16 round, fp4 (e2m1 per 32,
 * e8m0 scale) round trip + bf16 round. Same operations in the same order as
 * indexer_prepare_queries. */
__global__ void indexer_prep_kernel(float*q,const float*cosv,const float*sinv,int heads,int dim,
        int rope_dim,float had_scale,int rope_fma){
    int h=blockIdx.x,t=blockIdx.y,i=threadIdx.x,pairs=rope_dim/2;
    float*row=q+((long long)t*heads+h)*dim;
    extern __shared__ float s[];
    __shared__ float bsc[32];
    s[i]=bf16_round_dev(row[i]);
    __syncthreads();
    if(i<pairs){
        int base=dim-rope_dim+2*i;
        float re=s[base],im=s[base+1];
        float c=cosv[(long long)t*pairs+i],sn=sinv[(long long)t*pairs+i];
        float a,b;
        if(rope_fma==1){a=fmaf(re,c,-__fmul_rn(im,sn));b=fmaf(im,c,__fmul_rn(re,sn));}
        else if(rope_fma==2){a=fmaf(-im,sn,__fmul_rn(re,c));b=fmaf(re,sn,__fmul_rn(im,c));}
        else{a=__fsub_rn(__fmul_rn(re,c),__fmul_rn(im,sn));b=__fadd_rn(__fmul_rn(re,sn),__fmul_rn(im,c));}
        s[base]=bf16_round_dev(a);s[base+1]=bf16_round_dev(b);
    }
    __syncthreads();
    for(int width=1;width<dim;width*=2){
        if(i<dim/2){
            int base=(i/width)*2*width,j=i-(i/width)*width;
            float l=s[base+j],r=s[base+width+j];
            s[base+j]=__fadd_rn(l,r);s[base+width+j]=__fsub_rn(l,r);
        }
        __syncthreads();
    }
    s[i]=bf16_round_dev(__fmul_rn(s[i],had_scale));
    __syncthreads();
    if(i<dim/32){
        float m=0.f;
        for(int k=0;k<32;k++)m=fmaxf(m,fabsf(s[i*32+k]));
        m=fmaxf(m,6.0f*ldexpf(1.0f,-126));
        int e;float fr=frexpf(__fdiv_rn(m,6.0f),&e);
        int ex=(fr==0.5f)?e-1:e;
        if(ex<-127)ex=-127;
        if(ex>127)ex=127;
        bsc[i]=ldexpf(1.0f,ex);
    }
    __syncthreads();
    float sc=bsc[i>>5];
    float v=fmaxf(-6.0f,fminf(6.0f,__fdiv_rn(s[i],sc)));
    int best=0;float d=fabsf(__fsub_rn(v,c_idx_e2m1[0]));
    for(int code=1;code<16;code++){float cd=fabsf(__fsub_rn(v,c_idx_e2m1[code]));if(cd<d){d=cd;best=code;}}
    row[i]=bf16_round_dev(__fmul_rn(c_idx_e2m1[best],sc));
}

/* head_w[t][h] = (sum_c bf16(w[h][c]) * x[t][c]) * wscale, sequential fp32;
 * hw_fma 1 = the accumulate fused (vfmadd132ss in the build), 0 = separate. */
__device__ __forceinline__ void headw_step8(float&sum,uint4 wv,float4 x0,float4 x1,int hw_fma){
    float w0=__uint_as_float(wv.x<<16),w1=__uint_as_float(wv.x&0xffff0000u),
          w2=__uint_as_float(wv.y<<16),w3=__uint_as_float(wv.y&0xffff0000u),
          w4=__uint_as_float(wv.z<<16),w5=__uint_as_float(wv.z&0xffff0000u),
          w6=__uint_as_float(wv.w<<16),w7=__uint_as_float(wv.w&0xffff0000u);
    if(hw_fma){
        sum=fmaf(w0,x0.x,sum);sum=fmaf(w1,x0.y,sum);sum=fmaf(w2,x0.z,sum);sum=fmaf(w3,x0.w,sum);
        sum=fmaf(w4,x1.x,sum);sum=fmaf(w5,x1.y,sum);sum=fmaf(w6,x1.z,sum);sum=fmaf(w7,x1.w,sum);
    }else{
        sum=__fadd_rn(sum,__fmul_rn(w0,x0.x));sum=__fadd_rn(sum,__fmul_rn(w1,x0.y));
        sum=__fadd_rn(sum,__fmul_rn(w2,x0.z));sum=__fadd_rn(sum,__fmul_rn(w3,x0.w));
        sum=__fadd_rn(sum,__fmul_rn(w4,x1.x));sum=__fadd_rn(sum,__fmul_rn(w5,x1.y));
        sum=__fadd_rn(sum,__fmul_rn(w6,x1.z));sum=__fadd_rn(sum,__fmul_rn(w7,x1.w));
    }
}
__global__ void indexer_headw_kernel(float*hw,const uint16_t*w,const float*x,int heads,int hidden,
        float wscale,int hw_fma){
    int t=blockIdx.x,h=threadIdx.x;
    if(h>=heads)return;
    const uint16_t*row=w+(long long)h*hidden;const float*xi=x+(long long)t*hidden;
    float sum=0.f;
    if((hidden&7)==0&&!((uintptr_t)row&15)&&!((uintptr_t)xi&15)){
        /* 8 columns per step through 16-byte loads (the chain is latency-bound
         * on the loads, not the FMAs); the same elements in the same order. */
        const uint4*rv=(const uint4*)row;const float4*xv=(const float4*)xi;
        for(int c=0;c<hidden/8;c++)headw_step8(sum,rv[c],xv[2*c],xv[2*c+1],hw_fma);
    }else if(hw_fma)for(int c=0;c<hidden;c++)sum=fmaf(__uint_as_float((unsigned)row[c]<<16),xi[c],sum);
    else for(int c=0;c<hidden;c++)sum=__fadd_rn(sum,__fmul_rn(__uint_as_float((unsigned)row[c]<<16),xi[c]));
    hw[(long long)t*heads+h]=__fmul_rn(sum,wscale);
}
/* One token's head weights with sixteen lanes per head: the lanes fetch the
 * head's next 128 columns (256 bytes, one coalesced load) into shared
 * memory and lane 0 runs the row's serial chain from there in the same
 * element order as the kernel above (bitwise). The direct kernel's one
 * thread per head is latency-bound on 512 strided 16-byte loads (0.115
 * ms per layer at 64 heads x 4096); here each chunk costs one load and
 * 128 steps of the chain. heads*16 threads per block (heads <= 64),
 * heads*272 bytes of shared memory. */
#define HEADW_LANES 16
#define HEADW_CHUNK (HEADW_LANES*8)
#define HEADW_PAD (HEADW_LANES*16+16)
__global__ void indexer_headw_lanes_kernel(float*hw,const uint16_t*w,const float*x,int heads,int hidden,
        float wscale,int hw_fma){
    extern __shared__ unsigned char headw_sh[];
    int t=blockIdx.x,h=threadIdx.x/HEADW_LANES,lane=threadIdx.x%HEADW_LANES;
    if(h>=heads)return;
    const uint16_t*row=w+(long long)h*hidden;const float4*xv=(const float4*)(x+(long long)t*hidden);
    uint4*my=(uint4*)(headw_sh+(size_t)h*HEADW_PAD);
    float sum=0.f;
    for(int c0=0;c0<hidden;c0+=HEADW_CHUNK){
        my[lane]=*(const uint4*)(row+c0+lane*8);
        __syncwarp();
        if(lane==0){
            #pragma unroll
            for(int part=0;part<HEADW_LANES;part++){
                int c=(c0>>3)+part;
                headw_step8(sum,my[part],xv[2*c],xv[2*c+1],hw_fma);
            }
        }
        __syncwarp();
    }
    if(lane==0)hw[(long long)t*heads+h]=__fmul_rn(sum,wscale);
}

static int h2d(Dev*c,void*dst,const void*src,size_t n,const char*what);

/* Batch of one: one thread per output row (the per-token kernels above give a
 * block of 32 threads to each row and leave 31 idle), the same per-row
 * accumulation. */
__global__ void fp8_ref_matmul_rows8_t1_kernel(float*y,const uint8_t*w,const float*bscale,const float*x,int rows,int cols){
    __shared__ float tab[256];   /* e4m3_dev's values, one load per element instead of its ldexpf */
    for(int i=threadIdx.x;i<256;i+=blockDim.x)tab[i]=e4m3_dev((uint8_t)i);
    __syncthreads();
    int o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=rows)return;
    int nblk=cols/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    long long tile=o>>3;int r=o&7;
    float sum=0.f;
    /* The chain is one thread's serial fadd (the reference order); what
     * hides its load latency is loads in flight: the inner 128 unrolled by
     * 16 lets the compiler issue the byte loads ahead of the chain
     * (DSV4_IDX_KPROF: the 8 MB layer's proj time, before 0.18 ms). */
    for(int base=0;base<cols;base+=128){
        float sc=scl[base/128];
        #pragma unroll 16
        for(int i=base;i<base+128;i++){
            float v=tab[w[((tile*cols)+i)*8+r]];
            sum=__fadd_rn(sum,__fmul_rn(__fmul_rn(x[i],v),sc));
        }
    }
    y[o]=sum;
}
__global__ void fp8_ref_matmul_t1_kernel(float*y,const uint8_t*w,const float*bscale,const float*x,int rows,int cols){
    __shared__ float tab[256];
    for(int i=threadIdx.x;i<256;i+=blockDim.x)tab[i]=e4m3_dev((uint8_t)i);
    __syncthreads();
    int o=blockIdx.x*blockDim.x+threadIdx.x;
    if(o>=rows)return;
    const uint8_t*wr=w+(long long)o*cols;
    int nblk=(cols+127)/128;
    const float*scl=bscale+(long long)(o/128)*nblk;
    double a=0.0;
    for(int bi=0;bi*128<cols;bi++){
        int base=bi*128,blen=cols-base<128?cols-base:128;
        float acc=0.f;
        #pragma unroll 16
        for(int i=base;i<base+blen;i++)acc=__fadd_rn(acc,__fmul_rn(tab[wr[i]],x[i]));
        a=__dadd_rn(a,__dmul_rn((double)acc,(double)scl[bi]));
    }
    y[o]=__double2float_rn(a);
}

/* Few tokens: a block of `heads` threads per (token, candidate). Thread h runs
 * head h's dot sequentially over dim, then thread 0 folds the heads in order:
 * the reference's arithmetic with the independent heads side by side instead
 * of one thread per candidate walking 820 strided key rows. Shared q is
 * padded to dim+1 so the per-head rows sit in distinct banks. */
__global__ void indexer_score_small_kernel(float*scores,const float*queries,const float*keys,
        const float*head_w,const int*counts,int heads,int dim,int count){
    int t=blockIdx.y,c=blockIdx.x,h=threadIdx.x;
    extern __shared__ float sm_[];
    float*qs=sm_;                      /* heads x (dim+1) */
    float*ks=sm_+(size_t)heads*(dim+1);  /* dim */
    float*dots=ks+dim;                 /* heads */
    int qn=heads*dim;
    for(int i=h;i<qn;i+=blockDim.x){int hh=i/dim,d=i-hh*dim;qs[hh*(dim+1)+d]=queries[(long long)t*qn+i];}
    for(int i=h;i<dim;i+=blockDim.x)ks[i]=keys[(long long)c*dim+i];
    __syncthreads();
    {
        const float*q=qs+h*(dim+1);
        float dot=0.f;
        for(int i=0;i<dim;i++)dot=__fadd_rn(dot,__fmul_rn(q[i],ks[i]));
        dots[h]=dot;
    }
    __syncthreads();
    if(!h){
        float score=0.f;
        if(c<counts[t])
            for(int hh=0;hh<heads;hh++)
                score=__fadd_rn(score,__fmul_rn(fmaxf(dots[hh],0.f),head_w[(long long)t*heads+hh]));
        scores[(long long)t*count+c]=score;
    }
}

static int g_idx_kprof=-1;

extern "C" int dsv4_cuda_indexer_resident(int device,int layer,const uint8_t*w,const float*bscale,int rows,int cols,
        int packed_rows8,const uint16_t*head_w,int heads,int hidden){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!w||!bscale||!head_w||rows<1||cols<1||cols%128||heads<1||hidden<1||
       (packed_rows8&&rows%8)||!ok(cudaSetDevice(device),"select indexer resident device"))return 0;
    if(g_idx_w[layer]&&g_idx_w_src[layer]==w&&g_idx_dev[layer]==device&&g_idx_rows[layer]==rows&&
       g_idx_cols[layer]==cols&&g_idx_heads[layer]==heads&&g_idx_hidden[layer]==hidden)return 1;
    if(g_idx_w[layer]){cudaFree(g_idx_w[layer]);g_idx_w[layer]=NULL;}
    if(g_idx_s[layer]){cudaFree(g_idx_s[layer]);g_idx_s[layer]=NULL;}
    if(g_idx_hw[layer]){cudaFree(g_idx_hw[layer]);g_idx_hw[layer]=NULL;}
    size_t wb=(size_t)rows*cols,sb=(size_t)((rows+127)/128)*(cols/128)*4,hb=(size_t)heads*hidden*2;
    if(!alloc_ok(cudaMalloc((void**)&g_idx_w[layer],wb),"indexer weight residency")||
       !alloc_ok(cudaMalloc((void**)&g_idx_s[layer],sb),"indexer scale residency")||
       !alloc_ok(cudaMalloc((void**)&g_idx_hw[layer],hb),"indexer head-weight residency")||
       !h2d(c,g_idx_w[layer],w,wb,"indexer weight upload")||!h2d(c,g_idx_s[layer],bscale,sb,"indexer scale upload")||
       !h2d(c,g_idx_hw[layer],head_w,hb,"indexer head-weight upload")){
        if(g_idx_w[layer])cudaFree(g_idx_w[layer]);
        if(g_idx_s[layer])cudaFree(g_idx_s[layer]);
        if(g_idx_hw[layer])cudaFree(g_idx_hw[layer]);
        g_idx_w[layer]=NULL;g_idx_s[layer]=NULL;g_idx_hw[layer]=NULL;return 0;}
    g_idx_w_src[layer]=w;g_idx_rows[layer]=rows;g_idx_cols[layer]=cols;g_idx_packed[layer]=packed_rows8;
    g_idx_heads[layer]=heads;g_idx_hidden[layer]=hidden;g_idx_dev[layer]=device;
    return 1;
}

/* Rows [n, count) of the host's append-only key cache go up; count 0, another
 * owner, another dim or a count below what is resident re-seed from row 0. */
extern "C" int dsv4_cuda_indexer_keys_sync(int device,int layer,unsigned long long owner,const float*keys,int count,int dim){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||count<0||dim<1||(count&&!keys)||
       !ok(cudaSetDevice(device),"select indexer key device"))return 0;
    if(!count||g_idx_keys_owner[layer]!=owner||g_idx_keys_dim[layer]!=dim||count<g_idx_keys_n[layer]||
       (g_idx_keys[layer]&&g_idx_dev[layer]!=device)){
        g_idx_keys_n[layer]=0;g_idx_keys_owner[layer]=owner;g_idx_keys_dim[layer]=dim;}
    if(!count)return 1;
    if(g_idx_keys_cap[layer]<count){
        int cap=g_idx_keys_cap[layer]?g_idx_keys_cap[layer]:512;
        while(cap<count)cap*=2;
        float*grown=NULL;
        if(!alloc_ok(cudaMalloc((void**)&grown,(size_t)cap*dim*4),"indexer key residency"))return 0;
        int have=g_idx_keys_n[layer];
        if(g_idx_keys[layer]&&have&&
           !ok(cudaMemcpyAsync(grown,g_idx_keys[layer],(size_t)have*dim*4,cudaMemcpyDeviceToDevice,c->stream),"indexer key grow copy")){
            cudaFree(grown);return 0;}
        if(g_idx_keys[layer]){cudaStreamSynchronize(c->stream);cudaFree(g_idx_keys[layer]);}
        g_idx_keys[layer]=grown;g_idx_keys_cap[layer]=cap;
        if(g_idx_keys_t[layer]){cudaFree(g_idx_keys_t[layer]);g_idx_keys_t[layer]=nullptr;}}
    int have=g_idx_keys_n[layer];
    if(count>have){
        if(!ok(cudaMemcpyAsync(g_idx_keys[layer]+(size_t)have*dim,keys+(size_t)have*dim,(size_t)(count-have)*dim*4,
                               cudaMemcpyHostToDevice,c->stream),"indexer key append"))return 0;
        g_idx_keys_n[layer]=count;
        /* DSV4_IDX_COAL: the [dim][cap] copy, the whole prefix again (an
         * append is a few rows; the copy is a few hundred KB) */
        if(idx_coal_on()){
            if(!g_idx_keys_t[layer]&&
               !alloc_ok(cudaMalloc((void**)&g_idx_keys_t[layer],(size_t)g_idx_keys_cap[layer]*dim*4),"indexer key transpose"))
                g_idx_keys_t[layer]=nullptr;
            if(g_idx_keys_t[layer])transpose_f32(g_idx_keys_t[layer],g_idx_keys[layer],count,dim,g_idx_keys_cap[layer],c->stream);
        }}
    return 1;
}

/* x [tokens][cols] fp8-qdq'd query ranks, inputs [tokens][hidden], cosv/sinv
 * [tokens][rope_dim/2], counts[tokens]; scores [tokens][count]. One staged
 * upload, four launches, one download, one sync. */
extern "C" int dsv4_cuda_indexer_select_fused(int device,int layer,const float*x,const float*inputs,const float*cosv,
        const float*sinv,const int*counts,int tokens,int heads,int dim,int rope_dim,int hidden,int count,
        float had_scale,float wscale,int rope_fma,int hw_fma,float*scores,float*queries_out,float*head_w_out){
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_idx_w[layer]||!g_idx_keys[layer]||!x||!inputs||!cosv||!sinv||
       !counts||!scores||tokens<1||tokens>1024||heads<1||heads>1024||dim<32||dim>1024||(dim&(dim-1))||
       rope_dim<2||rope_dim>dim||(rope_dim&1)||hidden<1||count<1||count>g_idx_keys_n[layer]||
       g_idx_keys_dim[layer]!=dim||g_idx_rows[layer]!=heads*dim||g_idx_heads[layer]!=heads||
       g_idx_hidden[layer]!=hidden)return 0;
    int cols=g_idx_cols[layer],rows=g_idx_rows[layer],pairs=rope_dim/2;
    size_t xb=(size_t)tokens*cols*4,ib=(size_t)tokens*hidden*4,rb=(size_t)tokens*pairs*4,cb=(size_t)tokens*sizeof(int);
    size_t stage=xb+ib+2*rb+cb,qb=(size_t)tokens*rows*4,sb=(size_t)tokens*count*4,hb=(size_t)tokens*heads*4,sm=(size_t)rows*4;
    if(sm>96*1024||!ok(cudaSetDevice(device),"select fused indexer device"))return 0;
    if(g_idx_stage_cap<stage){
        if(g_idx_stage)cudaFreeHost(g_idx_stage);
        g_idx_stage=NULL;g_idx_stage_cap=0;
        if(!ok(cudaHostAlloc(&g_idx_stage,stage,cudaHostAllocPortable),"indexer staging allocation"))return 0;
        g_idx_stage_cap=stage;}
    char*st=(char*)g_idx_stage;
    memcpy(st,x,xb);memcpy(st+xb,inputs,ib);memcpy(st+xb+ib,cosv,rb);memcpy(st+xb+ib+rb,sinv,rb);
    memcpy(st+xb+ib+2*rb,counts,cb);
    if(!buf((void**)&c->p1,&c->p1cap,stage)||!buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p4,&c->p4cap,sb)||
       !buf((void**)&c->aux3,&c->aux3cap,hb))return 0;
    if(!ok(cudaMemcpyAsync(c->p1,st,stage,cudaMemcpyHostToDevice,c->stream),"fused indexer upload"))return 0;
    const char*base=(const char*)c->p1;
    const float*dx=(const float*)base;const float*din=(const float*)(base+xb);
    const float*dcos=(const float*)(base+xb+ib);const float*dsin=(const float*)(base+xb+ib+rb);
    const int*dcnt=(const int*)(base+xb+ib+2*rb);
    if(g_idx_kprof<0){const char*e=getenv("DSV4_IDX_KPROF");g_idx_kprof=e&&*e&&*e!='0';}
    cudaEvent_t ev[6];
    if(g_idx_kprof)for(int i=0;i<6;i++)cudaEventCreate(&ev[i]);
    if(g_idx_kprof)cudaEventRecord(ev[0],c->stream);
    if(tokens==1){
        /* 64-thread blocks: 8192 rows as 128 blocks over the card's SMs
         * instead of 32 (each thread's serial chain is unchanged) */
        if(g_idx_packed[layer])
            fp8_ref_matmul_rows8_t1_kernel<<<(rows+63)/64,64,0,c->stream>>>(c->p2,g_idx_w[layer],g_idx_s[layer],dx,rows,cols);
        else
            fp8_ref_matmul_t1_kernel<<<(rows+63)/64,64,0,c->stream>>>(c->p2,g_idx_w[layer],g_idx_s[layer],dx,rows,cols);
        if(!ok(cudaGetLastError(),"fused indexer projection launch"))return 0;
    }else{
        float*xt=(idx_coal_on()&&buf((void**)&c->p3,&c->p3cap,xb))?c->p3:nullptr;
        if(!fp8_ref_matmul_launch(c,c->p2,g_idx_w[layer],g_idx_s[layer],dx,xt,rows,cols,g_idx_packed[layer],tokens))return 0;
    }
    if(g_idx_kprof)cudaEventRecord(ev[1],c->stream);
    indexer_prep_kernel<<<dim3(heads,tokens),dim,(size_t)dim*4,c->stream>>>(c->p2,dcos,dsin,heads,dim,rope_dim,had_scale,rope_fma);
    if(!ok(cudaGetLastError(),"fused indexer prep launch"))return 0;
    if(g_idx_kprof)cudaEventRecord(ev[2],c->stream);
    if(tokens==1&&hidden%HEADW_CHUNK==0&&heads*HEADW_LANES<=1024&&!((uintptr_t)g_idx_hw[layer]&15)&&!((uintptr_t)din&15))
        indexer_headw_lanes_kernel<<<1,heads*HEADW_LANES,(size_t)heads*HEADW_PAD,c->stream>>>(c->aux3,g_idx_hw[layer],din,heads,hidden,wscale,hw_fma);
    else
        indexer_headw_kernel<<<tokens,heads,0,c->stream>>>(c->aux3,g_idx_hw[layer],din,heads,hidden,wscale,hw_fma);
    if(!ok(cudaGetLastError(),"fused indexer head-weight launch"))return 0;
    if(g_idx_kprof)cudaEventRecord(ev[3],c->stream);
    size_t sm_small=((size_t)heads*(dim+1)+dim+heads)*4;
    if(tokens<=4&&sm_small<=48*1024){
        indexer_score_small_kernel<<<dim3(count,tokens),heads,sm_small,c->stream>>>(c->p4,c->p2,g_idx_keys[layer],c->aux3,dcnt,heads,dim,count);
    }else{
        if(idx_coal_on()&&g_idx_keys_t[layer]){
            if(sm>48*1024&&cudaFuncSetAttribute(indexer_score_coal_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess)return 0;
            indexer_score_coal_kernel<<<dim3((count+127)/128,tokens),128,sm,c->stream>>>(c->p4,c->p2,g_idx_keys_t[layer],g_idx_keys_cap[layer],c->aux3,dcnt,heads,dim,count);
        }else{
            if(sm>48*1024&&cudaFuncSetAttribute(indexer_score_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm)!=cudaSuccess)return 0;
            indexer_score_kernel<<<dim3((count+127)/128,tokens),128,sm,c->stream>>>(c->p4,c->p2,g_idx_keys[layer],c->aux3,dcnt,heads,dim,count);
        }
    }
    if(!ok(cudaGetLastError(),"fused indexer score launch"))return 0;
    if(g_idx_kprof)cudaEventRecord(ev[4],c->stream);
    if(!ok(cudaMemcpyAsync(scores,c->p4,sb,cudaMemcpyDeviceToHost,c->stream),"fused indexer score download"))return 0;
    if(queries_out&&!ok(cudaMemcpyAsync(queries_out,c->p2,qb,cudaMemcpyDeviceToHost,c->stream),"fused indexer query download"))return 0;
    if(head_w_out&&!ok(cudaMemcpyAsync(head_w_out,c->aux3,hb,cudaMemcpyDeviceToHost,c->stream),"fused indexer head-weight download"))return 0;
    if(g_idx_kprof)cudaEventRecord(ev[5],c->stream);
    int done=ok(cudaStreamSynchronize(c->stream),"fused indexer sync");
    if(g_idx_kprof){
        float ms[5];
        for(int i=0;i<5;i++)cudaEventElapsedTime(&ms[i],ev[i],ev[i+1]);
        fprintf(stderr,"idxkprof layer=%d tokens=%d cand=%d proj=%.3f prep=%.3f headw=%.3f score=%.3f down=%.3f ms\n",
                layer,tokens,count,ms[0],ms[1],ms[2],ms[3],ms[4]);
        for(int i=0;i<6;i++)cudaEventDestroy(ev[i]);
    }
    return done;
}

/* ---- Decode attention tail on the device (V4_WO_CHAIN) ---------------------
 * One token: the cached sparse attention (ring + compressed prefix, the
 * selected ordinals when the layer has an indexer) stays on the card for the
 * inverse RoPE + bf16 round, the grouped wo_a matvec (the decode's own mv
 * kernel: its result bitwise what the uploaded-activation call returns), the
 * bf16 round, and wo_b through the same mv kernel over its mirror (what the
 * engine's coli_fp8_matvec_ref dispatches to when the mirror exists), then
 * one download of the bf16-rounded hidden vector. Three syncs per layer
 * become one; the arithmetic is the separate calls' exactly. */
/* Per-head RoPE on the last rope_dim of each head's dim, bf16-rounded after:
 * the reference's coli_v4_rope_apply (sine * direction, direction +-1 exact)
 * + coli_bf16_round_array. fma_mode as the indexer prep. */
__global__ void attn_inv_rope_kernel(float*att,const float*cosv,const float*sinv,int dim,int rope_dim,int fma_mode,int inverse){
    int h=blockIdx.x,i=threadIdx.x,pairs=rope_dim/2;
    if(i>=pairs)return;
    float*v=att+(long long)h*dim+dim-rope_dim+2*i;
    float re=v[0],im=v[1],c=cosv[i],sn=inverse?-sinv[i]:sinv[i];
    float a,b;
    if(fma_mode==1){a=fmaf(re,c,-__fmul_rn(im,sn));b=fmaf(im,c,__fmul_rn(re,sn));}
    else if(fma_mode==2){a=fmaf(-im,sn,__fmul_rn(re,c));b=fmaf(re,sn,__fmul_rn(im,c));}
    else{a=__fsub_rn(__fmul_rn(re,c),__fmul_rn(im,sn));b=__fadd_rn(__fmul_rn(re,sn),__fmul_rn(im,c));}
    v[0]=bf16_round_dev(a);v[1]=bf16_round_dev(b);
}
/* the per-token RoPE of attn_inv_rope_kernel over a batch: blockIdx.y the
 * token (its own cos/sin row), blockIdx.x the head */
__global__ void attn_rope_batch_kernel(float*att,const float*cosv,const float*sinv,int dim,int rope_dim,int fma_mode,int inverse,int heads){
    int t=blockIdx.y,h=blockIdx.x,i=threadIdx.x,pairs=rope_dim/2;
    if(i>=pairs)return;
    float*v=att+((long long)t*heads+h)*dim+dim-rope_dim+2*i;
    const float*cr=cosv+(long long)t*pairs,*sr=sinv+(long long)t*pairs;
    float re=v[0],im=v[1],c=cr[i],sn=inverse?-sr[i]:sr[i];
    float a,b;
    if(fma_mode==1){a=fmaf(re,c,-__fmul_rn(im,sn));b=fmaf(im,c,__fmul_rn(re,sn));}
    else if(fma_mode==2){a=fmaf(-im,sn,__fmul_rn(re,c));b=fmaf(re,sn,__fmul_rn(im,c));}
    else{a=__fsub_rn(__fmul_rn(re,c),__fmul_rn(im,sn));b=__fadd_rn(__fmul_rn(re,sn),__fmul_rn(im,c));}
    v[0]=bf16_round_dev(a);v[1]=bf16_round_dev(b);
}
__global__ void bf16_round_ref_kernel(float*x,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)x[i]=bf16_round_dev(x[i]);
}
/* The decode query's per-head rmsnorm, the reference's arithmetic: one
 * sequential fp32 mean-square (fma_on: the accumulate fused, as the GCC
 * build contracts it), scale = 1 / sqrtf(ms / dim + eps), bf16 round. */
__global__ void head_rmsnorm_ref_kernel(float*q,int dim,float eps,int fma_on){
    int h=blockIdx.x;
    float*v=q+(long long)h*dim;
    extern __shared__ float sv[];
    __shared__ float sc;
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)sv[i]=v[i];
    __syncthreads();
    if(!threadIdx.x){
        float ms=0.f;
        if(fma_on)for(int i=0;i<dim;i++)ms=fmaf(sv[i],sv[i],ms);
        else for(int i=0;i<dim;i++)ms=__fadd_rn(ms,__fmul_rn(sv[i],sv[i]));
        sc=__fdiv_rn(1.0f,__fsqrt_rn(__fadd_rn(__fdiv_rn(ms,(float)dim),eps)));
    }
    __syncthreads();
    for(int i=threadIdx.x;i<dim;i+=blockDim.x)v[i]=bf16_round_dev(__fmul_rn(sv[i],sc));
}
/* The CPU reference matvec over a fmt-8 mirror, one thread per output row.
 * rows8 order: sum += (x*w)*scale sequentially over all columns (the AVX2
 * resident-layout kernel); block order: fp32 per 128-column block, double
 * across blocks. Both end with the reference's bf16 round of the output. */
/* V4_PREFILL_CHAIN: the batched prefill attention on the card end to end.
 * qa [tokens][q_rank] (host, the normalized query ranks) goes up once; the
 * wq_b GEMM (the batched dense path the host's matmul already takes), the
 * bf16 round, the per-head rmsnorm and the forward RoPE (the decode chain's
 * kernels, batched), the cached sparse attention over the chunk's kv rows,
 * the inverse RoPE and the grouped wo_a + wo_b follow on the stream; out
 * [tokens][hidden] comes down once (the host rounds it to bf16 as it does
 * for its own wo). Everything the host did between — the query's two
 * crossings of the link, its norm and RoPE, the attended's inverse RoPE —
 * is gone. 0 = refused (the host takes its usual path). */
extern "C" int dsv4_cuda_attn_prefill_chain(int device,int layer,const float*qa,int q_rank,Dsv4CudaTensor*wqb,float eps,int norm_fma,
        const float*icos,const float*isin,int rope_dim,int rope_fma,const float*chunk,int chunk_start,const float*sinks,
        const int*meta,const int*sel,int selstride,int abs_base,int comp_limit,int heads,int dim,float scale,int tokens,
        Dsv4CudaTensor*wa,int groups,Dsv4CudaTensor*wb,float*out){
    static int kprof=-1;
    if(kprof<0){const char*e=getenv("DSV4_PF_KPROF");kprof=e&&*e&&*e!='0';}
    std::chrono::steady_clock::time_point t_in;
    if(kprof)t_in=std::chrono::steady_clock::now();
    Dev*c=ctx(device);
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_kv_ring[layer]||!qa||!wqb||!icos||!isin||!chunk||!sinks||!meta||!out||
       !wa||!wb||q_rank<1||tokens<1||heads<1||dim<1||rope_dim<2||rope_dim>dim||(rope_dim&1)||!(scale>0.f)||
       wqb->device!=device||(wqb->fmt!=8&&wqb->fmt!=9)||wqb->I!=q_rank||(long long)wqb->O!=(long long)heads*dim||
       wa->device!=device||wb->device!=device||groups<1||wa->O%groups||(long long)wa->I*groups!=(long long)heads*dim||wb->I!=wa->O)return 0;
    if(comp_limit>0&&(!g_kv_comp[layer]||g_kv_comp_cap[layer]<comp_limit))return 0;
    int window=g_kv_ring_rows[layer],max_total=0,pairs=rope_dim/2;
    for(int t=0;t<tokens;t++){
        int wn=meta[3*t+1],cn=meta[3*t+2],tt=wn+cn;
        if(meta[3*t]<0||wn<1||wn>window||cn<0||cn>comp_limit||(sel&&cn>selstride)||(!sel&&cn>0&&cn>comp_limit))return 0;
        if(abs_base+meta[3*t]+wn>chunk_start+tokens)return 0;
        if(sel)for(int i=0;i<cn;i++){int o=sel[(long long)t*selstride+i];if(o<0||o>=comp_limit)return 0;}
        if(tt>max_total)max_total=tt;
    }
    size_t qab=(size_t)tokens*q_rank*4,rb=(size_t)tokens*pairs*4,qb=(size_t)tokens*heads*dim*4,cb=(size_t)tokens*dim*4,
           sb=(size_t)heads*4,mb=(size_t)tokens*3*sizeof(int),ib=sel?(size_t)tokens*selstride*sizeof(int):0,
           hb=(size_t)tokens*wb->O*4,sm=(size_t)max_total*4;
    if(sm>96*1024||!ok(cudaSetDevice(device),"select prefill chain device"))return 0;
    /* p1: qa | cos | sin (consumed before wo_b reuses p1); p2: q; p3: the chunk's kv rows; p4: attended; dx: the output */
    if(!buf((void**)&c->p1,&c->p1cap,qab+2*rb)||!buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p3,&c->p3cap,cb)||
       !buf((void**)&c->p4,&c->p4cap,qb)||!buf((void**)&c->aux1,&c->aux1cap,sb)||!buf((void**)&c->aux2,&c->aux2cap,mb)||
       (ib&&!buf((void**)&c->aux3,&c->aux3cap,ib))||!buf((void**)&c->dx,&c->xcap,hb))return 0;
    float*dqa=c->p1,*dcos=(float*)((char*)c->p1+qab),*dsin=(float*)((char*)c->p1+qab+rb);
    cudaEvent_t ev[8];
    if(kprof){for(int i=0;i<8;i++)cudaEventCreate(&ev[i]);cudaEventRecord(ev[0],c->stream);}
    /* DSV4_PIN_STAGE: the seven inputs as one block in the pinned staging,
     * one copy into p1 (all of it consumed before wo_b reuses p1); else
     * the seven pageable copies */
    size_t upb=qab+2*rb+cb+sb+mb+ib;
    char*stg=(char*)stage_get(c,upb>hb?upb:hb);
    const float*dchunk=c->p3,*dsink=c->aux1;const int*dmeta=(const int*)c->aux2,*dsel=ib?(const int*)c->aux3:nullptr;
    if(stg&&buf((void**)&c->p1,&c->p1cap,upb)){
        char*h=stg;
        memcpy(h,qa,qab);h+=qab;memcpy(h,icos,rb);h+=rb;memcpy(h,isin,rb);h+=rb;memcpy(h,chunk,cb);h+=cb;
        memcpy(h,sinks,sb);h+=sb;memcpy(h,meta,mb);h+=mb;if(ib)memcpy(h,sel,ib);
        if(!ok(cudaMemcpyAsync(c->p1,stg,upb,cudaMemcpyHostToDevice,c->stream),"prefill chain staged upload"))return 0;
        char*d=(char*)c->p1;
        dqa=(float*)d;dcos=(float*)(d+qab);dsin=(float*)(d+qab+rb);dchunk=(const float*)(d+qab+2*rb);
        dsink=(const float*)(d+qab+2*rb+cb);dmeta=(const int*)(d+qab+2*rb+cb+sb);dsel=ib?(const int*)(d+qab+2*rb+cb+sb+mb):nullptr;
    }else{
        stg=nullptr;
        if(!ok(cudaMemcpyAsync(dqa,qa,qab,cudaMemcpyHostToDevice,c->stream),"prefill chain qa upload")||
           !ok(cudaMemcpyAsync(dcos,icos,rb,cudaMemcpyHostToDevice,c->stream),"prefill chain cos upload")||
           !ok(cudaMemcpyAsync(dsin,isin,rb,cudaMemcpyHostToDevice,c->stream),"prefill chain sin upload")||
           !ok(cudaMemcpyAsync(c->p3,chunk,cb,cudaMemcpyHostToDevice,c->stream),"prefill chain kv upload")||
           !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"prefill chain sink upload")||
           !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"prefill chain meta upload")||
           (ib&&!ok(cudaMemcpyAsync(c->aux3,sel,ib,cudaMemcpyHostToDevice,c->stream),"prefill chain index upload")))return 0;
    }
    if(kprof)cudaEventRecord(ev[1],c->stream);
    /* the query projection: the same batched dense path dsv4_cuda_matmul_batch takes */
#ifdef COLI_DSV4_DEEPGEMM
    if(!dg_dense_batch_dispatch(c,wqb,dqa,c->p2,tokens))return 0;
#else
    run_mm_batch(wqb,dqa,c->p2,tokens,c->stream);if(!ok(cudaGetLastError(),"prefill chain wq_b launch"))return 0;
#endif
    if(kprof)cudaEventRecord(ev[2],c->stream);
    long long qn=(long long)tokens*heads*dim;
    bf16_round_ref_kernel<<<(unsigned)((qn+255)/256),256,0,c->stream>>>(c->p2,(int)qn);
    head_rmsnorm_ref_kernel<<<(unsigned)(tokens*heads),256,(size_t)dim*4,c->stream>>>(c->p2,dim,eps,norm_fma);
    attn_rope_batch_kernel<<<dim3(heads,tokens),pairs,0,c->stream>>>(c->p2,dcos,dsin,dim,rope_dim,rope_fma,0,heads);
    if(!ok(cudaGetLastError(),"prefill chain query launch"))return 0;
    if(kprof)cudaEventRecord(ev[3],c->stream);
    if(sel){
        if(sattn_v2_on())launch_sattn_v2<1>(heads,tokens,sm,c->stream,c->p4,c->p2,g_kv_ring[layer],dchunk,g_kv_comp[layer],dsink,dmeta,dsel,selstride,abs_base,chunk_start,window,dim,scale);
        else sparse_attn_batch_cached_idx_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(c->p4,c->p2,g_kv_ring[layer],dchunk,g_kv_comp[layer],dsink,dmeta,dsel,selstride,abs_base,chunk_start,window,heads,dim,scale);
    }else{
        if(sattn_v2_on())launch_sattn_v2<0>(heads,tokens,sm,c->stream,c->p4,c->p2,g_kv_ring[layer],dchunk,g_kv_comp[layer],dsink,dmeta,nullptr,0,abs_base,chunk_start,window,dim,scale);
        else sparse_attn_batch_cached_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(c->p4,c->p2,g_kv_ring[layer],dchunk,g_kv_comp[layer],dsink,dmeta,abs_base,chunk_start,window,heads,dim,scale);
    }
    if(!ok(cudaGetLastError(),"prefill chain attention launch"))return 0;
    if(kprof)cudaEventRecord(ev[4],c->stream);
    attn_rope_batch_kernel<<<dim3(heads,tokens),pairs,0,c->stream>>>(c->p4,dcos,dsin,dim,rope_dim,rope_fma,1,heads);
    if(!ok(cudaGetLastError(),"prefill chain inverse rope launch"))return 0;
    if(kprof)cudaEventRecord(ev[5],c->stream);
    Dsv4CudaActivation ctxa={c->p4,qn,device},outa={c->dx,(long long)tokens*wb->O,device};
    if(!dsv4_cuda_attention_output_batch(&ctxa,wa,wb,groups,tokens,&outa))return 0;
    if(kprof)cudaEventRecord(ev[6],c->stream);
    if(!ok(cudaMemcpyAsync(stg?(void*)stg:(void*)out,c->dx,hb,cudaMemcpyDeviceToHost,c->stream),"prefill chain output download"))return 0;
    if(kprof)cudaEventRecord(ev[7],c->stream);
    int done=ok(cudaStreamSynchronize(c->stream),"prefill chain sync");
    if(stg&&done)memcpy(out,stg,hb);
    if(kprof){
        float ms[7];
        for(int i=0;i<7;i++)cudaEventElapsedTime(&ms[i],ev[i],ev[i+1]);
        double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_in).count();
        fprintf(stderr,"pfkprof layer=%d tokens=%d total=%d up=%.3f qb=%.3f qk=%.3f attn=%.3f rope=%.3f wo=%.3f down=%.3f host=%.3f ms\n",
                layer,tokens,max_total,ms[0],ms[1],ms[2],ms[3],ms[4],ms[5],ms[6],host);
        for(int i=0;i<8;i++)cudaEventDestroy(ev[i]);
    }
    return done;
}
extern "C" int dsv4_cuda_attn_decode_chain(int device,int layer,const float*q,const float*chunk,int chunk_start,
        const float*sinks,const int*meta,const int*sel,int selstride,int abs_base,int comp_limit,int heads,int dim,
        float scale,const float*icos,const float*isin,int rope_dim,int rope_fma,Dsv4CudaTensor*wa,int groups,
        Dsv4CudaTensor*wb,float*out,float*att_out,float*oa_out,
        const float*qa,int q_rank,Dsv4CudaTensor*wqb,float eps,int norm_fma,float*q_out){
    Dev*c=ctx(device);
    if(qa&&(!wqb||wqb->device!=device||(wqb->fmt!=8&&wqb->fmt!=9)||wqb->I!=q_rank||(long long)wqb->O!=(long long)heads*dim||q_rank<1))return 0;
    if(!c||layer<0||layer>=DSV4_KV_CACHE_LAYERS||!g_kv_ring[layer]||(!q&&!qa)||!chunk||!sinks||!meta||!out||!icos||!isin||
       !wa||!wb||wa->device!=device||wb->device!=device||groups<1||wa->O%groups||(long long)wa->I*groups!=(long long)heads*dim||
       wb->I!=wa->O||(wb->fmt!=8&&wb->fmt!=9)||(wa->fmt!=8&&wa->fmt!=9)||
       abs_base<0||chunk_start<0||heads<1||dim<1||rope_dim<2||rope_dim>dim||(rope_dim&1)||!(scale>0.f))return 0;
    if(comp_limit>0&&(!g_kv_comp[layer]||g_kv_comp_cap[layer]<comp_limit))return 0;
    int window=g_kv_ring_rows[layer],wn=meta[1],cn=meta[2],total=wn+cn,pairs=rope_dim/2;
    if(meta[0]<0||wn<1||wn>window||cn<0||cn>comp_limit)return 0;
    if(abs_base+meta[0]+wn>chunk_start+1)return 0;
    if(sel){if(selstride<1||cn>selstride)return 0;for(int i=0;i<cn;i++){int o=sel[i];if(o<0||o>=comp_limit)return 0;}}
    size_t qb=(size_t)heads*dim*4,cb=(size_t)dim*4,sb=(size_t)heads*4,mb=3*sizeof(int),
           ib=sel?(size_t)cn*sizeof(int):0,rb=(size_t)pairs*4,sm=(size_t)total*4,
           ob=(size_t)wa->O*4,hb=(size_t)wb->O*4,qab=qa?(size_t)q_rank*4:0;
    if(sm>96*1024||!ok(cudaSetDevice(device),"select decode chain device"))return 0;
    if(!buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p4,&c->p4cap,qb)||!buf((void**)&c->p3,&c->p3cap,cb)||
       !buf((void**)&c->aux1,&c->aux1cap,sb)||!buf((void**)&c->aux2,&c->aux2cap,mb)||
       (ib&&!buf((void**)&c->aux3,&c->aux3cap,ib))||!buf((void**)&c->p1,&c->p1cap,2*rb+qab)||
       !buf((void**)&c->dy,&c->ycap,ob)||!buf((void**)&c->dx,&c->xcap,hb))return 0;
    static int kprof=-1;
    if(kprof<0){const char*e=getenv("DSV4_WO_KPROF");kprof=e&&*e&&*e!='0';}
    cudaEvent_t ev[7],ex[3];
    if(kprof){for(int i=0;i<7;i++)cudaEventCreate(&ev[i]);for(int i=0;i<3;i++)cudaEventCreate(&ex[i]);cudaEventRecord(ex[0],c->stream);}
    if(qa){
        /* q from qa on the card: the wq_b mv kernel (what coli_fp8_matvec_ref
         * dispatches to over the mirror), bf16 round, per-head rmsnorm,
         * forward RoPE + bf16 round — the host's q, never crossing. */
        const float*dqa=(const float*)((char*)c->p1+2*rb);
        if(!ok(cudaMemcpyAsync((void*)dqa,qa,qab,cudaMemcpyHostToDevice,c->stream),"decode chain qa upload")||
           !ok(cudaMemcpyAsync(c->p1,icos,rb,cudaMemcpyHostToDevice,c->stream),"decode chain cos upload")||
           !ok(cudaMemcpyAsync((char*)c->p1+rb,isin,rb,cudaMemcpyHostToDevice,c->stream),"decode chain sin upload"))return 0;
        if(!run_fp8_attn(wqb,dqa,c->p2,1,c->stream)){
        if(wqb->fmt==9)run_mv<9>((uint8_t*)wqb->w,wqb->scale,dqa,c->p2,wqb->O,wqb->I,1,c->stream);
        else run_mv<8>((uint8_t*)wqb->w,wqb->scale,dqa,c->p2,wqb->O,wqb->I,1,c->stream);}
        if(!ok(cudaGetLastError(),"decode chain wq_b launch"))return 0;
        bf16_round_ref_kernel<<<(heads*dim+255)/256,256,0,c->stream>>>(c->p2,heads*dim);
        head_rmsnorm_ref_kernel<<<heads,256,(size_t)dim*4,c->stream>>>(c->p2,dim,eps,norm_fma);
        if(!ok(cudaGetLastError(),"decode chain q norm launch"))return 0;
        attn_inv_rope_kernel<<<heads,pairs,0,c->stream>>>(c->p2,c->p1,(const float*)((char*)c->p1+rb),dim,rope_dim,rope_fma,0);
        if(!ok(cudaGetLastError(),"decode chain q rope launch"))return 0;
        if(q_out&&!ok(cudaMemcpyAsync(q_out,c->p2,qb,cudaMemcpyDeviceToHost,c->stream),"decode chain q download"))return 0;
    }
    if(kprof)cudaEventRecord(ex[1],c->stream);
    if(!qa&&!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"decode chain query upload"))return 0;
    if(
       !ok(cudaMemcpyAsync(c->p3,chunk,cb,cudaMemcpyHostToDevice,c->stream),"decode chain kv upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"decode chain sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"decode chain meta upload")||
       (ib&&!ok(cudaMemcpyAsync(c->aux3,sel,ib,cudaMemcpyHostToDevice,c->stream),"decode chain index upload"))||
       (!qa&&(!ok(cudaMemcpyAsync(c->p1,icos,rb,cudaMemcpyHostToDevice,c->stream),"decode chain cos upload")||
              !ok(cudaMemcpyAsync((char*)c->p1+rb,isin,rb,cudaMemcpyHostToDevice,c->stream),"decode chain sin upload"))))return 0;
    if(kprof)cudaEventRecord(ex[2],c->stream);
    if(sel&&sattn_v2_on())
        launch_sattn_v2<1>(heads,1,sm,c->stream,
            c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,
            cn,abs_base,chunk_start,window,dim,scale);
    else if(sel)
        sparse_attn_batch_cached_idx_kernel<<<dim3(heads,1),256,sm,c->stream>>>(
            c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,(int*)c->aux3,
            cn,abs_base,chunk_start,window,heads,dim,scale);
    else if(sattn_v2_on())
        launch_sattn_v2<0>(heads,1,sm,c->stream,
            c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,nullptr,0,
            abs_base,chunk_start,window,dim,scale);
    else
        sparse_attn_batch_cached_kernel<<<dim3(heads,1),256,sm,c->stream>>>(
            c->p4,c->p2,g_kv_ring[layer],c->p3,g_kv_comp[layer],c->aux1,(int*)c->aux2,
            abs_base,chunk_start,window,heads,dim,scale);
    if(!ok(cudaGetLastError(),"decode chain attention launch"))return 0;
    if(kprof)cudaEventRecord(ev[0],c->stream);
    attn_inv_rope_kernel<<<heads,pairs,0,c->stream>>>(c->p4,c->p1,(const float*)((char*)c->p1+rb),dim,rope_dim,rope_fma,1);
    if(!ok(cudaGetLastError(),"decode chain rope launch"))return 0;
    if(kprof)cudaEventRecord(ev[1],c->stream);
    if(!run_fp8_attn(wa,c->p4,c->dy,groups,c->stream)){
    if(wa->fmt==9)run_mv<9>((uint8_t*)wa->w,wa->scale,c->p4,c->dy,wa->O,wa->I,groups,c->stream);
    else run_mv<8>((uint8_t*)wa->w,wa->scale,c->p4,c->dy,wa->O,wa->I,groups,c->stream);}
    if(!ok(cudaGetLastError(),"decode chain wo_a launch"))return 0;
    if(kprof)cudaEventRecord(ev[2],c->stream);
    bf16_round_ref_kernel<<<(wa->O+255)/256,256,0,c->stream>>>(c->dy,wa->O);
    if(oa_out&&!ok(cudaMemcpyAsync(oa_out,c->dy,ob,cudaMemcpyDeviceToHost,c->stream),"decode chain oa download"))return 0;
    if(kprof)cudaEventRecord(ev[3],c->stream);
    if(!run_fp8_attn(wb,c->dy,c->dx,1,c->stream)){
    if(wb->fmt==9)run_mv<9>((uint8_t*)wb->w,wb->scale,c->dy,c->dx,wb->O,wb->I,1,c->stream);
    else run_mv<8>((uint8_t*)wb->w,wb->scale,c->dy,c->dx,wb->O,wb->I,1,c->stream);}
    if(!ok(cudaGetLastError(),"decode chain wo_b launch"))return 0;
    bf16_round_ref_kernel<<<(wb->O+255)/256,256,0,c->stream>>>(c->dx,wb->O);
    if(kprof)cudaEventRecord(ev[4],c->stream);
    if(!ok(cudaMemcpyAsync(out,c->dx,hb,cudaMemcpyDeviceToHost,c->stream),"decode chain output download"))return 0;
    if(att_out&&!ok(cudaMemcpyAsync(att_out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"decode chain attended download"))return 0;
    if(kprof)cudaEventRecord(ev[5],c->stream);
    int done=ok(cudaStreamSynchronize(c->stream),"decode chain sync");
    if(kprof){
        float ms[5],mq=0,mu=0,ma=0;
        for(int i=0;i<5;i++)cudaEventElapsedTime(&ms[i],ev[i],ev[i+1]);
        cudaEventElapsedTime(&mq,ex[0],ex[1]);cudaEventElapsedTime(&mu,ex[1],ex[2]);cudaEventElapsedTime(&ma,ex[2],ev[0]);
        fprintf(stderr,"wokprof layer=%d q=%.3f up=%.3f attn=%.3f rope=%.3f wo_a=%.3f round=%.3f wo_b=%.3f down=%.3f ms rows=%d\n",layer,mq,mu,ma,ms[0],ms[1],ms[2],ms[3],ms[4],total);
        for(int i=0;i<7;i++)cudaEventDestroy(ev[i]);for(int i=0;i<3;i++)cudaEventDestroy(ex[i]);
    }
    return done;
}

__global__ void sparse_attn_batch_kernel(float*out,const float*q,const float*vals,const float*sink,
        const int*meta,int comp_base,int heads,int dim,float scale){
    int t=blockIdx.y,h=blockIdx.x;
    int wo=meta[3*t],wn=meta[3*t+1],cn=meta[3*t+2],total=wn+cn;
    const float*qh=q+((long long)t*heads+h)*dim;
    extern __shared__ float score[];
    __shared__ float den;
    int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=vals+(long long)(item<wn?wo+item:comp_base+item-wn)*dim;
        float z=0;for(int d=lane;d<dim;d+=32)z+=qh[d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);
        if(!lane)score[item]=z*scale;
    }
    __syncthreads();
    if(!threadIdx.x){float m=-INFINITY;for(int i=0;i<total;i++)m=fmaxf(m,score[i]);
        float d0=expf(sink[h]-m);
        for(int i=0;i<total;i++){float a=expf(score[i]-m);d0+=a;score[i]=__bfloat162float(__float2bfloat16(a));}
        den=d0;}
    __syncthreads();
    for(int d=threadIdx.x;d<dim;d+=blockDim.x){
        float sum=0;
        for(int i=0;i<wn;i++)sum+=score[i]*vals[(long long)(wo+i)*dim+d];
        for(int i=0;i<cn;i++)sum+=score[wn+i]*vals[(long long)(comp_base+i)*dim+d];
        out[((long long)t*heads+h)*dim+d]=__bfloat162float(__float2bfloat16(sum/den));
    }
}
extern "C" int dsv4_cuda_sparse_attn_batch(int device,const float*q,const float*vals,const float*sinks,
        const int*meta,int value_rows,int comp_base,int heads,int dim,int tokens,float scale,float*out){
    Dev*c=ctx(device);
    if(!c||!q||!vals||!sinks||!meta||!out||value_rows<1||comp_base<0||comp_base>value_rows||
       heads<1||dim<1||tokens<1||!(scale>0.f))return 0;
    int max_total=0;
    for(int t=0;t<tokens;t++){
        int wn=meta[3*t+1],cn=meta[3*t+2],tt=wn+cn;
        if(meta[3*t]<0||wn<1||cn<0||meta[3*t]+wn>comp_base||comp_base+cn>value_rows)return 0;
        if(tt>max_total)max_total=tt;
    }
    size_t qb=(size_t)tokens*heads*dim*4,vb=(size_t)value_rows*dim*4,sb=(size_t)heads*4,
           mb=(size_t)tokens*3*sizeof(int),sm=(size_t)max_total*4;
    if(sm>96*1024)return 0;
    if(!ok(cudaSetDevice(device),"select batched sparse attention device")||
       !buf((void**)&c->p2,&c->p2cap,qb)||!buf((void**)&c->p1,&c->p1cap,vb)||
       !buf((void**)&c->p4,&c->p4cap,qb)||!buf((void**)&c->aux1,&c->aux1cap,sb)||
       !buf((void**)&c->aux2,&c->aux2cap,mb))return 0;
    if(!ok(cudaMemcpyAsync(c->p2,q,qb,cudaMemcpyHostToDevice,c->stream),"batched attention query upload")||
       !ok(cudaMemcpyAsync(c->p1,vals,vb,cudaMemcpyHostToDevice,c->stream),"batched attention KV upload")||
       !ok(cudaMemcpyAsync(c->aux1,sinks,sb,cudaMemcpyHostToDevice,c->stream),"batched attention sink upload")||
       !ok(cudaMemcpyAsync(c->aux2,meta,mb,cudaMemcpyHostToDevice,c->stream),"batched attention meta upload"))return 0;
    if(sm>48*1024&&!ok(cudaFuncSetAttribute(sparse_attn_batch_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sm),"batched sparse attention shared memory"))return 0;
    sparse_attn_batch_kernel<<<dim3(heads,tokens),256,sm,c->stream>>>(c->p4,c->p2,c->p1,c->aux1,(int*)c->aux2,comp_base,heads,dim,scale);
    return ok(cudaGetLastError(),"batched sparse attention launch")&&
           ok(cudaMemcpyAsync(out,c->p4,qb,cudaMemcpyDeviceToHost,c->stream),"batched attention context download")&&
           ok(cudaStreamSynchronize(c->stream),"batched sparse attention sync");
}
/* Prefill batch GEMM over a resident bf16 matrix (compressor / indexer
 * projections).  Same tiling as mm_batch; numerics match the CPU reference:
 * bf16-decoded weights, fp32 accumulation, no activation quantization. */
__global__ void mm_bf16_batch(const __nv_bfloat16*w,const float*x,float*y,int O,int I,int T){
    int o=blockIdx.x;if(o>=O)return;int tbase=blockIdx.y*16,tcount=T-tbase<16?T-tbase:16;
    float sum[16]={};
    for(int i=threadIdx.x;i<I;i+=blockDim.x){float v=__bfloat162float(w[(long long)o*I+i]);
        #pragma unroll
        for(int t=0;t<16;t++)if(t<tcount)sum[t]+=x[(long long)(tbase+t)*I+i]*v;}
    __shared__ float s[16][256];
    #pragma unroll
    for(int t=0;t<16;t++)s[t][threadIdx.x]=sum[t];
    __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){
        #pragma unroll
        for(int t=0;t<16;t++)s[t][threadIdx.x]+=s[t][threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x)for(int t=0;t<tcount;t++)y[(long long)(tbase+t)*O+o]=s[t][0];
}
extern "C" int dsv4_cuda_matmul_bf16_batch(Dsv4CudaTensor*t,const float*x,int tokens,float*y){
    Dev*c=t?ctx(t->device):nullptr;
    if(!c||!x||!y||tokens<1||t->fmt!=16||!ok(cudaSetDevice(t->device),"select bf16 batch device"))return 0;
    size_t xb=(size_t)tokens*t->I*4,yb=(size_t)tokens*t->O*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"bf16 batch upload"))return 0;
    if(tokens==1)launch_bf16_one((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,c->stream);
    else mm_bf16_batch<<<dim3(t->O,(tokens+15)/16),256,0,c->stream>>>((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,tokens);
    return ok(cudaGetLastError(),"bf16 batch launch")&&
           ok(cudaMemcpyAsync(y,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"bf16 batch download")&&
           ok(cudaStreamSynchronize(c->stream),"bf16 batch sync");
}
/* Several bf16 matmuls over the same x in one round trip: one upload, one
 * launch each, one download each, one sync. Same kernel as the single call. */
extern "C" int dsv4_cuda_matmul_bf16_multi(Dsv4CudaTensor*const*ts,int n,const float*x,int tokens,float*const*ys){
    if(!ts||!ys||n<1||n>8||!x||tokens<1)return 0;
    Dev*c=ctx(ts[0]->device);
    if(!c||!ok(cudaSetDevice(ts[0]->device),"select bf16 multi device"))return 0;
    size_t xb=(size_t)tokens*ts[0]->I*4,yb=0;
    for(int i=0;i<n;i++){
        if(!ts[i]||!ys[i]||ts[i]->fmt!=16||ts[i]->device!=ts[0]->device||ts[i]->I!=ts[0]->I)return 0;
        yb+=(size_t)tokens*ts[i]->O*4;
    }
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"bf16 multi upload"))return 0;
    size_t off=0;
    for(int i=0;i<n;i++){
        size_t ob=(size_t)tokens*ts[i]->O*4;
        float*dy=(float*)((char*)c->dy+off);
        if(tokens==1)launch_bf16_one((__nv_bfloat16*)ts[i]->w,c->dx,dy,ts[i]->O,ts[i]->I,c->stream);
        else mm_bf16_batch<<<dim3(ts[i]->O,(tokens+15)/16),256,0,c->stream>>>((__nv_bfloat16*)ts[i]->w,c->dx,dy,ts[i]->O,ts[i]->I,tokens);
        if(!ok(cudaGetLastError(),"bf16 multi launch")||
           !ok(cudaMemcpyAsync(ys[i],dy,ob,cudaMemcpyDeviceToHost,c->stream),"bf16 multi download"))return 0;
        off+=ob;
    }
    return ok(cudaStreamSynchronize(c->stream),"bf16 multi sync");
}
__device__ __forceinline__ int decode_pos(const int *state){return state[1];}
__device__ __forceinline__ int compression_ready(const int *state,int ratio){return ratio&&decode_pos(state)%ratio==ratio-1;}
__global__ void rope_heads_dynamic(float*x,int heads,int dim,int rope,const int*state,int delta,int inverse,int predicate,const float*cs,const float*sn){if(predicate&&!compression_ready(state,predicate))return;int pos=decode_pos(state)+delta,p=blockIdx.x*blockDim.x+threadIdx.x,pairs=rope/2;if(p<heads*pairs){int h=p/pairs,k=p%pairs,off=h*dim+dim-rope+2*k;float s=inverse?-sn[(long long)pos*pairs+k]:sn[(long long)pos*pairs+k],u=x[off],v=x[off+1];x[off]=__bfloat162float(__float2bfloat16(u*cs[(long long)pos*pairs+k]-v*s));x[off+1]=__bfloat162float(__float2bfloat16(u*s+v*cs[(long long)pos*pairs+k]));}}
__global__ void cache_store_dynamic(float*cache,const float*kv,const int*state,int window,int dim){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim)cache[(long long)(decode_pos(state)%window)*dim+d]=kv[d];}
/* Unconditional, not FLASHINFER-gated: the DeepGEMM batched wo path
 * (dsv4_cuda_wo) launches these too, and a DEEPGEMM=1 FLASHINFER=0 build
 * otherwise fails to compile. */
__global__ void gather_group_rows(float*out,const float*in,int tokens,int groups,int width,int group){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*width;if(p<n){int row=p/width,col=p%width;out[p]=in[((long long)row*groups+group)*width+col];}}
__global__ void scatter_group_rows(float*out,const float*in,int tokens,int groups,int width,int group){long long p=(long long)blockIdx.x*blockDim.x+threadIdx.x,n=(long long)tokens*width;if(p<n){int row=p/width,col=p%width;out[((long long)row*groups+group)*width+col]=in[p];}}
__global__ void compress_store_dynamic(float*values,float*scores,const float*v,const float*g,const float*ape,const int*state,int ratio,int overlap,int O){int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<O){int phase=decode_pos(state)%ratio,slot=(overlap?ratio:0)+phase;values[(long long)slot*O+d]=v[d];scores[(long long)slot*O+d]=g[d]+ape[(long long)phase*O+d];}}
__global__ void compress_pool_dynamic(float*out,const float*values,const float*scores,const int*state,int ratio,int overlap,int dim){if(!compression_ready(state,ratio))return;int d=blockIdx.x*blockDim.x+threadIdx.x;if(d<dim){int O=(1+overlap)*dim,items=overlap?2*ratio:ratio;float mx=-INFINITY;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d;mx=fmaxf(mx,scores[(long long)j*O+col]);}float den=0,sum=0;for(int j=0;j<items;j++){int col=overlap&&j>=ratio?dim+d:d,aoff=j*O+col;float a=expf(scores[aoff]-mx);den+=a;sum+=a*values[aoff];}out[d]=sum/den;}}
__global__ void rmsnorm_dynamic(float*out,const float*x,const float*w,int n,float eps,const int*state,int ratio){if(!compression_ready(state,ratio))return;__shared__ float sum[256],inv;float ss=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)ss+=x[i]*x[i];sum[threadIdx.x]=ss;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k)sum[threadIdx.x]+=sum[threadIdx.x+k];__syncthreads();}if(!threadIdx.x)inv=rsqrtf(sum[0]/n+eps);__syncthreads();for(int i=threadIdx.x;i<n;i+=blockDim.x)out[i]=__bfloat162float(__float2bfloat16(x[i]*inv*w[i]));}
__global__ void fp8_sim64_dynamic(float*x,int n,const int*state,int ratio){if(!compression_ready(state,ratio))return;int b=blockIdx.x,d=b*64+threadIdx.x;if(d>=n)return;float mx=0;for(int k=0;k<64&&b*64+k<n;k++)mx=fmaxf(mx,fabsf(x[b*64+k]));float sc=fmaxf(mx,1e-4f)/448.f;x[d]=e4m3(__nv_cvt_float_to_fp8(x[d]/sc,__NV_SATFINITE,__NV_E4M3))*sc;}
__global__ void compressed_store_dynamic(float*cache,const float*kv,int*count,const int*state,int ratio,int max_count,int dim){if(!compression_ready(state,ratio))return;int d=blockIdx.x*blockDim.x+threadIdx.x,slot=*count;if(slot<max_count&&d<dim)cache[(long long)slot*dim+d]=kv[d];}
__global__ void compression_commit(int*count,float*values,float*scores,const int*state,int ratio,int overlap,int O,int max_count){if(!compression_ready(state,ratio))return;int d=blockIdx.x*blockDim.x+threadIdx.x;if(overlap&&d<ratio*O){values[d]=values[(long long)ratio*O+d];scores[d]=scores[(long long)ratio*O+d];}if(!blockIdx.x&&!threadIdx.x&&*count<max_count)(*count)++;}
__global__ void attention_window_dynamic(float*context,const float*q,const float*cache,const float*compressed,const float*sink,int heads,int dim,int window,const int*state,const int*cnp){
    int pos=decode_pos(state),cn=*cnp,h=blockIdx.x;if(h>=heads)return;extern __shared__ float score[];__shared__ float mx,den;
    int start=pos-window+1;if(start<0)start=0;int nw=pos-start+1,total=nw+cn,warp=threadIdx.x>>5,lane=threadIdx.x&31;
    for(int item=warp;item<total;item+=blockDim.x/32){
        const float*v=item<nw?cache+(long long)((start+item)%window)*dim:compressed+(long long)(item-nw)*dim;float z=0;
        for(int d=lane;d<dim;d+=32)z+=q[(long long)h*dim+d]*v[d];
        for(int n=16;n;n>>=1)z+=__shfl_down_sync(0xffffffff,z,n);if(!lane)score[item]=z*rsqrtf((float)dim);
    }
    __syncthreads();if(!threadIdx.x){mx=sink[h];for(int i=0;i<total;i++)mx=fmaxf(mx,score[i]);den=expf(sink[h]-mx);for(int i=0;i<total;i++){float a=expf(score[i]-mx);score[i]=a;den+=a;}}
    __syncthreads();for(int d=threadIdx.x;d<dim;d+=blockDim.x){float sum=0;for(int i=0;i<nw;i++)sum+=score[i]*cache[(long long)((start+i)%window)*dim+d];for(int i=0;i<cn;i++)sum+=score[nw+i]*compressed[(long long)i*dim+d];context[(long long)h*dim+d]=__bfloat162float(__float2bfloat16(sum/den));}
}
/* Adapted from ds4_cuda.cu (MIT, Copyright 2026 ds4.c authors). */
__device__ __forceinline__ bool route_better(float av,unsigned ai,float bv,unsigned bi){return av>bv||(av==bv&&ai<bi);}
__device__ __forceinline__ float route_prob(float x){return sqrtf(log1pf(expf(-fabsf(x)))+fmaxf(x,0.f));}
__global__ void route_top6(int *selected,float *weights,const float *bias,const float *logits,int fixed,float scale){unsigned lane=threadIdx.x;if(lane>=32)return;float prob[8],score[8];
    #pragma unroll
    for(unsigned j=0;j<8;j++){unsigned e=lane+j*32;prob[j]=route_prob(logits[e]);score[j]=prob[j]+(bias?bias[e]:0.f);}
    if(fixed){__shared__ float all[256];for(unsigned j=0;j<8;j++)all[lane+j*32]=prob[j];__syncwarp();if(!lane){float sum=0;for(int k=0;k<6;k++){weights[k]=all[selected[k]];sum+=weights[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}return;}
    float outp[6]={};unsigned outi[6]={};
    #pragma unroll
    for(unsigned k=0;k<6;k++){float bs=-INFINITY,bp=0;unsigned bi=UINT_MAX;
        #pragma unroll
        for(unsigned j=0;j<8;j++){unsigned e=lane+j*32;if(route_better(score[j],e,bs,bi)){bs=score[j];bp=prob[j];bi=e;}}
        #pragma unroll
        for(unsigned mask=16;mask;mask>>=1){float os=__shfl_xor_sync(0xffffffff,bs,mask),op=__shfl_xor_sync(0xffffffff,bp,mask);unsigned oi=__shfl_xor_sync(0xffffffff,bi,mask);if(route_better(os,oi,bs,bi)){bs=os;bp=op;bi=oi;}}
        #pragma unroll
        for(unsigned j=0;j<8;j++)if(lane+j*32==bi)score[j]=-INFINITY;if(!lane){outp[k]=bp;outi[k]=bi;}}
    if(!lane){float sum=0;for(int k=0;k<6;k++){selected[k]=(int)outi[k];weights[k]=outp[k];sum+=outp[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}}
__global__ void route_top6_parallel(int *selected,float *weights,const float *bias,const float *logits,int fixed,float scale){unsigned e=threadIdx.x;__shared__ float prob[256];if(e<256)prob[e]=route_prob(logits[e]);__syncthreads();if(e)return;if(!fixed){for(int k=0;k<6;k++)selected[k]=-1;for(int i=0;i<256;i++){float s=prob[i]+(bias?bias[i]:0.f);for(int k=0;k<6;k++)if(selected[k]<0||route_better(s,i,prob[selected[k]]+(bias?bias[selected[k]]:0.f),(unsigned)selected[k])){for(int j=5;j>k;j--)selected[j]=selected[j-1];selected[k]=i;break;}}}float sum=0;for(int k=0;k<6;k++){weights[k]=prob[selected[k]];sum+=weights[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}
__global__ void route_top6_serial(int *selected,float *weights,const float *bias,const float *logits,int fixed,float scale){if(threadIdx.x||blockIdx.x)return;if(!fixed){for(int k=0;k<6;k++){int best=-1;float bv=-INFINITY;for(int i=0;i<256;i++){bool used=false;for(int j=0;j<k;j++)used|=selected[j]==i;if(used)continue;float s=route_prob(logits[i])+(bias?bias[i]:0.f);if(s>bv){bv=s;best=i;}}selected[k]=best;}}float sum=0;for(int k=0;k<6;k++){weights[k]=route_prob(logits[selected[k]]);sum+=weights[k];}sum=fmaxf(sum,6.103515625e-5f);for(int k=0;k<6;k++)weights[k]=weights[k]/sum*scale;}
__global__ void route_fixed6(int *selected,float *weights,const float *logits,float scale){int k=threadIdx.x;__shared__ float prob[6];if(k<6)prob[k]=route_prob(logits[selected[k]]);__syncthreads();if(!k){float sum=0;for(int i=0;i<6;i++)sum+=prob[i];sum=fmaxf(sum,6.103515625e-5f);for(int i=0;i<6;i++)weights[i]=prob[i]/sum*scale;}}
__global__ void build_moe_desc(MvDesc *desc,const ExpertPtr *table,const int *ids,const float *x,
                               float *gate,float *up,float *act,float *down,int count,int experts,int H,int I){
    int k=threadIdx.x;if(k>=count)return;int e=ids[k];if(e<0||e>=experts)return;
    ExpertPtr g=table[e],u=table[experts+e],d=table[2*experts+e];
    desc[2*k]={g.w,g.scale,x,gate+(long long)k*I};desc[2*k+1]={u.w,u.scale,x,up+(long long)k*I};
    desc[2*count+k]={d.w,d.scale,act+(long long)k*I,down+(long long)k*H};
}
__global__ void load_hash_ids(int *ids,const int64_t *map,const int *state,int topk){int k=threadIdx.x;if(k<topk)ids[k]=(int)map[(long long)state[0]*topk+k];}
__global__ void mhc_mix_scale(float *mix,const float *residual,const float *scale,const float *base,int M,int H,float eps){
    __shared__ float sum[256],inv;int MH=M*H,N=2*M+M*M;float ss=0;for(int i=threadIdx.x;i<MH;i+=blockDim.x)ss+=residual[i]*residual[i];
    sum[threadIdx.x]=ss;__syncthreads();for(int k=128;k;k>>=1){if(threadIdx.x<k)sum[threadIdx.x]+=sum[threadIdx.x+k];__syncthreads();}
    if(!threadIdx.x)inv=rsqrtf(sum[0]/MH+eps);__syncthreads();for(int n=threadIdx.x;n<N;n+=blockDim.x){int group=n<M?0:n<2*M?1:2;mix[n]=mix[n]*inv*scale[group]+base[n];}
}
__global__ void mhc_coeff(float *state,const float *mix,int M,float pre_eps,float sink_eps,float post_mult,int sink_iters){
    if(blockIdx.x)return;int t=threadIdx.x;float *post=state,*comb=state+M,*pre=comb+M*M;
    if(M==4){int lane=t&31;if(lane<4){pre[lane]=1.f/(1.f+expf(-mix[lane]))+pre_eps;post[lane]=(1.f/(1.f+expf(-mix[4+lane])))*post_mult;}
        float v=lane<16?mix[8+lane]:-INFINITY,mx=v;mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,1));mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,2));
        v=lane<16?expf(v-mx):0.f;float sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v=v/sum+sink_eps;
        for(int it=0;it<sink_iters;it++){if(it){sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,1);sum+=__shfl_xor_sync(0xffffffff,sum,2);if(lane<16)v/=sum+sink_eps;}
            sum=v;sum+=__shfl_xor_sync(0xffffffff,sum,4);sum+=__shfl_xor_sync(0xffffffff,sum,8);if(lane<16)v/=sum+sink_eps;}
        if(lane<16)comb[lane]=v;return;}
    if(t<M){pre[t]=1.f/(1.f+expf(-mix[t]))+pre_eps;post[t]=(1.f/(1.f+expf(-mix[M+t])))*post_mult;
        float mx=-INFINITY,sum=0;for(int j=0;j<M;j++)mx=fmaxf(mx,mix[2*M+t*M+j]);
        for(int j=0;j<M;j++){float v=expf(mix[2*M+t*M+j]-mx);comb[t*M+j]=v;sum+=v;}
        for(int j=0;j<M;j++)comb[t*M+j]=comb[t*M+j]/sum+sink_eps;}
    __syncthreads();for(int it=0;it<sink_iters;it++){
        if(it&&t<M){float sum=0;for(int j=0;j<M;j++)sum+=comb[t*M+j];for(int j=0;j<M;j++)comb[t*M+j]/=sum+sink_eps;}
        __syncthreads();if(t<M){float sum=0;for(int i=0;i<M;i++)sum+=comb[i*M+t];for(int i=0;i<M;i++)comb[i*M+t]/=sum+sink_eps;}
        __syncthreads();
    }
}
__global__ void mhc_input(float *out,const float *residual,const float *state,int M,int H){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<H){
    const float *pre=state+M+M*M;float v=0;for(int i=0;i<M;i++)v+=pre[i]*residual[(long long)i*H+h];out[h]=__bfloat162float(__float2bfloat16(v));}}
__global__ void mhc_input_norm(float*out,const float*residual,const float*state,const float*w,int M,int H,float eps){__shared__ float ws[8],inv;const float*pre=state+M+M*M;float ss=0;int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    for(int h=threadIdx.x;h<H;h+=blockDim.x){float v=0;for(int i=0;i<M;i++)v+=pre[i]*residual[(long long)i*H+h];v=__bfloat162float(__float2bfloat16(v));out[h]=v;ss+=v*v;}ss=warp_sum_f32(ss);if(!lane)ws[warp]=ss;__syncthreads();if(!warp){float v=lane<8?ws[lane]:0.f;v=warp_sum_f32(v);if(!lane)inv=rsqrtf(v/H+eps);}__syncthreads();for(int h=threadIdx.x;h<H;h+=blockDim.x)out[h]=__bfloat162float(__float2bfloat16(out[h]*inv*w[h]));}
__global__ void mhc_output(float *out,const float *x,const float *residual,const float *state,int M,int H){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<M*H){
    int j=p/H,h=p%H;const float *post=state,*comb=state+M;float v=post[j]*x[h];for(int i=0;i<M;i++)v+=comb[i*M+j]*residual[(long long)i*H+h];out[p]=__bfloat162float(__float2bfloat16(v));}}
__global__ void final_mhc_coeff(float*pre,const float*res,const float*fn,const float*scale,const float*base,int M,int H,float eps,float pre_eps){__shared__ float sum[5][256];int MH=M*H;float v[5]={};
    for(int i=threadIdx.x;i<MH;i+=blockDim.x){float x=res[i];v[0]+=x*x;for(int m=0;m<M;m++)v[m+1]+=x*fn[(long long)m*MH+i];}for(int m=0;m<=M;m++)sum[m][threadIdx.x]=v[m];__syncthreads();
    for(int k=128;k;k>>=1){if(threadIdx.x<k)for(int m=0;m<=M;m++)sum[m][threadIdx.x]+=sum[m][threadIdx.x+k];__syncthreads();}if(threadIdx.x<M){float inv=rsqrtf(sum[0][0]/MH+eps);pre[threadIdx.x]=1.f/(1.f+expf(-(sum[threadIdx.x+1][0]*inv*scale[0]+base[threadIdx.x])))+pre_eps;}}
__global__ void final_mhc_mix(float*out,const float*res,const float*pre,int M,int H){int h=blockIdx.x*blockDim.x+threadIdx.x;if(h<H){float v=0;for(int m=0;m<M;m++)v+=pre[m]*res[(long long)m*H+h];out[h]=__bfloat162float(__float2bfloat16(v));}}

/* the cuBLASLt tensor-core path's handle and workspace (nothing without COLI_DSV4_TC) */
#if COLI_DSV4_TC
static int tc_dev_init(Dev*c){return bok(cublasLtCreate(&c->lt),"cuBLASLt create")&&ok(cudaMalloc(&c->tc_workspace,32<<20),"Tensor Core workspace");}
#else
static int tc_dev_init(Dev*){return 1;}
#endif
/* DSV4_CUDA_SCHED=spin|yield|blocking|auto: how a host thread waits in a stream
 * sync. Decode syncs the device several times a layer (hundreds a token); the
 * runtime's default (auto) spins while the box has more cores than contexts,
 * which this one has, so the flag is here for the measurement, not a tuning. */
static unsigned sched_flags(void){const char*e=getenv("DSV4_CUDA_SCHED");if(!e||!*e)return 0;
    if(!strcmp(e,"spin"))return cudaDeviceScheduleSpin;if(!strcmp(e,"yield"))return cudaDeviceScheduleYield;
    if(!strcmp(e,"blocking"))return cudaDeviceScheduleBlockingSync;return 0;}
extern "C" int dsv4_cuda_init(const int *dev,int n){if(n<1||n>16)return 0;float scale[256];for(int b=0;b<256;b++)scale[b]=b?ldexpf(1.f,b-127):ldexpf(1.f,-127);ng=0;for(int i=0;i<n;i++){cudaDeviceProp p{};if(!ok(cudaSetDevice(dev[i]),"select device"))return 0;{unsigned f=sched_flags();if(f){cudaError_t e=cudaSetDeviceFlags(f);if(e!=cudaSuccess){cudaGetLastError();fprintf(stderr,"[DSV4 CUDA] device %d: schedule flag refused (%s)\n",dev[i],cudaGetErrorString(e));}}}if(!ok(cudaMemcpyToSymbol(e8_table,scale,sizeof(scale)),"scale table upload")||!ok(cudaGetDeviceProperties(&p,dev[i]),"device properties")||!ok(cudaStreamCreateWithFlags(&g[i].stream,cudaStreamNonBlocking),"stream")||!ok(cudaStreamCreateWithFlags(&g[i].aux,cudaStreamNonBlocking),"aux stream")||!ok(cudaEventCreateWithFlags(&g[i].ready,cudaEventDisableTiming),"device event")||!ok(cudaEventCreateWithFlags(&g[i].fork,cudaEventDisableTiming),"fork event")||!ok(cudaEventCreateWithFlags(&g[i].join,cudaEventDisableTiming),"join event")||!tc_dev_init(&g[i])||!ok(cudaMalloc(&g[i].decode_state,2*sizeof(int)),"decode state allocation"))return 0;g[i].id=dev[i];g[i].sms=p.multiProcessorCount;ng++;fprintf(stderr,"[DSV4 CUDA] device %d: %s %.1f GB sm_%d%d\n",dev[i],p.name,p.totalGlobalMem/1e9,p.major,p.minor);}for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(i!=j){int can=0;if(!ok(cudaDeviceCanAccessPeer(&can,dev[i],dev[j]),"peer query"))return 0;if(can){if(!ok(cudaSetDevice(dev[i]),"select peer device"))return 0;cudaError_t e=cudaDeviceEnablePeerAccess(dev[j],0);if(e!=cudaSuccess&&e!=cudaErrorPeerAccessAlreadyEnabled)return ok(e,"enable peer access");}}
#ifdef COLI_DSV4_NCCL
    if(n==6)for(int pair=0;pair<3;pair++){int ids[2]={dev[2*pair],dev[2*pair+1]};ncclComm_t comm[2];if(!nok(ncclCommInitAll(comm,2,ids),"EP2 communicator init"))return 0;g[2*pair].ep_comm=comm[0];g[2*pair+1].ep_comm=comm[1];}
#endif
    return 1;}
#if COLI_DSV4_TC
static void plan_free(TcPlan*p){if(p->d)cublasLtMatrixLayoutDestroy(p->d);if(p->c)cublasLtMatrixLayoutDestroy(p->c);if(p->b)cublasLtMatrixLayoutDestroy(p->b);if(p->a)cublasLtMatrixLayoutDestroy(p->a);if(p->op)cublasLtMatmulDescDestroy(p->op);memset(p,0,sizeof(*p));}
static void tc_dev_free(Dev*c){for(int p=0;p<2;p++)plan_free(&c->plans[p]);if(c->lt)cublasLtDestroy(c->lt);}
#else
static void tc_dev_free(Dev*){}
#endif
extern "C" void dsv4_cuda_shutdown(void){
    for(int i=0;i<ng;i++){cudaSetDevice(g[i].id);if(g[i].decode_state)cudaFree(g[i].decode_state);
#ifdef COLI_DSV4_NCCL
        if(g[i].ep_comm)ncclCommDestroy(g[i].ep_comm);
#endif
    }
    for(int i=0;i<ng;i++){cudaSetDevice(g[i].id);if(g[i].dx)cudaFree(g[i].dx);if(g[i].dy)cudaFree(g[i].dy);if(g[i].p1)cudaFree(g[i].p1);if(g[i].p2)cudaFree(g[i].p2);if(g[i].p3)cudaFree(g[i].p3);if(g[i].p4)cudaFree(g[i].p4);if(g[i].aux1)cudaFree(g[i].aux1);if(g[i].aux2)cudaFree(g[i].aux2);if(g[i].aux3)cudaFree(g[i].aux3);if(g[i].mvdesc)cudaFree(g[i].mvdesc);if(g[i].expert_weights)cudaFree(g[i].expert_weights);if(g[i].expert_ids)cudaFree(g[i].expert_ids);if(g[i].tcw)cudaFree(g[i].tcw);if(g[i].tcs)cudaFree(g[i].tcs);if(g[i].tcx)cudaFree(g[i].tcx);if(g[i].tcxs)cudaFree(g[i].tcxs);if(g[i].tc_workspace)cudaFree(g[i].tc_workspace);
        if(g[i].emaj)cudaFree(g[i].emaj);if(g[i].emv1)cudaFree(g[i].emv1);if(g[i].emv2)cudaFree(g[i].emv2);if(g[i].emv4)cudaFree(g[i].emv4);if(g[i].emcount)cudaFree(g[i].emcount);if(g[i].mmv)cudaFree(g[i].mmv);if(g[i].mmvcount)cudaFree(g[i].mmvcount);
#ifdef COLI_DSV4_DEEPGEMM
        if(g[i].dga1)cudaFree(g[i].dga1);if(g[i].dga2)cudaFree(g[i].dga2);if(g[i].dgsfa1)cudaFree(g[i].dgsfa1);if(g[i].dgsfa2)cudaFree(g[i].dgsfa2);if(g[i].dglayout)cudaFree(g[i].dglayout);if(g[i].dgrowmap)cudaFree(g[i].dgrowmap);if(g[i].dgmm1)cudaFree(g[i].dgmm1);if(g[i].dgmm2)cudaFree(g[i].dgmm2);if(g[i].dgref)cudaFree(g[i].dgref);if(g[i].dgdensews)cudaFree(g[i].dgdensews);
#endif
        tc_dev_free(&g[i]);if(g[i].hx)cudaFreeHost(g[i].hx);if(g[i].hy)cudaFreeHost(g[i].hy);if(g[i].ready)cudaEventDestroy(g[i].ready);if(g[i].fork)cudaEventDestroy(g[i].fork);if(g[i].join)cudaEventDestroy(g[i].join);cudaStreamDestroy(g[i].aux);cudaStreamDestroy(g[i].stream);
        /* zero the slot: every buffer above is grown lazily through buf(), which
         * would cudaFree a stale pointer (and trust its cap) after a re-init */
        memset(&g[i],0,sizeof g[i]);}ng=0;}
extern "C" Dsv4CudaActivation *dsv4_cuda_activation_create(int device,long long elements){Dev*c=ctx(device);if(!c||elements<1||!ok(cudaSetDevice(device),"select activation device"))return nullptr;Dsv4CudaActivation*a=(Dsv4CudaActivation*)calloc(1,sizeof(*a));if(!a)return nullptr;a->device=device;a->elements=elements;if(!alloc_ok(cudaMalloc(&a->data,(size_t)elements*sizeof(float)),"activation allocation")){free(a);return nullptr;}return a;}
extern "C" void dsv4_cuda_activation_free(Dsv4CudaActivation*a){if(!a)return;cudaSetDevice(a->device);cudaFree(a->data);free(a);}
extern "C" int dsv4_cuda_activation_upload(Dsv4CudaActivation*a,const float*x,long long n){Dev*c=a?ctx(a->device):nullptr;if(!c||!x||n<0||n>a->elements||!ok(cudaSetDevice(a->device),"select activation upload device"))return 0;return ok(cudaMemcpyAsync(a->data,x,(size_t)n*sizeof(float),cudaMemcpyHostToDevice,c->stream),"activation upload");}
extern "C" int dsv4_cuda_activation_download(float*x,const Dsv4CudaActivation*a,long long n){Dev*c=a?ctx(a->device):nullptr;if(!c||!x||n<0||n>a->elements||!ok(cudaSetDevice(a->device),"select activation download device"))return 0;return ok(cudaMemcpyAsync(x,a->data,(size_t)n*sizeof(float),cudaMemcpyDeviceToHost,c->stream),"activation download");}
extern "C" int dsv4_cuda_activation_copy(Dsv4CudaActivation*dst,const Dsv4CudaActivation*src,long long n){Dev*c=dst?ctx(dst->device):nullptr,*s=src?ctx(src->device):nullptr;if(!c||!s||n<0||n>dst->elements||n>src->elements)return 0;
    if(dst->device==src->device)return ok(cudaSetDevice(dst->device),"select local activation copy")&&ok(cudaMemcpyAsync(dst->data,src->data,(size_t)n*sizeof(float),cudaMemcpyDeviceToDevice,c->stream),"local activation copy");
    if(!ok(cudaSetDevice(src->device),"select activation source")||!ok(cudaEventRecord(s->ready,s->stream),"record activation readiness")||!ok(cudaSetDevice(dst->device),"select activation copy device")||!ok(cudaStreamWaitEvent(c->stream,s->ready,0),"wait for activation"))return 0;
    return ok(cudaMemcpyPeerAsync(dst->data,dst->device,src->data,src->device,(size_t)n*sizeof(float),c->stream),"activation device copy");}
extern "C" int dsv4_cuda_activation_copy_range(Dsv4CudaActivation*dst,long long doff,const Dsv4CudaActivation*src,long long soff,long long n){Dev*c=dst?ctx(dst->device):nullptr,*s=src?ctx(src->device):nullptr;if(!c||!s||doff<0||soff<0||n<0||doff+n>dst->elements||soff+n>src->elements)return 0;
    if(dst->device==src->device)return ok(cudaSetDevice(dst->device),"select local activation range copy")&&ok(cudaMemcpyAsync(dst->data+doff,src->data+soff,(size_t)n*sizeof(float),cudaMemcpyDeviceToDevice,c->stream),"local activation range copy");
    if(!ok(cudaSetDevice(src->device),"select activation range source")||!ok(cudaEventRecord(s->ready,s->stream),"record activation range readiness")||!ok(cudaSetDevice(dst->device),"select activation range copy device")||!ok(cudaStreamWaitEvent(c->stream,s->ready,0),"wait for activation range"))return 0;
    return ok(cudaMemcpyPeerAsync(dst->data+doff,dst->device,src->data+soff,src->device,(size_t)n*sizeof(float),c->stream),"activation range device copy");}
extern "C" int dsv4_cuda_activation_sync(const Dsv4CudaActivation*a){Dev*c=a?ctx(a->device):nullptr;return c&&ok(cudaSetDevice(a->device),"select activation sync device")&&ok(cudaStreamSynchronize(c->stream),"activation sync");}
extern "C" int dsv4_cuda_activation_device(const Dsv4CudaActivation*a){return a?a->device:-1;}
extern "C" int dsv4_cuda_decode_state_set(int device,int token,int position){Dev*c=ctx(device);if(!c||token<0||position<0||!ok(cudaSetDevice(device),"select decode state device"))return 0;set_decode_state<<<1,1,0,c->stream>>>(c->decode_state,token,position);return ok(cudaGetLastError(),"decode state update");}
extern "C" void dsv4_cuda_profiler_start(void){cudaProfilerStart();}
extern "C" void dsv4_cuda_profiler_stop(void){for(int i=0;i<ng;i++){cudaSetDevice(g[i].id);cudaStreamSynchronize(g[i].stream);}cudaProfilerStop();}
extern "C" int dsv4_cuda_graph_begin(int device){Dev*c=ctx(device);return c&&ok(cudaSetDevice(device),"select graph device")&&ok(cudaStreamBeginCapture(c->stream,cudaStreamCaptureModeThreadLocal),"begin graph capture");}
extern "C" Dsv4CudaGraph *dsv4_cuda_graph_end(int device){Dev*c=ctx(device);if(!c||!ok(cudaSetDevice(device),"select graph device"))return nullptr;Dsv4CudaGraph*g=(Dsv4CudaGraph*)calloc(1,sizeof(*g));if(!g)return nullptr;g->device=device;if(!ok(cudaStreamEndCapture(c->stream,&g->graph),"end graph capture")||!ok(cudaGraphInstantiate(&g->exec,g->graph,0),"instantiate graph")){dsv4_cuda_graph_free(g);return nullptr;}return g;}
extern "C" int dsv4_cuda_graph_end_pair(int primary,int peer,Dsv4CudaGraph **pg,Dsv4CudaGraph **qg){
    Dev*p=ctx(primary),*q=ctx(peer);if(!p||!q||!pg||!qg||*pg||*qg)return 0;
    Dsv4CudaGraph*a=(Dsv4CudaGraph*)calloc(1,sizeof(*a)),*b=(Dsv4CudaGraph*)calloc(1,sizeof(*b));if(!a||!b){free(a);free(b);return 0;}
    a->device=primary;b->device=peer;
    if(!ok(cudaSetDevice(peer),"select peer graph device")||!ok(cudaStreamEndCapture(q->stream,&b->graph),"end peer graph capture")||
       !ok(cudaSetDevice(primary),"select primary graph device")||!ok(cudaStreamEndCapture(p->stream,&a->graph),"end primary graph capture")||
       !ok(cudaGraphInstantiate(&b->exec,b->graph,0),"instantiate peer graph")||!ok(cudaGraphInstantiate(&a->exec,a->graph,0),"instantiate primary graph")){
        dsv4_cuda_graph_free(a);dsv4_cuda_graph_free(b);return 0;}
    *pg=a;*qg=b;return 1;
}
extern "C" int dsv4_cuda_graph_launch(Dsv4CudaGraph*g){Dev*c=g?ctx(g->device):nullptr;return c&&ok(cudaSetDevice(g->device),"select graph device")&&ok(cudaGraphLaunch(g->exec,c->stream),"launch graph");}
extern "C" void dsv4_cuda_graph_free(Dsv4CudaGraph*g){if(!g)return;cudaSetDevice(g->device);if(g->exec)cudaGraphExecDestroy(g->exec);if(g->graph)cudaGraphDestroy(g->graph);free(g);}
/* Host->device copy ordered on the device's compute stream, then drained.
 * A plain cudaMemcpy runs on the legacy stream, which the non-blocking
 * c->stream does not wait for, and from pageable memory it may return before
 * the DMA has landed: a kernel queued on c->stream right after could read a
 * half-written decode mirror, and decode output then drifted with the mirror
 * cache's recycling timing (expert slabs past the 20 GiB page-lock cap,
 * or with DSV4_CUDA_PIN_HOST=0, are pageable). Same blocking contract for the caller as cudaMemcpy. */
static int h2d(Dev*c,void*dst,const void*src,size_t n,const char*what){
    return ok(cudaMemcpyAsync(dst,src,n,cudaMemcpyHostToDevice,c->stream),what)&&ok(cudaStreamSynchronize(c->stream),what);}
static int upload(Dsv4CudaTensor **pt,const void *w,size_t wb,const uint8_t *sc,size_t sb,int O,int I,int d,int fmt){if(*pt)return 1;Dev *c=ctx(d);if(!c||!ok(cudaSetDevice(d),"select upload device"))return 0;Dsv4CudaTensor *t=(Dsv4CudaTensor*)calloc(1,sizeof(*t));t->O=O;t->I=I;t->device=d;t->fmt=fmt;t->bytes=wb+sb;
    if(!alloc_ok(cudaMalloc(&t->w,wb),"weight allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_w=1;if(!h2d(c,t->w,w,wb,"weight upload")){dsv4_cuda_tensor_free(t);return 0;}
    if(sb){if(!alloc_ok(cudaMalloc(&t->scale,sb),"scale allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_scale=1;if(!h2d(c,t->scale,sc,sb,"scale upload")){dsv4_cuda_tensor_free(t);return 0;}}
#ifdef COLI_DSV4_DEEPGEMM
    if((fmt==8||fmt==9)&&O%128==0&&I%128==0){size_t db=(size_t)O*(I/128/4)*sizeof(int);if(!alloc_ok(cudaMalloc(&t->dg_scale,db),"dense DeepGEMM scale allocation")){dsv4_cuda_tensor_free(t);return 0;}t->own_dg_scale=1;
        if(fmt==9){int groups=O/1024;dg_pack_batched_scale<<<(O*(I/128/4)+255)/256,256,0,c->stream>>>(t->dg_scale,t->scale,O/groups,I/128,groups);}
        else dg_pack_dense_scale<<<(O*(I/128/4)+255)/256,256,0,c->stream>>>(t->dg_scale,t->scale,O,I/128);
        if(!ok(cudaGetLastError(),"dense DeepGEMM scale packing")){dsv4_cuda_tensor_free(t);return 0;}t->bytes+=db;}
#endif
    *pt=t;return 1;}
extern "C" int dsv4_cuda_upload_fp8(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){return upload(t,w,(size_t)O*I,s,(size_t)((O+127)/128)*((I+127)/128),O,I,d,8);}
extern "C" int dsv4_cuda_upload_fp8_bf16(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){return upload(t,w,(size_t)O*I,s,(size_t)((O+127)/128)*((I+127)/128),O,I,d,9);}
extern "C" int dsv4_cuda_upload_fp4(Dsv4CudaTensor **t,const uint8_t*w,const uint8_t*s,int O,int I,int d){if(!upload(t,w,(size_t)O*I/2,s,(size_t)O*I/32,O,I,d,4))return 0;
    return 1;}
/* In-place refill of an existing fp4 mirror with a same-shape expert: no
 * cudaFree/cudaMalloc churn, DMA from page-locked slabs when possible. The
 * copies are ordered on c->stream from pinned AND pageable slabs (a pageable
 * cudaMemcpyAsync stages the source before it returns, so the slab may be
 * released at once); sync=1 drains before returning. A legacy-stream
 * cudaMemcpy here was not ordered against the kernels on c->stream (see h2d). */
static bool host_pinned(const void*p,size_t len);
extern "C" int dsv4_cuda_tensor_refill_fp4(Dsv4CudaTensor *t,const uint8_t*w,const uint8_t*s,int O,int I,int sync){
    if(!t||!w||!s||t->fmt!=4||t->O!=O||t->I!=I||!t->w||!t->scale)return 0;
    Dev*c=ctx(t->device);if(!c||!ok(cudaSetDevice(t->device),"select refill device"))return 0;
    size_t wb=(size_t)O*I/2,sb=(size_t)O*I/32;
    /* sync>=2: the decode uploader threads. Their copies ride the aux stream
     * (3 = enqueue only, 2 = enqueue and drain it), so a synchronous refill
     * never waits behind the attention/dense kernels queued on the compute
     * stream (measured: that wait made a 0.7 ms pinned copy cost 2.3 ms from
     * the uploader thread). */
    cudaStream_t cs=sync>=2?c->aux:c->stream;
    if(sync==3)sync=0;
    if(host_pinned(w,wb))host_pinned(s,sb);   /* page-lock the slab when the cap allows: pure DMA then */
    if(!ok(cudaMemcpyAsync(t->w,w,wb,cudaMemcpyHostToDevice,cs),"refill weight upload")||
       !ok(cudaMemcpyAsync(t->scale,s,sb,cudaMemcpyHostToDevice,cs),"refill scale upload"))return 0;
    if(sync&&!ok(cudaStreamSynchronize(cs),"refill drain"))return 0;
    return 1;
}
/* One-block expert mirror: a single device allocation holding the three
 * tensors' scales (the record's scale run, S bytes) and, 16-byte aligned
 * after them, their weights (the weight run, Wb bytes). A refill is TWO
 * cudaMemcpyAsync of the host record's runs instead of six per-tensor
 * copies (measured on WDDM: ~110 us per enqueue, 0.66 ms of the 0.90 ms a
 * queued upload cost on its thread), and re-points the tensors to where
 * their matrices landed, so the record's tensor order (which varies by
 * shard) never matters. The gate tensor owns the block; up and down point
 * into it and own nothing, so the engine's free order (gate, up, down)
 * stays. block = the weight run, block2 = the scale run, blockpad = S. */
static void aux_flush(Dev*c);   /* below, by group_fence */
extern "C" int dsv4_cuda_mirror_alloc(Dsv4CudaTensor**g,Dsv4CudaTensor**u,Dsv4CudaTensor**d,int Og,int Ig,int Od,int Id,int dev){
    if(!g||!u||!d||Og<=0||Ig<=0||Od<=0||Id<=0||Ig%32||Id%32)return 0;Dev*c=ctx(dev);if(!c||!ok(cudaSetDevice(dev),"select mirror device"))return 0;
    long long sg=(long long)Og*Ig/32,sd=(long long)Od*Id/32,wg=(long long)Og*Ig/2,wd=(long long)Od*Id/2,S=2*sg+sd,Wb=2*wg+wd;
    /* two allocations: the weight run alone is a whole number of 2 MiB
     * pages (12 MiB for this model); scales + weights in one took a page
     * more per mirror and cost the second card 57 of its 663 mirrors */
    void*wblock=nullptr,*sblock=nullptr;if(!alloc_ok(cudaMalloc(&wblock,(size_t)Wb),"mirror weight-run allocation"))return 0;
    if(!alloc_ok(cudaMalloc(&sblock,(size_t)S),"mirror scale-run allocation")){cudaFree(wblock);return 0;}
    Dsv4CudaTensor*t[3];
    for(int k=0;k<3;k++){t[k]=(Dsv4CudaTensor*)calloc(1,sizeof(Dsv4CudaTensor));if(!t[k]){for(int j=0;j<k;j++)free(t[j]);cudaFree(wblock);cudaFree(sblock);return 0;}
        t[k]->device=dev;t[k]->fmt=4;t[k]->O=k<2?Og:Od;t[k]->I=k<2?Ig:Id;
        t[k]->scale=(uint8_t*)sblock+(k<2?k*sg:2*sg);t[k]->w=(uint8_t*)wblock+(k<2?k*wg:2*wg);}   /* provisional: a refill re-points */
    t[0]->block=wblock;t[0]->block2=sblock;t[0]->blockpad=S;t[0]->bytes=Wb;*g=t[0];*u=t[1];*d=t[2];return 1;}
extern "C" int dsv4_cuda_mirror_refill(Dsv4CudaTensor*g,Dsv4CudaTensor*u,Dsv4CudaTensor*d,const uint8_t*sc,long long S,const long long*soff,const uint8_t*w,long long Wb,const long long*woff,int sync){
    if(!g||!g->block||!g->block2||!u||!d||!sc||!w||!soff||!woff||S<=0||Wb<=0)return 0;if(S>g->blockpad||Wb>g->bytes)return 0;
    Dsv4CudaTensor*t[3]={g,u,d};
    for(int k=0;k<3;k++){long long sb=(long long)t[k]->O*t[k]->I/32,wb=(long long)t[k]->O*t[k]->I/2;
        if(soff[k]<0||soff[k]+sb>S||woff[k]<0||woff[k]+wb>Wb||(woff[k]&15))return 0;}
    Dev*c=ctx(g->device);if(!c||!ok(cudaSetDevice(g->device),"select mirror refill device"))return 0;
    cudaStream_t cs=sync>=2?c->aux:c->stream;if(sync==3)sync=0;
    host_pinned(sc,(size_t)S);host_pinned(w,(size_t)Wb);
    if(!ok(cudaMemcpyAsync(g->block2,sc,(size_t)S,cudaMemcpyHostToDevice,cs),"mirror scales upload")||
       !ok(cudaMemcpyAsync(g->block,w,(size_t)Wb,cudaMemcpyHostToDevice,cs),"mirror weights upload"))return 0;
    if(sync&&!ok(cudaStreamSynchronize(cs),"mirror block drain"))return 0;
    if(!sync&&cs==c->aux)aux_flush(c);   /* enqueue-only: submit now, the fence must see it */
    for(int k=0;k<3;k++){t[k]->scale=(uint8_t*)g->block2+soff[k];t[k]->w=(uint8_t*)g->block+woff[k];}
    return 1;}
/* The chunked read -> upload pipeline (V4_UPLOAD_PIECES): a piece of a
 * block mirror's run (0 = the scale run, 1 = the weight run) at off,
 * enqueued on the card's aux stream and never waited for here — the
 * expert group fences the stream (group_fence) as it does for the
 * uploader's copies. mirror_point re-points the three tensors once every
 * piece is in (host side only, no device work). */
static cudaStream_t pre_stream(Dev*c){
    if(!c->pre&&!ok(cudaStreamCreateWithFlags(&c->pre,cudaStreamNonBlocking),"prefetch stream"))c->pre=nullptr;
    return c->pre;}
extern "C" int dsv4_cuda_mirror_copy(Dsv4CudaTensor*g,int run,long long off,const uint8_t*src,long long len){
    int pre=run&2;run&=~2;   /* run | 2: a prefetch worker's piece, on the prefetch stream (V4_PRED_PIECES=2) */
    if(!g||!g->block||!g->block2||!src||off<0||len<=0||(run!=0&&run!=1))return 0;
    long long cap=run?g->bytes:g->blockpad;if(off+len>cap)return 0;
    Dev*c=ctx(g->device);if(!c||!ok(cudaSetDevice(g->device),"select mirror piece device"))return 0;
    cudaStream_t cs=pre?pre_stream(c):c->aux;if(!cs)return 0;
    host_pinned(src,(size_t)len);
    void*dst=(uint8_t*)(run?g->block:g->block2)+off;
    return ok(cudaMemcpyAsync(dst,src,(size_t)len,cudaMemcpyHostToDevice,cs),"mirror piece upload")?1:0;}
extern "C" int dsv4_cuda_mirror_point(Dsv4CudaTensor*g,Dsv4CudaTensor*u,Dsv4CudaTensor*d,long long S,const long long*soff,long long Wb,const long long*woff){
    int pre=S<0;if(pre)S=-S;
    if(!g||!g->block||!g->block2||!u||!d||!soff||!woff||S<=0||Wb<=0||S>g->blockpad||Wb>g->bytes)return 0;
    if(pre){Dev*c=ctx(g->device);if(!c||!ok(cudaSetDevice(g->device),"select mirror point device")||!pre_stream(c))return 0;
        if(!g->ev&&!ok(cudaEventCreateWithFlags(&g->ev,cudaEventDisableTiming),"prefetch mirror event"))return 0;
        if(!ok(cudaEventRecord(g->ev,c->pre),"prefetch mirror event record"))return 0;g->evwait=1;
        for(int k=0;k<3;k++){Dsv4CudaTensor*t=k?(k==1?u:d):g;t->scale=(uint8_t*)g->block2+soff[k];t->w=(uint8_t*)g->block+woff[k];}
        return 1;}
    Dsv4CudaTensor*t[3]={g,u,d};
    for(int k=0;k<3;k++){long long sb=(long long)t[k]->O*t[k]->I/32,wb=(long long)t[k]->O*t[k]->I/2;
        if(soff[k]<0||soff[k]+sb>S||woff[k]<0||woff[k]+wb>Wb||(woff[k]&15))return 0;}
    {Dev*c=ctx(g->device);if(c&&ok(cudaSetDevice(g->device),"select mirror point device"))aux_flush(c);}   /* the pieces this thread enqueued: submitted before the entry becomes a hit */
    for(int k=0;k<3;k++){t[k]->scale=(uint8_t*)g->block2+soff[k];t[k]->w=(uint8_t*)g->block+woff[k];}
    return 1;}
/* diagnostic: compare a run of a block mirror with host bytes (aux stream drained first); returns the first mismatching offset, -1 when equal, -2 on error */
extern "C" long long dsv4_cuda_mirror_check(Dsv4CudaTensor*g,int run,long long off,const uint8_t*src,long long len){
    if(!g||!g->block||!g->block2||!src||off<0||len<=0||(run!=0&&run!=1))return -2;
    long long cap=run?g->bytes:g->blockpad;if(off+len>cap)return -2;
    Dev*c=ctx(g->device);if(!c||!ok(cudaSetDevice(g->device),"select mirror check device"))return -2;
    if(!ok(cudaStreamSynchronize(c->aux),"mirror check drain"))return -2;
    if(c->pre&&!ok(cudaStreamSynchronize(c->pre),"mirror check prefetch drain"))return -2;
    std::vector<uint8_t> h((size_t)len);
    if(!ok(cudaMemcpy(h.data(),(uint8_t*)(run?g->block:g->block2)+off,(size_t)len,cudaMemcpyDeviceToHost),"mirror check download"))return -2;
    for(long long i=0;i<len;i++)if(h[(size_t)i]!=src[i])return i;
    return -1;}
extern "C" int dsv4_cuda_upload_bf16(Dsv4CudaTensor **t,const uint16_t*w,int O,int I,int d){return upload(t,w,(size_t)O*I*2,nullptr,0,O,I,d,16);}
extern "C" int dsv4_cuda_upload_f32(Dsv4CudaTensor **t,const float*w,int O,int I,int d){return upload(t,w,(size_t)O*I*4,nullptr,0,O,I,d,32);}
extern "C" int dsv4_cuda_mhc_pre(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,int M,int H,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    int N=2*M+M*M,MH=M*H;Dev*c=r?ctx(r->device):nullptr;if(!c||!fn||!scale||!base||!state||!input||fn->fmt!=32||scale->fmt!=32||base->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||state->device!=r->device||input->device!=r->device||r->elements<MH||fn->O!=N||fn->I!=MH||scale->O*scale->I<3||base->O*base->I<N||state->elements<M+M*M+M||input->elements<H||!ok(cudaSetDevice(r->device),"select mHC device")||!buf((void**)&c->p1,&c->p1cap,(size_t)N*4))return 0;
    if(N==24&&M==4){constexpr int parts=8;if(!buf((void**)&c->dy,&c->ycap,(size_t)(24+1)*parts*sizeof(float)))return 0;mhc_fn_split<<<dim3(parts,N),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,N,MH,parts);mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,MH,rms_eps,parts);}
    else{mv_f32<<<N,256,0,c->stream>>>((float*)fn->w,r->data,c->p1,N,MH);mhc_mix_scale<<<1,256,0,c->stream>>>(c->p1,r->data,(float*)scale->w,(float*)base->w,M,H,rms_eps);}
    mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);mhc_input<<<(H+255)/256,256,0,c->stream>>>(input->data,r->data,state->data,M,H);return ok(cudaGetLastError(),"mHC pre launch");}
extern "C" int dsv4_cuda_mhc_post(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,const Dsv4CudaActivation*state,int M,int H,Dsv4CudaActivation*out){
    Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||r->device!=x->device||state->device!=x->device||out->device!=x->device||x->elements<H||r->elements<M*H||state->elements<M+M*M+M||out->elements<M*H||!ok(cudaSetDevice(x->device),"select mHC device"))return 0;
    mhc_output<<<(M*H+255)/256,256,0,c->stream>>>(out->data,x->data,r->data,state->data,M,H);return ok(cudaGetLastError(),"mHC post launch");}
extern "C" int dsv4_cuda_attention_first(const Dsv4CudaActivation*input,Dsv4CudaTensor*an,Dsv4CudaTensor*qa,Dsv4CudaTensor*qn,Dsv4CudaTensor*qb,Dsv4CudaTensor*kv,Dsv4CudaTensor*kvn,Dsv4CudaTensor*sink,Dsv4CudaTensor*wa,Dsv4CudaTensor*wb,int heads,int dim,int qk_rope,int groups,float eps,Dsv4CudaActivation*output){
    Dev*c=input?ctx(input->device):nullptr;int H=qa?qa->I:0,R=qa?qa->O:0,Q=heads*dim,K=dim,WR=wa?wa->O:0;if(!c||!output||!an||!qa||!qn||!qb||!kv||!kvn||!sink||!wa||!wb||input->elements<H||output->elements<H||an->fmt!=32||qn->fmt!=16||kvn->fmt!=32||sink->fmt!=32||qa->fmt!=8||qb->fmt!=8||kv->fmt!=8||wa->fmt!=9||wb->fmt!=8||an->O*an->I<H||qn->O*qn->I<R||kvn->O*kvn->I<K||sink->O*sink->I<heads||qb->I!=R||qb->O!=Q||kv->I!=H||kv->O!=K||wa->I*groups!=Q||wb->I!=WR||wb->O!=H||heads%groups)return 0;
    Dsv4CudaTensor*tensors[]={an,qa,qn,qb,kv,kvn,sink,wa,wb};for(auto*t:tensors)if(t->device!=input->device)return 0;int P1=R>WR?R:WR;if(output->device!=input->device||!ok(cudaSetDevice(input->device),"select attention device")||!buf((void**)&c->dx,&c->xcap,(size_t)H*4)||!buf((void**)&c->p1,&c->p1cap,(size_t)P1*4)||!buf((void**)&c->p2,&c->p2cap,(size_t)Q*4)||!buf((void**)&c->p3,&c->p3cap,(size_t)K*4)||!buf((void**)&c->p4,&c->p4cap,(size_t)Q*4))return 0;
    int dense_dg=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG"));
    rmsnorm_f32<<<1,256,0,c->stream>>>(c->dx,input->data,(float*)an->w,H,eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qa,c->dx,c->p1)||!dg_attention_kv_reuse(c,kv,c->dx,c->p3))return 0;}else
#endif
    {fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);run_mv<8>((uint8_t*)qa->w,qa->scale,c->dx,c->p1,R,H,1,c->stream);}
    bf16_round<<<(R+255)/256,256,0,c->stream>>>(c->p1,R);rmsnorm_seq<<<1,256,0,c->stream>>>(c->p1,(__nv_bfloat16*)qn->w,R,eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qb,c->p1,c->p2))return 0;}else
#endif
    {fp8_sim<<<(R+127)/128,128,0,c->stream>>>(c->p1,1,R);run_mv<8>((uint8_t*)qb->w,qb->scale,c->p1,c->p2,Q,R,1,c->stream);}
    bf16_round<<<(Q+255)/256,256,0,c->stream>>>(c->p2,Q);head_rmsnorm<<<heads,256,0,c->stream>>>(c->p2,heads,dim,eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){}else
#endif
    run_mv<8>((uint8_t*)kv->w,kv->scale,c->dx,c->p3,K,H,1,c->stream);
    bf16_round<<<(K+255)/256,256,0,c->stream>>>(c->p3,K);rmsnorm_f32<<<1,256,0,c->stream>>>(c->p3,c->p3,(float*)kvn->w,K,eps);if(K>qk_rope)fp8_sim64<<<(K-qk_rope+63)/64,64,0,c->stream>>>(c->p3,K-qk_rope);
    attention_one<<<heads,256,0,c->stream>>>(c->p4,c->p2,c->p3,(float*)sink->w,heads,dim);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_wo_a(c,wa,c->p4,c->p1))return 0;}else
#endif
    run_mv<9>((uint8_t*)wa->w,wa->scale,c->p4,c->p1,WR,wa->I,groups,c->stream);
    bf16_round<<<(WR+255)/256,256,0,c->stream>>>(c->p1,WR);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,wb,c->p1,output->data))return 0;}else
#endif
    {fp8_sim<<<(WR+127)/128,128,0,c->stream>>>(c->p1,1,WR);run_mv<8>((uint8_t*)wb->w,wb->scale,c->p1,output->data,H,WR,1,c->stream);}
    bf16_round<<<(H+255)/256,256,0,c->stream>>>(output->data,H);return ok(cudaGetLastError(),"attention first launch");}
extern "C" Dsv4CudaKvCache *dsv4_cuda_kv_create(int device, int window, int dim, int max_tokens, int rope_pairs,
                                                const float *cosv, const float *sinv, const float *ccos,
                                                const float *csin) {
    Dev *c = ctx(device);
    if (!c || window < 1 || dim < 1 || max_tokens < 1 || rope_pairs < 0 ||
        (rope_pairs && (!cosv || !sinv || !ccos || !csin)) || !ok(cudaSetDevice(device), "select KV device"))
        return nullptr;
    Dsv4CudaKvCache *k = (Dsv4CudaKvCache *)calloc(1, sizeof(*k));
    if (!k) return nullptr;
    k->device = device;
    k->window = window;
    k->dim = dim;
    k->max_tokens = max_tokens;
    k->max_compressed = (max_tokens + 3) / 4;
    k->rope_pairs = rope_pairs;
    size_t kb = (size_t)window * dim * 4, cb = (size_t)k->max_compressed * dim * 4, pb = (size_t)2 * window * dim * 4,
           rb = (size_t)max_tokens * rope_pairs * 4;
    if (!ok(cudaMalloc(&k->kv, kb), "KV allocation") || !ok(cudaMemset(k->kv, 0, kb), "KV clear") ||
        !ok(cudaMalloc(&k->compressed, cb), "compressed KV allocation") ||
        !ok(cudaMalloc(&k->compressed_len, sizeof(int)), "compressed length allocation") ||
        !ok(cudaMemset(k->compressed_len, 0, sizeof(int)), "compressed length clear") ||
        !ok(cudaMalloc(&k->comp_value, pb), "compress value allocation") ||
        !ok(cudaMalloc(&k->comp_score, pb), "compress score allocation") ||
        !ok(cudaMemset(k->compressed, 0, cb), "compressed KV clear") ||
        (rb && (!ok(cudaMalloc(&k->rope_cos, rb), "RoPE cos allocation") ||
                !ok(cudaMalloc(&k->rope_sin, rb), "RoPE sin allocation") ||
                !ok(cudaMalloc(&k->comp_cos, rb), "compress RoPE cos allocation") ||
                !ok(cudaMalloc(&k->comp_sin, rb), "compress RoPE sin allocation") ||
                !h2d(c, k->rope_cos, cosv, rb, "RoPE cos upload") ||
                !h2d(c, k->rope_sin, sinv, rb, "RoPE sin upload") ||
                !h2d(c, k->comp_cos, ccos, rb, "compress RoPE cos upload") ||
                !h2d(c, k->comp_sin, csin, rb, "compress RoPE sin upload")))) {
        dsv4_cuda_kv_free(k);
        return nullptr;
    }
    fill_value<<<(pb / 4 + 255) / 256, 256, 0, c->stream>>>(k->comp_score, pb / 4, -INFINITY);
    if (!ok(cudaGetLastError(), "compress score clear")) {
        dsv4_cuda_kv_free(k);
        return nullptr;
    }
    return k;
}
extern "C" int dsv4_cuda_mhc_pre_norm(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,int M,int H,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,float norm_eps,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    int N=2*M+M*M,MH=M*H;Dev*c=r?ctx(r->device):nullptr;if(!c||!fn||!scale||!base||!norm||!state||!input||M!=4||fn->fmt!=32||scale->fmt!=32||base->fmt!=32||norm->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||norm->device!=r->device||state->device!=r->device||input->device!=r->device||r->elements<MH||fn->O!=N||fn->I!=MH||norm->O*norm->I<H||!ok(cudaSetDevice(r->device),"select normalized mHC device")||!buf((void**)&c->p1,&c->p1cap,(size_t)N*4)||!buf((void**)&c->dy,&c->ycap,(size_t)(N+1)*8*sizeof(float)))return 0;
    mhc_fn_split<<<dim3(8,N),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,N,MH,8);mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,MH,rms_eps,8);mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);mhc_input_norm<<<1,256,0,c->stream>>>(input->data,r->data,state->data,(float*)norm->w,M,H,norm_eps);return ok(cudaGetLastError(),"normalized mHC pre launch");
}
extern "C" int dsv4_cuda_mhc_pre_norm_batch(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,int tokens,int H,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    constexpr int M=4,N=24,parts=8;long long RH=(long long)tokens*M*H,XH=(long long)tokens*H;Dev*c=r?ctx(r->device):nullptr;
    if(!c||!fn||!scale||!base||!norm||!state||!input||tokens<1||tokens>65535||H!=4096||r->elements<RH||state->elements<(long long)tokens*20||input->elements<XH||
       fn->fmt!=32||scale->fmt!=32||base->fmt!=32||norm->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||norm->device!=r->device||state->device!=r->device||input->device!=r->device||
       fn->O!=N||fn->I!=M*H||scale->O*scale->I<3||base->O*base->I<N||norm->O*norm->I<H||!ok(cudaSetDevice(r->device),"select initial batched mHC device")||
       !buf((void**)&c->p1,&c->p1cap,(size_t)tokens*N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(size_t)tokens*(N+1)*parts*sizeof(float)))return 0;
    mhc_fn_split_batch<<<dim3(parts,N,tokens),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,tokens,N,M*H,parts);
    mhc_fn_finish_batch<<<tokens,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M*H,1e-6f,parts);
    float*post=state->data,*comb=post+(long long)tokens*M;mhc_coeff_batch<<<tokens,32,0,c->stream>>>(post,comb,c->p1);
    mhc_input_norm_batch<<<tokens,256,0,c->stream>>>(input->data,r->data,c->p1,(float*)norm->w,H);
    return ok(cudaGetLastError(),"initial batched mHC launch");
}
extern "C" int dsv4_cuda_mhc_pre_batch(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,int tokens,int H,Dsv4CudaActivation*state,Dsv4CudaActivation*input){
    constexpr int M=4,N=24,parts=8;long long RH=(long long)tokens*M*H,XH=(long long)tokens*H;Dev*c=r?ctx(r->device):nullptr;
    if(!c||!fn||!scale||!base||!state||!input||tokens<1||tokens>65535||H!=4096||r->elements<RH||state->elements<(long long)tokens*20||input->elements<XH||fn->fmt!=32||scale->fmt!=32||base->fmt!=32||fn->device!=r->device||scale->device!=r->device||base->device!=r->device||state->device!=r->device||input->device!=r->device||fn->O!=N||fn->I!=M*H||scale->O*scale->I<3||base->O*base->I<N||!ok(cudaSetDevice(r->device),"select initial batched mHC device")||!buf((void**)&c->p1,&c->p1cap,(size_t)tokens*N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(size_t)tokens*(N+1)*parts*sizeof(float)))return 0;
    mhc_fn_split_batch<<<dim3(parts,N,tokens),256,0,c->stream>>>((float*)fn->w,r->data,c->dy,tokens,N,M*H,parts);mhc_fn_finish_batch<<<tokens,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M*H,1e-6f,parts);float*post=state->data,*comb=post+(long long)tokens*M;mhc_coeff_batch<<<tokens,32,0,c->stream>>>(post,comb,c->p1);mhc_input_batch<<<tokens,256,0,c->stream>>>(input->data,r->data,c->p1,H);return ok(cudaGetLastError(),"initial batched mHC launch");
}
extern "C" int dsv4_cuda_mhc_post_pre(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,Dsv4CudaActivation*state,int M,int H,Dsv4CudaActivation*out,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,Dsv4CudaActivation*input){
    constexpr int N=24,parts=8;int I=M*H;Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||!fn||!scale||!base||!input||M!=4||r->device!=x->device||state->device!=x->device||out->device!=x->device||fn->device!=x->device||scale->device!=x->device||base->device!=x->device||input->device!=x->device||x->elements<H||r->elements<I||state->elements<M+M*M+M||out->elements<I||fn->fmt!=32||fn->O!=N||fn->I!=I||!ok(cudaSetDevice(x->device),"select fused mHC device")||!buf((void**)&c->p1,&c->p1cap,N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(N+1)*parts*sizeof(float)))return 0;
    mhc_post_fn_split<<<parts,256,0,c->stream>>>((float*)fn->w,x->data,r->data,state->data,out->data,c->dy,N,M,H,parts);
    mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,I,rms_eps,parts);
    mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);
    mhc_input<<<(H+255)/256,256,0,c->stream>>>(input->data,out->data,state->data,M,H);
    return ok(cudaGetLastError(),"fused mHC post/pre launch");
}
extern "C" int dsv4_cuda_mhc_post_pre_norm(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,Dsv4CudaActivation*state,int M,int H,Dsv4CudaActivation*out,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,float rms_eps,float pre_eps,float sink_eps,float post_mult,int sink_iters,float norm_eps,Dsv4CudaActivation*input){
    constexpr int N=24,parts=8;int I=M*H;Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||!fn||!scale||!base||!norm||!input||M!=4||r->device!=x->device||state->device!=x->device||out->device!=x->device||fn->device!=x->device||scale->device!=x->device||base->device!=x->device||norm->device!=x->device||input->device!=x->device||x->elements<H||r->elements<I||state->elements<M+M*M+M||out->elements<I||fn->fmt!=32||norm->fmt!=32||fn->O!=N||fn->I!=I||norm->O*norm->I<H||!ok(cudaSetDevice(x->device),"select fused normalized mHC device")||!buf((void**)&c->p1,&c->p1cap,N*sizeof(float))||!buf((void**)&c->dy,&c->ycap,(N+1)*parts*sizeof(float)))return 0;
    mhc_post_fn_split<<<parts,256,0,c->stream>>>((float*)fn->w,x->data,r->data,state->data,out->data,c->dy,N,M,H,parts);mhc_fn_finish<<<1,32,0,c->stream>>>(c->p1,c->dy,(float*)scale->w,(float*)base->w,N,M,I,rms_eps,parts);mhc_coeff<<<1,32,0,c->stream>>>(state->data,c->p1,M,pre_eps,sink_eps,post_mult,sink_iters);mhc_input_norm<<<1,256,0,c->stream>>>(input->data,out->data,state->data,(float*)norm->w,M,H,norm_eps);return ok(cudaGetLastError(),"fused normalized mHC post/pre launch");
}
extern "C" int dsv4_cuda_mhc_post_pre_norm_batch(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,Dsv4CudaActivation*state,int tokens,int H,Dsv4CudaActivation*out,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,Dsv4CudaActivation*input){
    (void)x;(void)r;(void)state;(void)tokens;(void)H;(void)out;(void)fn;(void)scale;(void)base;(void)norm;(void)input;return 0;
}
extern "C" int dsv4_cuda_mhc_post_batch(const Dsv4CudaActivation*x,const Dsv4CudaActivation*r,const Dsv4CudaActivation*state,int tokens,int H,Dsv4CudaActivation*out){
    long long RH=(long long)tokens*4*H,XH=(long long)tokens*H;Dev*c=x?ctx(x->device):nullptr;if(!c||!r||!state||!out||tokens<1||H!=4096||r->device!=x->device||state->device!=x->device||out->device!=x->device||x->elements<XH||r->elements<RH||state->elements<(long long)tokens*20||out->elements<RH||!ok(cudaSetDevice(x->device),"select batched mHC output device"))return 0;mhc_output_batch<<<(RH+255)/256,256,0,c->stream>>>(out->data,x->data,r->data,state->data,state->data+(long long)tokens*4,H);return ok(cudaGetLastError(),"batched mHC output launch");
}
extern "C" void dsv4_cuda_kv_free(Dsv4CudaKvCache *k) {
    if (!k) return;
    cudaSetDevice(k->device);
    if (k->kv) cudaFree(k->kv);
    if (k->compressed) cudaFree(k->compressed);
    if (k->compressed_len) cudaFree(k->compressed_len);
    if (k->comp_value) cudaFree(k->comp_value);
    if (k->comp_score) cudaFree(k->comp_score);
    if (k->rope_cos) cudaFree(k->rope_cos);
    if (k->rope_sin) cudaFree(k->rope_sin);
    if (k->comp_cos) cudaFree(k->comp_cos);
    if (k->comp_sin) cudaFree(k->comp_sin);
    free(k);
}
extern "C" int dsv4_cuda_attention_window(const Dsv4CudaActivation *input, Dsv4CudaTensor *an, Dsv4CudaTensor *qa,
                                          Dsv4CudaTensor *qn, Dsv4CudaTensor *qb, Dsv4CudaTensor *kv,
                                          Dsv4CudaTensor *kvn, Dsv4CudaTensor *sink, Dsv4CudaTensor *wa,
                                          Dsv4CudaTensor *wb, Dsv4CudaTensor *cwkv, Dsv4CudaTensor *cwg,
                                          Dsv4CudaTensor *ape, Dsv4CudaTensor *cnorm, int ratio, int heads, int dim,
                                          int qk_rope, int groups, int pos, float eps, Dsv4CudaKvCache *cache,
                                          Dsv4CudaActivation *output) {
    Dev *c = input ? ctx(input->device) : nullptr;
    int H = qa ? qa->I : 0, R = qb ? qb->I : 0, Q = heads * dim, K = dim, WR = wa ? wa->O : 0, overlap = ratio == 4,
        O = (1 + overlap) * dim;
    int fused_qkv = qa && qa->O == R + K;
    if (!c || !cache || !output || !an || !qa || !qn || !qb || !kv || !kvn || !sink || !wa || !wb || pos < 0 ||
        pos >= cache->max_tokens || cache->device != input->device || cache->window < 1 || cache->dim != dim ||
        cache->rope_pairs * 2 != qk_rope || input->elements < H || output->elements < H || an->fmt != 32 ||
        qn->fmt != 16 || kvn->fmt != 32 || sink->fmt != 32 || qa->fmt != 8 || qb->fmt != 8 || kv->fmt != 8 ||
        wa->fmt != 9 || wb->fmt != 8 || an->O * an->I < H || qn->O * qn->I < R || kvn->O * kvn->I < K ||
        sink->O * sink->I < heads || qa->O != R + (fused_qkv ? K : 0) || qb->O != Q || kv->I != H || kv->O != K || wa->I * groups != Q ||
        wb->I != WR || wb->O != H || heads % groups)
        return 0;
    if (ratio && (!cwkv || !cwg || !ape || !cnorm || cwkv->fmt != 16 || cwg->fmt != 16 || ape->fmt != 32 ||
                  cnorm->fmt != 32 || cwkv->O != O || cwkv->I != H || cwg->O != O || cwg->I != H ||
                  ape->O * ape->I != ratio * O || cnorm->O * cnorm->I < dim))
        return 0;
    Dsv4CudaTensor *tensors[] = {an, qa, qn, qb, kv, kvn, sink, wa, wb, cwkv, cwg, ape, cnorm};
    for (auto *t : tensors)
        if (t && t->device != input->device) return 0;
    int P1 = R + (fused_qkv ? K : 0); if (P1 < O) P1 = O;
    if (WR > P1) P1 = WR;
    if (output->device != input->device || !ok(cudaSetDevice(input->device), "select attention device") ||
        !buf((void **)&c->dx, &c->xcap, (size_t)H * 4) || !buf((void **)&c->p1, &c->p1cap, (size_t)P1 * 4) ||
        !buf((void **)&c->p2, &c->p2cap, (size_t)(Q > O ? Q : O) * 4) ||
        !buf((void **)&c->p3, &c->p3cap, (size_t)K * 4) || !buf((void **)&c->p4, &c->p4cap, (size_t)Q * 4) ||
        !buf((void **)&c->aux1, &c->aux1cap, (size_t)O * 4) ||
        !buf((void **)&c->aux2, &c->aux2cap, (size_t)O * 4) ||
        !buf((void **)&c->aux3, &c->aux3cap, (size_t)dim * 4))
        return 0;
    rmsnorm_f32<<<1, 256, 0, c->stream>>>(c->dx, input->data, (float *)an->w, H, eps);
    int async = ratio && getenv("DSV4_CUDA_ATTN_ASYNC") && atoi(getenv("DSV4_CUDA_ATTN_ASYNC"));
    cudaStream_t comp = async ? c->aux : c->stream;
    float *comp1 = async ? c->aux1 : c->p1, *comp2 = async ? c->aux2 : c->p2,
          *comp3 = async ? c->aux3 : c->p3;
    if (async && (!ok(cudaEventRecord(c->fork, c->stream), "attention fork") ||
                  !ok(cudaStreamWaitEvent(c->aux, c->fork, 0), "attention aux wait"))) return 0;
    if (ratio) {
        mv_bf16<<<O, 256, 0, comp>>>((__nv_bfloat16 *)cwkv->w, c->dx, comp1, O, H);
        mv_bf16<<<O, 256, 0, comp>>>((__nv_bfloat16 *)cwg->w, c->dx, comp2, O, H);
        compress_store_dynamic<<<(O + 255) / 256, 256, 0, comp>>>(cache->comp_value, cache->comp_score, comp1,
                                                                  comp2, (float *)ape->w, c->decode_state,
                                                                  ratio, overlap, O);
        compress_pool_dynamic<<<(dim + 255) / 256, 256, 0, comp>>>(comp3, cache->comp_value, cache->comp_score,
                                                                    c->decode_state, ratio, overlap, dim);
        rmsnorm_dynamic<<<1, 256, 0, comp>>>(comp3, comp3, (float *)cnorm->w, dim, eps, c->decode_state, ratio);
        if (qk_rope)
            rope_heads_dynamic<<<(qk_rope / 2 + 255) / 256, 256, 0, comp>>>(
                comp3, 1, dim, qk_rope, c->decode_state, 1 - ratio, 0, ratio, cache->comp_cos, cache->comp_sin);
        /* The FlashInfer cache packer performs the model's UE8M0 FP8
         * quantization.  Quantizing and dequantizing here first would apply
         * a second, differently-scaled FP8 round before the real cache
         * store.  Keep the simulation only for the float-cache fallback. */
        if (dim > qk_rope)
            fp8_sim64_dynamic<<<(dim - qk_rope + 63) / 64, 64, 0, comp>>>(comp3, dim - qk_rope,
                                                                          c->decode_state, ratio);
        compressed_store_dynamic<<<(dim + 255) / 256, 256, 0, comp>>>(
            cache->compressed, comp3, cache->compressed_len, c->decode_state, ratio, cache->max_compressed, dim);
        int commit_blocks = overlap ? (ratio * O + 255) / 256 : 1;
        compression_commit<<<commit_blocks, 256, 0, comp>>>(
            cache->compressed_len, cache->comp_value, cache->comp_score, c->decode_state, ratio, overlap, O,
            cache->max_compressed);
        if (async && !ok(cudaEventRecord(c->join, c->aux), "attention join record")) return 0;
    }
    int dense_dg=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG"));
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qa,c->dx,c->p1)||(!fused_qkv&&!dg_attention_kv_reuse(c,kv,c->dx,c->p3)))return 0;}else
#endif
    {fp8_sim<<<(H + 127) / 128, 128, 0, c->stream>>>(c->dx, 1, H);
    run_mv<8>((uint8_t *)qa->w, qa->scale, c->dx, c->p1, R, H, 1, c->stream);}
    bf16_round<<<(R + 255) / 256, 256, 0, c->stream>>>(c->p1, R);
    rmsnorm_seq<<<1, 256, 0, c->stream>>>(c->p1, (__nv_bfloat16 *)qn->w, R, eps);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,qb,c->p1,c->p2))return 0;}else
#endif
    {fp8_sim<<<(R + 127) / 128, 128, 0, c->stream>>>(c->p1, 1, R);
    run_mv<8>((uint8_t *)qb->w, qb->scale, c->p1, c->p2, Q, R, 1, c->stream);}
    bf16_round<<<(Q + 255) / 256, 256, 0, c->stream>>>(c->p2, Q);
    int sparse=0;
    if(!sparse)head_rmsnorm<<<heads, 256, 0, c->stream>>>(c->p2, heads, dim, eps);
    if (qk_rope&&!sparse)
        rope_heads_dynamic<<<(heads * (qk_rope / 2) + 255) / 256, 256, 0, c->stream>>>(
            c->p2, heads, dim, qk_rope, c->decode_state, 0, 0, 0, cache->rope_cos, cache->rope_sin);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){}else
#endif
    run_mv<8>((uint8_t *)kv->w, kv->scale, c->dx, c->p3, K, H, 1, c->stream);
    if (!fused_qkv) bf16_round<<<(K + 255) / 256, 256, 0, c->stream>>>(c->p3, K);
    rmsnorm_f32<<<1, 256, 0, c->stream>>>(c->p3, fused_qkv ? c->p1 + R : c->p3, (float *)kvn->w, K, eps);
    if (qk_rope&&!sparse)
        rope_heads_dynamic<<<(qk_rope / 2 + 255) / 256, 256, 0, c->stream>>>(
            c->p3, 1, dim, qk_rope, c->decode_state, 0, 0, 0, cache->rope_cos, cache->rope_sin);
    if(!sparse){if (K > qk_rope) fp8_sim64<<<(K - qk_rope + 63) / 64, 64, 0, c->stream>>>(c->p3, K - qk_rope);cache_store_dynamic<<<(K + 255) / 256, 256, 0, c->stream>>>(cache->kv, c->p3, c->decode_state, cache->window, K);}
    if (async && !ok(cudaStreamWaitEvent(c->stream, c->join, 0), "attention join wait")) return 0;
    attention_window_dynamic<<<heads, 256, (size_t)(cache->window + cache->max_compressed) * 4, c->stream>>>(
        c->p4, c->p2, cache->kv, cache->compressed, (float *)sink->w, heads, dim, cache->window, c->decode_state,
        cache->compressed_len);
    if (qk_rope)
        rope_heads_dynamic<<<(heads * (qk_rope / 2) + 255) / 256, 256, 0, c->stream>>>(
            c->p4, heads, dim, qk_rope, c->decode_state, 0, 1, 0, cache->rope_cos, cache->rope_sin);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_wo_a(c,wa,c->p4,c->p1))return 0;}else
#endif
    run_mv<9>((uint8_t *)wa->w, wa->scale, c->p4, c->p1, WR, wa->I, groups, c->stream);
    bf16_round<<<(WR + 255) / 256, 256, 0, c->stream>>>(c->p1, WR);
#ifdef COLI_DSV4_DEEPGEMM
    if(dense_dg){if(!dg_attention_projection(c,wb,c->p1,output->data))return 0;}else
#endif
    {fp8_sim<<<(WR + 127) / 128, 128, 0, c->stream>>>(c->p1, 1, WR);
    run_mv<8>((uint8_t *)wb->w, wb->scale, c->p1, output->data, H, WR, 1, c->stream);}
    bf16_round<<<(H + 255) / 256, 256, 0, c->stream>>>(output->data, H);
    return ok(cudaGetLastError(), "attention window launch");
}

extern "C" int dsv4_cuda_attention_sparse_batch(const Dsv4CudaActivation*input,Dsv4CudaTensor*an,Dsv4CudaTensor*qkv,Dsv4CudaTensor*qn,Dsv4CudaTensor*qb,Dsv4CudaTensor*kvn,Dsv4CudaTensor*sink,int heads,int dim,int start,int tokens,float eps,Dsv4CudaKvCache*cache,Dsv4CudaActivation*context){
    (void)input;(void)an;(void)qkv;(void)qn;(void)qb;(void)kvn;(void)sink;(void)heads;(void)dim;(void)start;(void)tokens;(void)eps;(void)cache;(void)context;return 0;
}

extern "C" int dsv4_cuda_attention_output_batch(const Dsv4CudaActivation*context,Dsv4CudaTensor*wa,Dsv4CudaTensor*wb,int groups,int tokens,Dsv4CudaActivation*output){
#ifdef COLI_DSV4_DEEPGEMM
    Dev*c=context?ctx(context->device):nullptr;int K=wa?wa->I:0,N=wa&&groups?wa->O/groups:0,WR=wa?wa->O:0,H=wb?wb->O:0;
    if(!c||!wa||!wb||!output||groups<1||tokens<1||wa->fmt!=9||wa->O%groups||wb->fmt!=8||wb->I!=WR||
       wa->device!=context->device||wb->device!=context->device||output->device!=context->device||
       context->elements<(long long)tokens*groups*K||output->elements<(long long)tokens*H||N!=1024||K!=4096)return 0;
    size_t in=(size_t)tokens*K*sizeof(float),part=(size_t)tokens*N*sizeof(float),joined=(size_t)tokens*WR*sizeof(float);
    if(!ok(cudaSetDevice(context->device),"select batched attention output device")||!buf((void**)&c->aux1,&c->aux1cap,in)||
       !buf((void**)&c->aux2,&c->aux2cap,part)||!buf((void**)&c->p1,&c->p1cap,joined))return 0;
    for(int gidx=0;gidx<groups;gidx++){gather_group_rows<<<((long long)tokens*K+255)/256,256,0,c->stream>>>(c->aux1,context->data,tokens,groups,K,gidx);
        Dsv4CudaTensor view=*wa;view.O=N;view.w=(uint8_t*)wa->w+(long long)gidx*N*K;view.dg_scale=wa->dg_scale+(long long)gidx*(K/512)*N;
        if(!dg_dense_batch_dispatch(c,&view,c->aux1,c->aux2,tokens))return 0;
        scatter_group_rows<<<((long long)tokens*N+255)/256,256,0,c->stream>>>(c->p1,c->aux2,tokens,groups,N,gidx);}
    return dg_dense_batch_dispatch(c,wb,c->p1,output->data,tokens);
#else
    /* GENERIC grouped wo_a (fmt 9 view per group) + wo_b via the portable
     * batch matmul; same gather/scatter structure as the DeepGEMM path. */
    Dev*c=context?ctx(context->device):nullptr;int K=wa?wa->I:0,N=wa&&groups?wa->O/groups:0,WR=wa?wa->O:0,H=wb?wb->O:0;
    if(!c||!wa||!wb||!output||groups<1||tokens<1||wa->fmt!=9||wa->O%groups||N%128||K%128||wb->fmt!=8||wb->I!=WR||
       wa->device!=context->device||wb->device!=context->device||output->device!=context->device||
       context->elements<(long long)tokens*groups*K||output->elements<(long long)tokens*H)return 0;
    size_t in=(size_t)tokens*K*sizeof(float),part=(size_t)tokens*N*sizeof(float),joined=(size_t)tokens*WR*sizeof(float);
    if(!ok(cudaSetDevice(context->device),"select generic attention output device")||!buf((void**)&c->aux1,&c->aux1cap,in)||
       !buf((void**)&c->aux2,&c->aux2cap,part)||!buf((void**)&c->p1,&c->p1cap,joined))return 0;
    for(int gidx=0;gidx<groups;gidx++){gather_group_rows<<<((long long)tokens*K+255)/256,256,0,c->stream>>>(c->aux1,context->data,tokens,groups,K,gidx);
        Dsv4CudaTensor view=*wa;view.O=N;view.w=(uint8_t*)wa->w+(long long)gidx*N*K;view.scale=wa->scale+(long long)gidx*(N/128)*(K/128);
        run_mm_batch(&view,c->aux1,c->aux2,tokens,c->stream);
        scatter_group_rows<<<((long long)tokens*N+255)/256,256,0,c->stream>>>(c->p1,c->aux2,tokens,groups,N,gidx);}
    run_mm_batch(wb,c->p1,output->data,tokens,c->stream);
    return ok(cudaGetLastError(),"generic attention output launch");
#endif
}

extern "C" int dsv4_cuda_attention_window_tp2(const Dsv4CudaActivation *input,Dsv4CudaActivation *peer_input,
        const Dsv4CudaAttentionWeights *a,const Dsv4CudaAttentionWeights *b,int ratio,int heads,int dim,int qk_rope,
        int groups,int pos,float eps,Dsv4CudaKvCache *cache,Dsv4CudaKvCache *peer_cache,
        Dsv4CudaActivation *output,Dsv4CudaActivation *peer_output){
#ifndef COLI_DSV4_NCCL
    (void)input;(void)peer_input;(void)a;(void)b;(void)ratio;(void)heads;(void)dim;(void)qk_rope;(void)groups;
    (void)pos;(void)eps;(void)cache;(void)peer_cache;(void)output;(void)peer_output;return 0;
#else
    Dev *p=input?ctx(input->device):nullptr,*q=peer_input?ctx(peer_input->device):nullptr;int H=a&&a->q_a?a->q_a->I:0;
    if(!p||!q||!a||!b||!cache||!peer_cache||!output||!peer_output||!p->ep_comm||!q->ep_comm||heads%2||groups%2||
       peer_input->elements<H||output->elements<H||peer_output->elements<H)return 0;
    int replicated=getenv("DSV4_CUDA_REPLICATED_TP2")&&atoi(getenv("DSV4_CUDA_REPLICATED_TP2"));
#ifdef COLI_DSV4_DEEPGEMM
    const char *qkv_env=getenv("DSV4_CUDA_QKV_FUSED");int fuse_qkv=!qkv_env||atoi(qkv_env);
#else
    int fuse_qkv=0;
#endif
    if(!replicated&&(!ok(cudaSetDevice(p->id),"select TP2 attention primary")||!nok(ncclGroupStart(),"TP2 attention broadcast start")||
       !nok(ncclBroadcast(input->data,input->data,H,ncclFloat,0,p->ep_comm,p->stream),"TP2 attention input root")||
       !ok(cudaSetDevice(q->id),"select TP2 attention peer")||
       !nok(ncclBroadcast(peer_input->data,peer_input->data,H,ncclFloat,0,q->ep_comm,q->stream),"TP2 attention input peer")||
       !nok(ncclGroupEnd(),"TP2 attention broadcast end")))return 0;
#define ATTN_LOCAL(x,w,k,y) dsv4_cuda_attention_window((x),(w)->attn_norm,fuse_qkv?(w)->qkv:(w)->q_a,(w)->q_norm,(w)->q_b,(w)->wkv,\
        (w)->kv_norm,(w)->sink,(w)->wo_a,(w)->wo_b,(w)->compress_wkv,(w)->compress_wgate,(w)->compress_ape,\
        (w)->compress_norm,ratio,heads/2,dim,qk_rope,groups/2,pos,eps,(k),(y))
    if(!ATTN_LOCAL(input,a,cache,output)||!ATTN_LOCAL(peer_input,b,peer_cache,peer_output))return 0;
#undef ATTN_LOCAL
    if(!ok(cudaSetDevice(p->id),"select TP2 attention reduce primary")||!nok(ncclGroupStart(),"TP2 attention reduce start")||
       !nok(ncclAllReduce(output->data,output->data,H,ncclFloat,ncclSum,p->ep_comm,p->stream),"TP2 attention primary reduce")||
       !ok(cudaSetDevice(q->id),"select TP2 attention reduce peer")||
       !nok(ncclAllReduce(peer_output->data,peer_output->data,H,ncclFloat,ncclSum,q->ep_comm,q->stream),"TP2 attention peer reduce")||
       !nok(ncclGroupEnd(),"TP2 attention reduce end"))return 0;
    return ok(cudaSetDevice(p->id),"restore TP2 attention primary");
#endif
}
extern "C" int dsv4_cuda_route(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,const int*fixed,float scale,int ids[6],float weights[6]){
    Dev*c=input?ctx(input->device):nullptr;if(!c||!gate||!ids||!weights||gate->fmt!=32||gate->O!=256||gate->I>input->elements||gate->device!=input->device||(bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||!ok(cudaSetDevice(input->device),"select router device")||!buf((void**)&c->p1,&c->p1cap,256*4)||!buf((void**)&c->expert_weights,&c->expert_weightscap,6*4)||!buf((void**)&c->expert_ids,&c->expert_idcap,6*sizeof(int)))return 0;
    mv_f32<<<256,256,0,c->stream>>>((float*)gate->w,input->data,c->p1,256,gate->I);if(!ok(cudaMemsetAsync(c->expert_ids,0,6*sizeof(int),c->stream),"router ids clear")||(fixed&&!ok(cudaMemcpyAsync(c->expert_ids,fixed,6*sizeof(int),cudaMemcpyHostToDevice,c->stream),"router fixed ids upload")))return 0;route_top6_serial<<<1,1,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->p1,fixed!=nullptr,scale);
    return ok(cudaGetLastError(),"router launch")&&ok(cudaMemcpyAsync(ids,c->expert_ids,6*sizeof(int),cudaMemcpyDeviceToHost,c->stream),"router ids download")&&ok(cudaMemcpyAsync(weights,c->expert_weights,6*sizeof(float),cudaMemcpyDeviceToHost,c->stream),"router weights download")&&ok(cudaStreamSynchronize(c->stream),"router sync");}
extern "C" int dsv4_cuda_matvec_grouped(Dsv4CudaTensor *t,float *y,const float*x,int groups){Dev*c=t?ctx(t->device):nullptr;if(!c||groups<1||t->O%groups||!ok(cudaSetDevice(t->device),"select matvec device"))return 0;size_t xb=(size_t)t->I*groups*4,yb=(size_t)t->O*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"activation upload"))return 0;
    if(t->fmt==8)run_mv<8>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==9)run_mv<9>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==4)run_mv<4>((uint8_t*)t->w,t->scale,c->dx,c->dy,t->O,t->I,groups,c->stream);
    else if(t->fmt==32)mv_f32<<<t->O,256,0,c->stream>>>((float*)t->w,c->dx,c->dy,t->O,t->I);
    else launch_bf16_one((__nv_bfloat16*)t->w,c->dx,c->dy,t->O,t->I,c->stream);
    return ok(cudaGetLastError(),"matvec launch")&&ok(cudaMemcpyAsync(y,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),"result download")&&ok(cudaStreamSynchronize(c->stream),"matvec sync");}
extern "C" int dsv4_cuda_matvec(Dsv4CudaTensor*t,float*y,const float*x){return dsv4_cuda_matvec_grouped(t,y,x,1);}
/* One token's projections of one input through several mirrors (bf16, fp8
 * or fp8-bf16) in one round trip: one upload, a launch per mirror (the
 * kernel each mirror's own call runs: mm_bf16_batch with a batch of one,
 * run_mv for fp8), the downloads, one sync. The outputs are what the
 * separate calls return, bitwise. */
extern "C" int dsv4_cuda_decode_proj_multi(Dsv4CudaTensor*const*ts,int n,const float*x,float*const*ys){
    if(!ts||!ys||n<1||n>8||!x||!ts[0])return 0;
    Dev*c=ctx(ts[0]->device);
    if(!c||!ok(cudaSetDevice(ts[0]->device),"select decode proj device"))return 0;
    size_t xb=(size_t)ts[0]->I*4,yb=0;
    for(int i=0;i<n;i++){
        if(!ts[i]||!ys[i]||ts[i]->device!=ts[0]->device||ts[i]->I!=ts[0]->I)return 0;
        int f=ts[i]->fmt;if(f!=16&&f!=8&&f!=9)return 0;
        yb+=(size_t)ts[i]->O*4;
    }
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,yb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,xb,cudaMemcpyHostToDevice,c->stream),"decode proj upload"))return 0;
    static int kprof=-1;
    if(kprof<0){const char*e=getenv("DSV4_PROJ_KPROF");kprof=e&&*e&&*e!='0';}
    cudaEvent_t ev[9];
    if(kprof){for(int i=0;i<=n;i++)cudaEventCreate(&ev[i]);cudaEventRecord(ev[0],c->stream);}
    size_t off=0;
    for(int i=0;i<n;i++){
        size_t ob=(size_t)ts[i]->O*4;float*dy=(float*)((char*)c->dy+off);
        if(ts[i]->fmt==16)launch_bf16_one((__nv_bfloat16*)ts[i]->w,c->dx,dy,ts[i]->O,ts[i]->I,c->stream);
        else if(ts[i]->fmt==8)run_mv<8>((uint8_t*)ts[i]->w,ts[i]->scale,c->dx,dy,ts[i]->O,ts[i]->I,1,c->stream);
        else run_mv<9>((uint8_t*)ts[i]->w,ts[i]->scale,c->dx,dy,ts[i]->O,ts[i]->I,1,c->stream);
        if(kprof)cudaEventRecord(ev[i+1],c->stream);
        if(!ok(cudaGetLastError(),"decode proj launch")||
           !ok(cudaMemcpyAsync(ys[i],dy,ob,cudaMemcpyDeviceToHost,c->stream),"decode proj download"))return 0;
        off+=ob;
    }
    int done=ok(cudaStreamSynchronize(c->stream),"decode proj sync");
    if(kprof){
        fprintf(stderr,"projkprof n=%d",n);
        for(int i=0;i<n;i++){float ms=0;cudaEventElapsedTime(&ms,ev[i],ev[i+1]);fprintf(stderr," [%d]%s%dx%d=%.3f",i,ts[i]->fmt==16?"bf16":"fp8",ts[i]->O,ts[i]->I,ms);}
        fprintf(stderr," ms\n");
        for(int i=0;i<=n;i++)cudaEventDestroy(ev[i]);
    }
    return done;
}
extern "C" int dsv4_cuda_matmul_batch(Dsv4CudaTensor*t,const Dsv4CudaActivation*input,int tokens,Dsv4CudaActivation*output){
    Dev*c=t?ctx(t->device):nullptr;if(!c||!input||!output||tokens<1||input->device!=t->device||output->device!=t->device||
       input->elements<(long long)tokens*t->I||output->elements<(long long)tokens*t->O||!ok(cudaSetDevice(t->device),"select batched dense device"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    return dg_dense_batch_dispatch(c,t,input->data,output->data,tokens);
#else
    if(t->fmt!=8&&t->fmt!=9&&t->fmt!=4)return 0;
    run_mm_batch(t,input->data,output->data,tokens,c->stream);
    return ok(cudaGetLastError(),"batched dense launch");
#endif
}
extern "C" int dsv4_cuda_head_argmax(Dsv4CudaTensor*t,const float*x,int*id,float*value){
#ifdef COLI_DSV4_DEEPGEMM
    constexpr int O=129280,I=4096;Dev*c=t?ctx(t->device):nullptr;if(!c||!x||!id||!value||t->fmt!=16||t->O!=O||t->I!=I||!ok(cudaSetDevice(t->device),"select output head device")||
       !buf((void**)&c->dx,&c->xcap,I*sizeof(float))||!buf((void**)&c->dga1,&c->dga1cap,I*sizeof(__nv_bfloat16))||!buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||
       !buf((void**)&c->expert_ids,&c->expert_idcap,sizeof(int))||!buf((void**)&c->expert_weights,&c->expert_weightscap,sizeof(float))||
       !ok(cudaMemcpyAsync(c->dx,x,I*sizeof(float),cudaMemcpyHostToDevice,c->stream),"output head input upload"))return 0;
    dg_float_to_bf16<<<(I+255)/256,256,0,c->stream>>>((__nv_bfloat16*)c->dga1,c->dx,I);CUtensorMap a,b,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dga1,I,1,(unsigned long long)I*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,t->w,I,O,(unsigned long long)I*2,64,128,CU_TENSOR_MAP_SWIZZLE_128B)||
       !dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>:deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>;
    if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,90368),"output head DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,90368,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(DgOutput*)c->dga1,(DgOutput*)t->w,nullptr,nullptr,1,O,I,a,b,d);
    dg_argmax_bf16<<<1,256,0,c->stream>>>(c->dgmm1,O,c->expert_ids,c->expert_weights);
    return ok(cudaGetLastError(),"output head DeepGEMM launch")&&ok(cudaMemcpyAsync(id,c->expert_ids,sizeof(*id),cudaMemcpyDeviceToHost,c->stream),"output id download")&&
           ok(cudaMemcpyAsync(value,c->expert_weights,sizeof(*value),cudaMemcpyDeviceToHost,c->stream),"output logit download")&&ok(cudaStreamSynchronize(c->stream),"output head sync");
#else
    (void)t;(void)x;(void)id;(void)value;return 0;
#endif
}
extern "C" int dsv4_cuda_final_argmax(const Dsv4CudaActivation*r,Dsv4CudaTensor*fn,Dsv4CudaTensor*scale,Dsv4CudaTensor*base,Dsv4CudaTensor*norm,Dsv4CudaTensor*head,int M,int H,float eps,float pre_eps,int*id,float*value){
#ifdef COLI_DSV4_DEEPGEMM
    constexpr int O=129280,I=4096;Dev*c=r?ctx(r->device):nullptr;int MH=M*H;if(!c||!fn||!scale||!base||!norm||!head||!id||!value||M!=4||H!=I||r->elements<MH||fn->fmt!=32||fn->O!=M||fn->I!=MH||scale->fmt!=32||base->fmt!=32||norm->fmt!=32||head->fmt!=16||head->O!=O||head->I!=I||
       fn->device!=r->device||scale->device!=r->device||base->device!=r->device||norm->device!=r->device||head->device!=r->device||!ok(cudaSetDevice(r->device),"select final head device")||
       !buf((void**)&c->dx,&c->xcap,I*sizeof(float))||!buf((void**)&c->p1,&c->p1cap,M*sizeof(float))||!buf((void**)&c->dga1,&c->dga1cap,I*sizeof(__nv_bfloat16))||!buf((void**)&c->dgmm1,&c->dgmm1cap,O*sizeof(__nv_bfloat16))||
       !buf((void**)&c->expert_ids,&c->expert_idcap,sizeof(int))||!buf((void**)&c->expert_weights,&c->expert_weightscap,sizeof(float)))return 0;
    final_mhc_coeff<<<1,256,0,c->stream>>>(c->p1,r->data,(float*)fn->w,(float*)scale->w,(float*)base->w,M,H,eps,pre_eps);final_mhc_mix<<<(H+255)/256,256,0,c->stream>>>(c->dx,r->data,c->p1,M,H);rmsnorm_f32<<<1,256,0,c->stream>>>(c->dx,c->dx,(float*)norm->w,H,eps);dg_float_to_bf16<<<(I+255)/256,256,0,c->stream>>>((__nv_bfloat16*)c->dga1,c->dx,I);CUtensorMap a,b,d;
    if(!dg_map(&a,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dga1,I,1,(unsigned long long)I*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&b,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,head->w,I,O,(unsigned long long)I*2,64,128,CU_TENSOR_MAP_SWIZZLE_128B)||!dg_map(&d,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,c->dgmm1,O,1,(unsigned long long)O*2,64,64,CU_TENSOR_MAP_SWIZZLE_128B))return 0;
    auto gemm=c->sms>=170?deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,170,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>:deep_gemm::sm120_bf16_gemm_impl<0,O,I,1,64,128,64,128,128,128,3,128,256,84,deep_gemm::GemmType::Normal,false,DgOutput,DgEpilogue,true,2>;if(!ok(cudaFuncSetAttribute(gemm,cudaFuncAttributeMaxDynamicSharedMemorySize,90368),"final head DeepGEMM shared memory"))return 0;
    gemm<<<(c->sms>=170?170:84),384,90368,c->stream>>>((DgOutput*)c->dgmm1,nullptr,(DgOutput*)c->dga1,(DgOutput*)head->w,nullptr,nullptr,1,O,I,a,b,d);dg_argmax_bf16<<<1,256,0,c->stream>>>(c->dgmm1,O,c->expert_ids,c->expert_weights);
    return ok(cudaGetLastError(),"final head launch")&&ok(cudaMemcpyAsync(id,c->expert_ids,sizeof(*id),cudaMemcpyDeviceToHost,c->stream),"final id download")&&ok(cudaMemcpyAsync(value,c->expert_weights,sizeof(*value),cudaMemcpyDeviceToHost,c->stream),"final logit download")&&ok(cudaStreamSynchronize(c->stream),"final head sync");
#else
    (void)r;(void)fn;(void)scale;(void)base;(void)norm;(void)head;(void)M;(void)H;(void)eps;(void)pre_eps;(void)id;(void)value;return 0;
#endif
}
#if COLI_DSV4_TC
static TcPlan *tc_plan(Dev*c,int O,int I){int pi=O>I;TcPlan*p=&c->plans[pi];if(p->ok)return p;if(p->op)plan_free(p);p->O=O;p->I=I;
    cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;cublasLtMatmulMatrixScale_t mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    if(!bok(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"TC operation")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)),"TC trans A")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)),"TC trans B")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)),"TC A scale mode")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)),"TC B scale mode")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&c->tcs,sizeof(c->tcs)),"TC initial A scale")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&c->tcxs,sizeof(c->tcxs)),"TC initial B scale")||
       !bok(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_8F_E4M3,I,O,I),"TC A layout")||
       !bok(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_8F_E4M3,I,1,I),"TC B layout")||
       !bok(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_32F,O,1,O),"TC C layout")||
       !bok(cublasLtMatrixLayoutCreate(&p->d,CUDA_R_32F,O,1,O),"TC D layout"))return NULL;
    cublasLtMatmulPreference_t pref=NULL;size_t wsz=32<<20;int found=0;if(!bok(cublasLtMatmulPreferenceCreate(&pref),"TC preference")||
       !bok(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&wsz,sizeof(wsz)),"TC workspace preference")||
       !bok(cublasLtMatmulAlgoGetHeuristic(c->lt,p->op,p->a,p->b,p->c,p->d,pref,1,&p->algo,&found),"TC heuristic")){if(pref)cublasLtMatmulPreferenceDestroy(pref);return NULL;}
    cublasLtMatmulPreferenceDestroy(pref);if(!found){fprintf(stderr,"[DSV4 CUDA] no MXFP8 Tensor Core plan O=%d I=%d\n",O,I);return NULL;}p->ok=1;return p;}
static int tc_prepare_x(Dev*c,const float*x,int I){size_t sb=(size_t)((I+127)/128)*4*128;if(!buf((void**)&c->tcx,&c->tcxcap,I)||!buf((void**)&c->tcxs,&c->tcxscap,sb))return 0;
    if(!ok(cudaMemsetAsync(c->tcxs,0,sb,c->stream),"TC activation scale clear"))return 0;tc_quant<<<(I+127)/128,128,0,c->stream>>>(x,c->tcx,c->tcxs,I);return ok(cudaGetLastError(),"TC activation quantize");}
static int tc_fp4_matvec(Dev*c,Dsv4CudaTensor*t,float*y){int O=t->O,I=t->I;size_t wb=(size_t)O*I,sb=(size_t)((I+127)/128)*4*((O+127)/128)*128;
    if(!buf((void**)&c->tcw,&c->tcwcap,wb)||!buf((void**)&c->tcs,&c->tcscap,sb))return 0;TcPlan*p=tc_plan(c,O,I);if(!p)return 0;
    tc_expand<<<256,256,0,c->stream>>>((uint8_t*)t->w,t->scale,c->tcw,c->tcs,O,I);if(!ok(cudaGetLastError(),"TC weight expansion"))return 0;
    void *as=c->tcs,*bs=c->tcxs;if(!bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&as,sizeof(as)),"TC A scale")||
       !bok(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&bs,sizeof(bs)),"TC B scale"))return 0;float alpha=1,beta=0;
    return bok(cublasLtMatmul(c->lt,p->op,&alpha,c->tcw,p->a,c->tcx,p->b,&beta,y,p->c,y,p->d,&p->algo.algo,c->tc_workspace,32<<20,c->stream),"TC matvec");}
#endif /* COLI_DSV4_TC */
/* Upload fence: the decode uploader threads enqueue their mirror refills on
 * the aux stream and return without draining (V4_UPLOAD_FENCE); the expert
 * group orders its kernels after those copies on the device, instead of the
 * host joining the DMA and only then launching. Cheap when nothing is
 * pending. Exported too, so the engine can probe for it. */
/* Precaution for the event fence: WDDM may submit a thread's queued work
 * lazily, so an enqueue-only producer (the uploader threads, the decode
 * lanes' pieces) flushes its submission with cudaStreamQuery after its
 * last copy, before the main thread can record the fence event on the
 * same stream. (Not measured to be necessary; the wrong text seen while
 * the pieces were built came from page-locking a slab under the drive's
 * DMA, see dsv4_cuda_host_pin.) */
static void aux_flush(Dev*c){cudaError_t e=cudaStreamQuery(c->aux);if(e!=cudaSuccess&&e!=cudaErrorNotReady)ok(e,"aux flush");else cudaGetLastError();}
/* the prefetched mirrors among a group's experts: the compute stream
 * waits for each one's event, once (the prefetch stream is not fenced) */
static int group_wait_pre(Dev*c,Dsv4CudaTensor *const *gate,int count){
    for(int e=0;e<count;e++){Dsv4CudaTensor*t=gate[e];if(t&&t->evwait&&t->ev){
        if(!ok(cudaStreamWaitEvent(c->stream,t->ev,0),"prefetch mirror wait"))return 0;t->evwait=0;}}
    return 1;}
static int group_fence(Dev*c){
    if(!c->upfence&&!ok(cudaEventCreateWithFlags(&c->upfence,cudaEventDisableTiming),"upload fence event"))return 0;
    return ok(cudaEventRecord(c->upfence,c->aux),"upload fence record")&&ok(cudaStreamWaitEvent(c->stream,c->upfence,0),"upload fence wait");
}
extern "C" int dsv4_cuda_group_fence(int device){
    Dev*c=ctx(device);if(!c||!ok(cudaSetDevice(device),"select fence device"))return 0;return group_fence(c);
}
/* Pinned staging for the group's small host<->device copies: a pageable
 * cudaMemcpyAsync stages synchronously and, on WDDM, flushes the command
 * buffer per copy (four of them per group call). DSV4_MOE_PINNED=0 keeps
 * the pageable copies. The descriptors of all three matrices and the
 * weights go up in ONE copy: gate/up at [0,2n), down at [GRP_DOWN,+n). */
enum{GRP_DOWN=32,GRP_DESC=48,GRP_W=16};
/* Decode matvecs, bandwidth-shaped (DSV4_MOE_MV, on): one warp per row,
 * 16-byte weight loads (32 fp4 or 16 fp8 values per lane per step), the
 * input vector in shared memory padded one float per lane-chunk so the
 * lanes' strided reads hit distinct banks, scales and values decoded with
 * bit arithmetic instead of table lookups, warp-shuffle reduction. The
 * mv / mv_fp4_grouped kernels read one byte per lane per step and ran at
 * ~15 % of the card's bandwidth (1.1 ms per decode layer for ~100 MB).
 * The summation order differs from theirs: DSV4_MOE_MV=0 restores them. */
__device__ __forceinline__ float f4_bits(uint32_t q){uint32_t e=(q>>1)&3u,m=q&1u,sg=(q&8u)<<28;
    uint32_t b=e?(((e+126u)<<23)|(m<<22)):(m?0x3f000000u:0u);return __uint_as_float(b|sg);}
__device__ __forceinline__ float e8_bits(uint32_t b){return b?__uint_as_float(b<<23):__uint_as_float(1u<<22);}
__device__ __forceinline__ float e4m3_bits(uint32_t v){uint32_t sg=(v&0x80u)<<24,e=(v>>3)&15u,m=v&7u;
    float r=e?__uint_as_float(((e+120u)<<23)|(m<<20)):(float)m*(1.f/512.f);return __uint_as_float(__float_as_uint(r)|sg);}
template<int RW> __global__ void __launch_bounds__(256) mv_fp4_dec(const MvDesc*desc,int count,int O,int I){
    extern __shared__ float xs[];
    int matrix=blockIdx.y;if(matrix>=count)return;MvDesc d=desc[matrix];
    for(int e=threadIdx.x;e<I;e+=blockDim.x)xs[e+(e>>5)]=d.x[e];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,chunks=I>>5;
    int row0=(blockIdx.x*(blockDim.x>>5)+warp)*RW;
    #pragma unroll
    for(int r=0;r<RW;r++){int o=row0+r;if(o>=O)break;
        const uint4*wr=(const uint4*)(d.w+(size_t)o*(I>>1));const uint8_t*sr=d.scale+(size_t)o*chunks;float sum=0.f;
        for(int c=lane;c<chunks;c+=32){uint4 w=__ldg(wr+c);float sc=e8_bits(__ldg(sr+c));const float*xp=xs+c*33;float part=0.f;
            uint32_t ws[4]={w.x,w.y,w.z,w.w};
            #pragma unroll
            for(int k=0;k<4;k++){uint32_t v=ws[k];
                #pragma unroll
                for(int n=0;n<8;n++)part+=xp[k*8+n]*f4_bits((v>>(4*n))&15u);}
            sum+=part*sc;}
        #pragma unroll
        for(int off=16;off;off>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,off);
        if(!lane)d.y[o]=sum;}
}
/* RND: fmt 9, the dequantized weight bf16-rounded per element before the
 * product (as mv<9>); blockIdx.y: the group (the matrix as `groups`
 * independent blocks of O rows, each over its own I-slice of x). */
template<int RW,int RND> __global__ void __launch_bounds__(256) mv_fp8_dec(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I){
    extern __shared__ float xs[];
    int grp=blockIdx.y;w+=(size_t)grp*O*I;sc+=(size_t)(grp*O/128)*((I+127)/128);x+=(size_t)grp*I;y+=(size_t)grp*O;
    for(int e=threadIdx.x;e<I;e+=blockDim.x)xs[e+(e>>4)]=x[e];
    __syncthreads();
    int warp=threadIdx.x>>5,lane=threadIdx.x&31,chunks=I>>4,sb=(I+127)/128;
    int row0=(blockIdx.x*(blockDim.x>>5)+warp)*RW;
    #pragma unroll
    for(int r=0;r<RW;r++){int o=row0+r;if(o>=O)break;
        const uint4*wr=(const uint4*)(w+(size_t)o*I);const uint8_t*srow=sc+(size_t)(o/128)*sb;float sum=0.f;
        for(int c=lane;c<chunks;c+=32){uint4 v=__ldg(wr+c);float s=e8_bits(__ldg(srow+(c>>3)));const float*xp=xs+c*17;float part=0.f;
            uint32_t ws[4]={v.x,v.y,v.z,v.w};
            #pragma unroll
            for(int k=0;k<4;k++){uint32_t b=ws[k];
                #pragma unroll
                for(int n=0;n<4;n++){float wv=e4m3_bits((b>>(8*n))&255u);
                    if constexpr(RND)part+=xp[k*4+n]*__bfloat162float(__float2bfloat16(wv*s));else part+=xp[k*4+n]*wv;}}
            if constexpr(RND)sum+=part;else sum+=part*s;}
        #pragma unroll
        for(int off=16;off;off>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,off);
        if(!lane)y[o]=sum;}
}
static unsigned float_bits_host(float f){unsigned u;memcpy(&u,&f,4);return u;}
static int g_mv_dec=-1;
static int mv_dec_on(void){if(g_mv_dec<0){const char*e=getenv("DSV4_MOE_MV");g_mv_dec=!e||!*e||*e!='0';}return g_mv_dec;}
/* grouped fp4 matvecs of `count` matrices, each O rows by I (I % 32 == 0) */
static void run_fp4_dec(const MvDesc*desc,int count,int O,int I,cudaStream_t st){
    mv_fp4_dec<4><<<dim3((O+31)/32,count),256,(size_t)(I+(I>>5))*sizeof(float),st>>>(desc,count,O,I);}
static void run_fp8_dec(const uint8_t*w,const uint8_t*sc,const float*x,float*y,int O,int I,cudaStream_t st){
    mv_fp8_dec<4,0><<<(O+31)/32,256,(size_t)(I+(I>>4))*sizeof(float),st>>>(w,sc,x,y,O,I);}
/* DSV4_ATTN_MV (1): the decode attention chain's fp8 matvecs on the shaped
 * kernel; a matrix it cannot take (I not a multiple of 16, the x slice past
 * the shared-memory budget, O not a multiple of 128 per group) keeps mv. */
static int attn_mv_on(void){static int on=-1;if(on<0){const char*e=getenv("DSV4_ATTN_MV");on=!e||!*e||*e!='0';}return on;}
static bool run_fp8_attn(const Dsv4CudaTensor*t,const float*x,float*y,int groups,cudaStream_t st){
    int Og=t->O/groups,I=t->I;size_t sm=(size_t)(I+(I>>4))*sizeof(float);
    if(!attn_mv_on()||groups<1||t->O%groups||I%16||Og%128||sm>48*1024)return false;
    dim3 grid((Og+31)/32,groups);
    if(t->fmt==9)mv_fp8_dec<4,1><<<grid,256,sm,st>>>((const uint8_t*)t->w,t->scale,x,y,Og,I);
    else mv_fp8_dec<4,0><<<grid,256,sm,st>>>((const uint8_t*)t->w,t->scale,x,y,Og,I);
    return true;}
static int group_stage(Dev*c,size_t xb){
    if(c->grp_pinned<0)return 0;
    if(c->grp_pinned>0)return xb<=c->grp_hcap;
    static int want=-1;if(want<0){const char*e=getenv("DSV4_MOE_PINNED");want=!e||!*e||*e!='0';}
    if(!want){c->grp_pinned=-1;return 0;}
    size_t pb=GRP_DESC*sizeof(MvDesc)+GRP_W*sizeof(float);
    if(!ok(cudaHostAlloc(&c->grp_hparams,pb,cudaHostAllocPortable),"group params staging")||
       !ok(cudaHostAlloc((void**)&c->hx_pin,xb,cudaHostAllocPortable),"group x staging")||
       !ok(cudaHostAlloc((void**)&c->hy_pin,xb,cudaHostAllocPortable),"group y staging")||
       !buf(&c->grp_params,&c->grp_paramscap,pb)){c->grp_pinned=-1;return 0;}
    c->grp_hcap=xb;c->grp_pinned=1;return 1;
}
static int group_params(Dev*c,int pinned,Dsv4CudaTensor*const*gate,Dsv4CudaTensor*const*up,Dsv4CudaTensor*const*down,
        const float*weights,int count,int I,int H,MvDesc**dd,float**dw){
    MvDesc hl[GRP_DESC]={};float wl[GRP_W]={0};
    MvDesc*hd=pinned?(MvDesc*)c->grp_hparams:hl;float*hw=pinned?(float*)((char*)c->grp_hparams+GRP_DESC*sizeof(MvDesc)):wl;
    for(int e=0;e<count;e++){
        hd[2*e]={(uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I};
        hd[2*e+1]={(uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I};
        hd[GRP_DOWN+e]={(uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H};
        hw[e]=weights[e];}
    size_t pb=GRP_DESC*sizeof(MvDesc)+GRP_W*sizeof(float);
    if(pinned){if(!ok(cudaMemcpyAsync(c->grp_params,c->grp_hparams,pb,cudaMemcpyHostToDevice,c->stream),"expert params upload"))return 0;
        *dd=(MvDesc*)c->grp_params;*dw=(float*)((char*)c->grp_params+GRP_DESC*sizeof(MvDesc));return 1;}
    if(!buf((void**)&c->mvdesc,&c->mvdesccap,GRP_DESC*sizeof(MvDesc))||!buf((void**)&c->expert_weights,&c->expert_weightscap,GRP_W*sizeof(float))||
       !ok(cudaMemcpyAsync(c->mvdesc,hl,GRP_DESC*sizeof(MvDesc),cudaMemcpyHostToDevice,c->stream),"expert descriptors upload")||
       !ok(cudaMemcpyAsync(c->expert_weights,wl,GRP_W*sizeof(float),cudaMemcpyHostToDevice,c->stream),"expert weights upload"))return 0;
    *dd=c->mvdesc;*dw=c->expert_weights;return 1;
}
/* DSV4_MOE_KPROF=1: device time of the group's kernels vs the call's host
 * wall (which includes the fence wait for pending uploads), per card. */
struct GroupProf{Dev*c;int on;cudaEvent_t e0,e1;std::chrono::steady_clock::time_point t0;
    GroupProf(Dev*d):c(d){static int k=-1;if(k<0){const char*e=getenv("DSV4_MOE_KPROF");k=e&&*e&&*e!='0';}on=k;
        if(on){t0=std::chrono::steady_clock::now();cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventRecord(e0,c->stream);}}};
static int group_finish(Dev*c,GroupProf&gp,int pinned,float*y,size_t yb,const char*what){
    if(gp.on)cudaEventRecord(gp.e1,c->stream);
    float*ys=pinned?c->hy_pin:y;char msg[64];
    snprintf(msg,sizeof msg,"%s result download",what);
    if(!ok(cudaMemcpyAsync(ys,c->dy,yb,cudaMemcpyDeviceToHost,c->stream),msg))return 0;
    snprintf(msg,sizeof msg,"%s sync",what);
    if(!ok(cudaStreamSynchronize(c->stream),msg))return 0;
    if(pinned)memcpy(y,c->hy_pin,yb);
    if(gp.on){float ms=0;cudaEventElapsedTime(&ms,gp.e0,gp.e1);cudaEventDestroy(gp.e0);cudaEventDestroy(gp.e1);
        c->kp_dev+=ms;c->kp_host+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-gp.t0).count();
        if(++c->kp_n%430==0)fprintf(stderr,"moekprof device=%d calls=%d kernels=%.3f host=%.3f ms/call pinned=%d\n",c->id,c->kp_n,c->kp_dev/c->kp_n,c->kp_host/c->kp_n,pinned);}
    return 1;
}
extern "C" int dsv4_cuda_expert_group(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,
        const float *weights,int count,float limit,float *y,const float *x){
    if(count<1||count>16||!gate||!up||!down)return 0;Dsv4CudaTensor *t=gate[0];Dev*c=t?ctx(t->device):nullptr;
    if(!c||t->fmt!=4||!ok(cudaSetDevice(t->device),"select expert-group device"))return 0;int H=t->I,I=t->O;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||gate[e]->device!=t->device||up[e]->device!=t->device||down[e]->device!=t->device||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4||gate[e]->I!=H||gate[e]->O!=I||up[e]->I!=H||up[e]->O!=I||down[e]->I!=I||down[e]->O!=H)return 0;
    size_t xb=(size_t)H*4,ib=(size_t)count*I*4,hb=(size_t)count*H*4;
    if(!buf((void**)&c->dx,&c->xcap,xb)||!buf((void**)&c->dy,&c->ycap,(size_t)H*4)||
       !buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!group_fence(c)||!group_wait_pre(c,gate,count))return 0;int pinned=group_stage(c,xb);const float*xs=x;if(pinned){memcpy(c->hx_pin,x,xb);xs=c->hx_pin;}
    if(!ok(cudaMemcpyAsync(c->dx,xs,xb,cudaMemcpyHostToDevice,c->stream),"expert-group activation upload"))return 0;GroupProf gp(c);
#if COLI_DSV4_TC
    static int tc=-1;if(tc<0)tc=getenv("DSV4_CUDA_TC")?atoi(getenv("DSV4_CUDA_TC")):0;
    if(tc){for(int e=0;e<count;e++){if(!tc_prepare_x(c,c->dx,H)||!tc_fp4_matvec(c,gate[e],c->p1+(long long)e*I)||!tc_fp4_matvec(c,up[e],c->p2+(long long)e*I))return 0;
            expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);
            if(!tc_prepare_x(c,c->p3+(long long)e*I,I)||!tc_fp4_matvec(c,down[e],c->p4+(long long)e*H))return 0;}
        expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,count,H);
        return ok(cudaGetLastError(),"TC expert-group launch")&&ok(cudaMemcpyAsync(y,c->dy,(size_t)H*4,cudaMemcpyDeviceToHost,c->stream),"TC expert-group result download")&&ok(cudaStreamSynchronize(c->stream),"TC expert-group sync");}
#endif /* COLI_DSV4_TC */
    static int batched=-1;if(batched<0)batched=getenv("DSV4_CUDA_BATCHED")?atoi(getenv("DSV4_CUDA_BATCHED")):1;
    if(batched){MvDesc*dd;float*dw;if(!group_params(c,pinned,gate,up,down,weights,count,I,H,&dd,&dw))return 0;int dec=mv_dec_on()&&H%32==0&&I%32==0;
        if(dec)run_fp4_dec(dd,2*count,I,H,c->stream);else mv_fp4_grouped<4><<<dim3((I+3)/4,2*count),256,0,c->stream>>>(dd,2*count,I,H);
        expert_act_grouped<<<(count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,dw,limit,count,I);
        fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
        if(dec)run_fp4_dec(dd+GRP_DOWN,count,H,I,c->stream);else mv_fp4_grouped<4><<<dim3((H+3)/4,count),256,0,c->stream>>>(dd+GRP_DOWN,count,H,I);
    }else for(int e=0;e<count;e++){
        run_mv<4>((uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I,I,H,1,c->stream);
        run_mv<4>((uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I,I,H,1,c->stream);
        expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);
        fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3+(long long)e*I,1,I);
        run_mv<4>((uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H,H,I,1,c->stream);
    }
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,count,H);
    return ok(cudaGetLastError(),"expert-group launch")&&group_finish(c,gp,pinned,y,(size_t)H*4,"expert-group");
}
extern "C" int dsv4_cuda_expert_fp8(Dsv4CudaTensor *gate,Dsv4CudaTensor *up,Dsv4CudaTensor *down,float limit,float *y,const float *x){
    Dev*c=gate?ctx(gate->device):nullptr;if(!c||!up||!down||gate->fmt!=8||up->fmt!=8||down->fmt!=8||
       up->device!=gate->device||down->device!=gate->device||!ok(cudaSetDevice(gate->device),"select shared-expert device"))return 0;
    int H=gate->I,I=gate->O;if(up->I!=H||up->O!=I||down->I!=I||down->O!=H)return 0;
    size_t hb=(size_t)H*4,ib=(size_t)I*4;if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,hb)||
       !buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,hb,cudaMemcpyHostToDevice,c->stream),"shared-expert activation upload"))return 0;
    run_mv<8>((uint8_t*)gate->w,gate->scale,c->dx,c->p1,I,H,1,c->stream);
    run_mv<8>((uint8_t*)up->w,up->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);
    fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    run_mv<8>((uint8_t*)down->w,down->scale,c->p3,c->p4,H,I,1,c->stream);
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,1,H);
    return ok(cudaGetLastError(),"shared-expert launch")&&ok(cudaMemcpyAsync(y,c->dy,hb,cudaMemcpyDeviceToHost,c->stream),"shared-expert result download")&&ok(cudaStreamSynchronize(c->stream),"shared-expert sync");
}
extern "C" int dsv4_cuda_moe(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,
        const float *weights,int count,Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd,float limit,float *y,const float *x){
    if(count<1||count>16||!gate||!up||!down||!sg||!su||!sd)return 0;Dsv4CudaTensor*t=gate[0];Dev*c=t?ctx(t->device):nullptr;
    if(!c||t->fmt!=4||sg->device!=t->device||su->device!=t->device||sd->device!=t->device||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||!ok(cudaSetDevice(t->device),"select MoE device"))return 0;
    int H=t->I,I=t->O;if(sg->I!=H||sg->O!=I||su->I!=H||su->O!=I||sd->I!=I||sd->O!=H)return 0;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||gate[e]->device!=t->device||up[e]->device!=t->device||down[e]->device!=t->device||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4)return 0;
    size_t hb=(size_t)H*4,ib=(size_t)count*I*4,parts=(size_t)count*H*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,hb)||!buf((void**)&c->p1,&c->p1cap,ib)||
       !buf((void**)&c->p2,&c->p2cap,ib)||!buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,parts))return 0;
    if(!group_fence(c)||!group_wait_pre(c,gate,count))return 0;int pinned=group_stage(c,hb);const float*xs=x;if(pinned){memcpy(c->hx_pin,x,hb);xs=c->hx_pin;}
    if(!ok(cudaMemcpyAsync(c->dx,xs,hb,cudaMemcpyHostToDevice,c->stream),"MoE activation upload"))return 0;GroupProf gp(c);
    static int batched=-1;if(batched<0)batched=getenv("DSV4_CUDA_BATCHED")?atoi(getenv("DSV4_CUDA_BATCHED")):1;
    int dec=mv_dec_on()&&H%32==0&&I%32==0;
    if(batched){MvDesc*dd;float*dw;if(!group_params(c,pinned,gate,up,down,weights,count,I,H,&dd,&dw))return 0;
        if(dec)run_fp4_dec(dd,2*count,I,H,c->stream);else mv_fp4_grouped<4><<<dim3((I+3)/4,2*count),256,0,c->stream>>>(dd,2*count,I,H);
        expert_act_grouped<<<(count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,dw,limit,count,I);fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
        if(dec)run_fp4_dec(dd+GRP_DOWN,count,H,I,c->stream);else mv_fp4_grouped<4><<<dim3((H+3)/4,count),256,0,c->stream>>>(dd+GRP_DOWN,count,H,I);
    }else for(int e=0;e<count;e++){run_mv<4>((uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I,I,H,1,c->stream);run_mv<4>((uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I,I,H,1,c->stream);expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3+(long long)e*I,1,I);run_mv<4>((uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H,H,I,1,c->stream);}
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,count,H);
    if(dec){run_fp8_dec((uint8_t*)sg->w,sg->scale,c->dx,c->p1,I,H,c->stream);run_fp8_dec((uint8_t*)su->w,su->scale,c->dx,c->p2,I,H,c->stream);}
    else{run_mv<8>((uint8_t*)sg->w,sg->scale,c->dx,c->p1,I,H,1,c->stream);run_mv<8>((uint8_t*)su->w,su->scale,c->dx,c->p2,I,H,1,c->stream);}
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    if(dec)run_fp8_dec((uint8_t*)sd->w,sd->scale,c->p3,c->p4,H,I,c->stream);else run_mv<8>((uint8_t*)sd->w,sd->scale,c->p3,c->p4,H,I,1,c->stream);
    moe_combine<<<(H+255)/256,256,0,c->stream>>>(c->dy,c->p4,H);
    if(!(ok(cudaGetLastError(),"MoE launch")&&group_finish(c,gp,pinned,y,hb,"MoE")))return 0;
    /* DSV4_MOE_MV_VERIFY=1: the same call through the old kernels, compared */
    static int verify=-1;static int in_verify=0;
    if(verify<0){const char*e=getenv("DSV4_MOE_MV_VERIFY");verify=e&&*e&&*e!='0';}
    if(verify&&dec&&!in_verify){in_verify=1;g_mv_dec=0;float*y2=(float*)malloc(hb);int r=y2&&dsv4_cuda_moe(gate,up,down,weights,count,sg,su,sd,limit,y2,x);g_mv_dec=1;in_verify=0;
        if(r){static int calls,off1,off2,maxulp;static double maxabs;double ma=0;
            for(int i=0;i<H;i++){double dlt=fabs((double)y[i]-y2[i]);if(dlt>ma)ma=dlt;
                /* distance in bf16 ulps (the outputs are bf16-rounded): sign-magnitude codes */
                unsigned a=float_bits_host(y[i]),b2=float_bits_host(y2[i]);int ca=(a>>16)&0x7fff,cb=(b2>>16)&0x7fff;if(a>>31)ca=-ca;if(b2>>31)cb=-cb;
                int u=abs(ca-cb);if(u>maxulp)maxulp=u;if(u>=1)off1++;if(u>=2)off2++;}
            if(ma>maxabs)maxabs=ma;if(++calls%43==0)fprintf(stderr,"moemv verify calls=%d maxabs=%.3g max_ulp=%d elements: 1ulp=%d >=2ulp=%d of %lld\n",calls,maxabs,maxulp,off1-off2,off2,(long long)calls*H);}
        free(y2);}
    return 1;
}
extern "C" int dsv4_cuda_rmsnorm(Dsv4CudaActivation*x,Dsv4CudaTensor*w,float eps,int elements){
    Dev*c=x?ctx(x->device):nullptr;
    if(!c||!w||w->device!=x->device||w->fmt!=32||elements<1||x->elements<elements||
       (long long)w->O*w->I<elements||!ok(cudaSetDevice(x->device),"select RMSNorm device"))return 0;
    rmsnorm_f32<<<1,256,0,c->stream>>>(x->data,x->data,(float*)w->w,elements,eps);
    return ok(cudaGetLastError(),"RMSNorm launch");
}
extern "C" int dsv4_cuda_moe_activation(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,Dsv4CudaTensor *const *down,
        const float *weights,int count,Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd,float limit,
        const Dsv4CudaActivation*input,Dsv4CudaActivation*output){
    if(count<1||count>16||!gate||!up||!down||!weights||!sg||!su||!sd||!input||!output)return 0;
    Dsv4CudaTensor*t=gate[0];Dev*c=t?ctx(t->device):nullptr;
    if(!c||input->device!=t->device||output->device!=t->device||t->fmt!=4||sg->device!=t->device||
       su->device!=t->device||sd->device!=t->device||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||
       !ok(cudaSetDevice(t->device),"select activation MoE device"))return 0;
    int H=t->I,I=t->O;
    if(input->elements<H||output->elements<H||sg->I!=H||sg->O!=I||su->I!=H||su->O!=I||sd->I!=I||sd->O!=H)return 0;
    for(int e=0;e<count;e++)if(!gate[e]||!up[e]||!down[e]||gate[e]->device!=t->device||up[e]->device!=t->device||
       down[e]->device!=t->device||gate[e]->fmt!=4||up[e]->fmt!=4||down[e]->fmt!=4)return 0;
    size_t hb=(size_t)H*4,ib=(size_t)count*I*4,parts=(size_t)count*H*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,parts))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"activation MoE input copy"))return 0;
    fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);
    static int batched=-1;if(batched<0)batched=getenv("DSV4_CUDA_BATCHED")?atoi(getenv("DSV4_CUDA_BATCHED")):1;
    if(batched){
        MvDesc hd[32];for(int e=0;e<count;e++){hd[2*e]={(uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I};hd[2*e+1]={(uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I};}
        if(!buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)2*count*sizeof(*hd))||
           !buf((void**)&c->expert_weights,&c->expert_weightscap,(size_t)count*sizeof(float))||
           !ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)2*count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"activation MoE descriptors upload")||
           !ok(cudaMemcpyAsync(c->expert_weights,weights,(size_t)count*sizeof(float),cudaMemcpyHostToDevice,c->stream),"activation MoE weights upload"))return 0;
        mv_fp4_grouped<4><<<dim3((I+3)/4,2*count),256,0,c->stream>>>(c->mvdesc,2*count,I,H);
        expert_act_grouped<<<(count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,count,I);
        fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
        for(int e=0;e<count;e++)hd[e]={(uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H};
        if(!ok(cudaMemcpyAsync(c->mvdesc,hd,(size_t)count*sizeof(*hd),cudaMemcpyHostToDevice,c->stream),"activation MoE down descriptors upload"))return 0;
        mv_fp4_grouped<4><<<dim3((H+3)/4,count),256,0,c->stream>>>(c->mvdesc,count,H,I);
    }else for(int e=0;e<count;e++){
        run_mv<4>((uint8_t*)gate[e]->w,gate[e]->scale,c->dx,c->p1+(long long)e*I,I,H,1,c->stream);
        run_mv<4>((uint8_t*)up[e]->w,up[e]->scale,c->dx,c->p2+(long long)e*I,I,H,1,c->stream);
        expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3+(long long)e*I,c->p1+(long long)e*I,c->p2+(long long)e*I,weights[e],limit,I);
        fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3+(long long)e*I,1,I);
        run_mv<4>((uint8_t*)down[e]->w,down[e]->scale,c->p3+(long long)e*I,c->p4+(long long)e*H,H,I,1,c->stream);
    }
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,count,H);
    run_mv<8>((uint8_t*)sg->w,sg->scale,c->dx,c->p1,I,H,1,c->stream);
    run_mv<8>((uint8_t*)su->w,su->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);
    fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    run_mv<8>((uint8_t*)sd->w,sd->scale,c->p3,c->p4,H,I,1,c->stream);
    moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
    return ok(cudaGetLastError(),"activation MoE launch");
}
extern "C" Dsv4CudaExpertSet *dsv4_cuda_expert_set_create(Dsv4CudaTensor *const *gate,Dsv4CudaTensor *const *up,
        Dsv4CudaTensor *const *down,int count,Dsv4CudaTensor *sg,Dsv4CudaTensor *su,Dsv4CudaTensor *sd){
    if(!gate||!up||!down||count<1||!sg||!su||!sd||!gate[0])return nullptr;int device=gate[0]->device,H=gate[0]->I,I=gate[0]->O;
    if(sg->device!=device||su->device!=device||sd->device!=device||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||
       sg->I!=H||sg->O!=I||su->I!=H||su->O!=I||sd->I!=I||sd->O!=H||!ok(cudaSetDevice(device),"select expert table device"))return nullptr;
    ExpertPtr *host=(ExpertPtr*)malloc((size_t)3*count*sizeof(*host));if(!host)return nullptr;
    for(int e=0;e<count;e++){
        Dsv4CudaTensor*t[]={gate[e],up[e],down[e]};if(!t[0]||!t[1]||!t[2]||t[0]->device!=device||t[1]->device!=device||t[2]->device!=device||
           t[0]->fmt!=4||t[1]->fmt!=4||t[2]->fmt!=4||t[0]->I!=H||t[0]->O!=I||t[1]->I!=H||t[1]->O!=I||t[2]->I!=I||t[2]->O!=H){free(host);return nullptr;}
        host[e]={(uint8_t*)t[0]->w,t[0]->scale};host[count+e]={(uint8_t*)t[1]->w,t[1]->scale};host[2*count+e]={(uint8_t*)t[2]->w,t[2]->scale};
    }
    Dsv4CudaExpertSet *set=(Dsv4CudaExpertSet*)calloc(1,sizeof(*set));if(!set){free(host);return nullptr;}
    set->count=count;set->device=device;set->H=H;set->I=I;set->sg=sg;set->su=su;set->sd=sd;
    size_t bytes=(size_t)3*count*sizeof(*host);Dev*tc=ctx(device);if(!tc||!ok(cudaMalloc(&set->table,bytes),"expert table allocation")||
       !h2d(tc,set->table,host,bytes,"expert table upload")){free(host);dsv4_cuda_expert_set_free(set);return nullptr;}
    free(host);return set;
}
static Dsv4CudaTensor *expert_view(uint8_t*w,uint8_t*scale,int O,int I,int device){
    Dsv4CudaTensor*t=(Dsv4CudaTensor*)calloc(1,sizeof(*t));if(!t)return nullptr;
    t->w=w;t->scale=scale;t->O=O;t->I=I;t->fmt=4;t->device=device;t->bytes=(long long)O*(I/2)+(long long)O*(I/32);return t;
}
/* Slots the batched-MoE bank needs for a model with `experts` routed experts:
 * the DeepGEMM kernels are compiled for a 256-expert grouping, the generic
 * kernels take the model's own count (smaller bank for pruned checkpoints). */
extern "C" int dsv4_cuda_expert_bank_slots(int experts){
#ifdef COLI_DSV4_DEEPGEMM
    return experts>=1&&experts<=256?256:0;
#else
    return experts>=1&&experts<=256?experts:0;
#endif
}
/* Device bytes per slot of dsv4_cuda_expert_bank_create(count,H,I,...) below
 * (FC1/FC2 fp4 weights and scales, DeepGEMM's scale copies, the pointer
 * table): the engine sizes the VRAM decode leaves free for a bank before one
 * exists. 12.75 MiB generic, 13.5 MiB DeepGEMM at H=4096, I=2048. */
extern "C" long long dsv4_cuda_expert_bank_slot_bytes(int H,int I){
    if(H<1||I<1)return 0;long long fc1_s=2LL*I*(H/32),fc2_s=(long long)H*(I/32);
    long long bytes=(long long)I*H+fc1_s+(long long)H*(I/2)+fc2_s+3*(long long)sizeof(ExpertPtr);
#ifdef COLI_DSV4_DEEPGEMM
    bytes+=fc1_s+fc2_s;
#endif
    return bytes;
}
extern "C" Dsv4CudaExpertSet *dsv4_cuda_expert_bank_create(int count,int H,int I,int device,
        Dsv4CudaTensor*sg,Dsv4CudaTensor*su,Dsv4CudaTensor*sd){
    int shared=sg||su||sd,J=sg?sg->O:0;
    if(count<1||H<1||I<1||(shared&&(!sg||!su||!sd))||(shared&&(sg->device!=device||su->device!=device||sd->device!=device||
       sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->I!=H||su->I!=H||su->O!=J||sd->I!=J||sd->O!=H))||
       !ok(cudaSetDevice(device),"select expert bank device"))return nullptr;
    Dsv4CudaExpertSet*set=(Dsv4CudaExpertSet*)calloc(1,sizeof(*set));if(!set)return nullptr;
    set->count=count;set->device=device;set->H=H;set->I=I;set->sg=sg;set->su=su;set->sd=sd;
    size_t fc1_w=(size_t)count*I*H,fc1_s=(size_t)count*2*I*(H/32);
    size_t fc2_w=(size_t)count*H*(I/2),fc2_s=(size_t)count*H*(I/32);
    if(!ok(cudaMalloc(&set->bank_fc1,fc1_w),"expert FC1 bank allocation")||
       !ok(cudaMalloc(&set->bank_fc1_scale,fc1_s),"expert FC1 scale bank allocation")||
       !ok(cudaMalloc(&set->bank_fc2,fc2_w),"expert FC2 bank allocation")||
       !ok(cudaMalloc(&set->bank_fc2_scale,fc2_s),"expert FC2 scale bank allocation")||
#ifdef COLI_DSV4_DEEPGEMM
       !ok(cudaMalloc(&set->bank_fc1_dg_scale,fc1_s),"expert FC1 DeepGEMM scale allocation")||
       !ok(cudaMalloc(&set->bank_fc2_dg_scale,fc2_s),"expert FC2 DeepGEMM scale allocation")||
#endif
       !ok(cudaMalloc(&set->table,(size_t)3*count*sizeof(ExpertPtr)),"expert table allocation")){
        /* A failed cudaMalloc is also the runtime's last error: clear it, or
         * the next launch check (cudaGetLastError) reports this OOM and fails
         * a chunk that never touched the missing bank. */
        (void)cudaGetLastError();
        dsv4_cuda_expert_set_free(set);return nullptr;
    }
    return set;
}
/* ---- transparent host pinning for expert uploads ----
 * The engine's expert-cache slot slabs are stable for the process lifetime
 * (posix_memalign once per slot, freed only at store destroy), so the first
 * upload from a slab page-locks its whole allocation region (VirtualQuery
 * bounds) and every later upload from it is a pure DMA at PCIe speed instead
 * of a CPU staging copy (~4.7 GB/s measured while the refill's parallel
 * lookups compete for the cores). Failures fall back to pageable copies;
 * DSV4_CUDA_PIN_HOST=0 disables; total pinned bytes are capped. */
struct PinRegion{uintptr_t base;size_t size;};
static std::vector<PinRegion> g_pin_regions,g_pin_failed;
static size_t g_pin_bytes;
static int g_pin_enabled=-1;
static std::mutex g_pin_mutex;   /* the uploader threads and the decode lanes (mirror pieces) all come here */
static bool host_pinned(const void*p,size_t len){
    if(g_pin_enabled<0){const char*e=getenv("DSV4_CUDA_PIN_HOST");g_pin_enabled=!(e&&*e=='0');}
    if(!g_pin_enabled||!p||!len)return false;
    std::lock_guard<std::mutex> pin_lock(g_pin_mutex);
    uintptr_t a=(uintptr_t)p;
    for(auto&r:g_pin_regions)if(a>=r.base&&a+len<=r.base+r.size)return true;
    for(auto&r:g_pin_failed)if(a>=r.base&&a+len<=r.base+r.size)return false;
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if(!VirtualQuery(p,&mbi,sizeof mbi)||mbi.State!=MEM_COMMIT)return false;
    uintptr_t base=(uintptr_t)mbi.BaseAddress;size_t size=mbi.RegionSize;
    if(a<base||a+len>base+size)return false;
    static const size_t cap=(size_t)20<<30;
    if(g_pin_bytes+size>cap){g_pin_failed.push_back({base,size});return false;}
    cudaError_t err=cudaHostRegister((void*)base,size,cudaHostRegisterPortable);
    if(err==cudaErrorHostMemoryAlreadyRegistered){cudaGetLastError();g_pin_regions.push_back({base,size});return true;}
    if(err!=cudaSuccess){cudaGetLastError();g_pin_failed.push_back({base,size});return false;}
    g_pin_bytes+=size;g_pin_regions.push_back({base,size});
    return true;
#else
    /* Linux/WSL: no VirtualQuery, but none is needed — cudaHostRegister pins
     * whole pages, so register the requested range rounded out to page
     * boundaries. The slot slabs are stable for the process lifetime (one
     * posix_memalign per slot, freed only at store destroy), exactly like the
     * Windows branch assumes. Overlapping ranges collapse into the
     * already-registered success case; any failure (WSL builds without
     * pinning support included) lands in g_pin_failed and every later upload
     * from that slab takes the pageable path, same as before this branch
     * existed. */
    long page=sysconf(_SC_PAGESIZE);if(page<=0)page=4096;
    uintptr_t base=a&~((uintptr_t)page-1);
    size_t size=((a+len+page-1)&~((uintptr_t)page-1))-base;
    static const size_t cap=(size_t)20<<30;
    if(g_pin_bytes+size>cap){g_pin_failed.push_back({base,size});return false;}
    cudaError_t err=cudaHostRegister((void*)base,size,cudaHostRegisterPortable);
    if(err==cudaErrorHostMemoryAlreadyRegistered){cudaGetLastError();g_pin_regions.push_back({base,size});return true;}
    if(err!=cudaSuccess){cudaGetLastError();g_pin_failed.push_back({base,size});return false;}
    g_pin_bytes+=size;g_pin_regions.push_back({base,size});
    return true;
#endif
}

/* Drain the device's expert stream: the hybrid decode split enqueues its
 * fill-set uploads asynchronously, computes the CPU subset while the DMA is
 * in flight, and closes the pipeline here before the GPU group runs. */
/* Page-lock a host region now, with nothing in flight on it: the read ->
 * upload pieces pin each slab before its first read (registering a region
 * while the drive still writes into it left stale bytes behind — WDDM moves
 * the pages). 1 when pinned (or already), 0 when not. */
extern "C" int dsv4_cuda_host_pin(const void*p,size_t len){return host_pinned(p,len)?1:0;}
/* V4_PIN_SCRATCH: page-locked host memory for the engine's own staging (a
 * parallel memcpy into it, then a copy at link speed); NULL when refused */
extern "C" void *dsv4_cuda_host_alloc(size_t n){void*p=nullptr;if(!n||!alloc_ok(cudaHostAlloc(&p,n,cudaHostAllocPortable),"pinned scratch"))return nullptr;return p;}
extern "C" void dsv4_cuda_host_free(void*p){if(p)cudaFreeHost(p);}
/* V4_HC_PIN: page-lock a range the engine owns for as long as it says (no
 * registry: the engine unregisters before freeing or reusing it). 1 =
 * registered now, 2 = was already (leave it), 0 = refused. */
extern "C" int dsv4_cuda_host_register(void*p,size_t len){
    if(!p||!len)return 0;
    cudaError_t e=cudaHostRegister(p,len,cudaHostRegisterPortable);
    if(e==cudaErrorHostMemoryAlreadyRegistered){cudaGetLastError();return 2;}
    if(e!=cudaSuccess){cudaGetLastError();return 0;}
    return 1;
}
extern "C" void dsv4_cuda_host_unregister(void*p){if(p&&cudaHostUnregister(p)!=cudaSuccess)cudaGetLastError();}
extern "C" int dsv4_cuda_stream_drain(int device){
    Dev*c=ctx(device);if(!c)return 0;
    if(!ok(cudaSetDevice(device),"select drain device"))return 0;
    return ok(cudaStreamSynchronize(c->aux),"aux drain")&&ok(cudaStreamSynchronize(c->stream),"hybrid drain")?1:0;
}

static int bank_upload_stream(Dsv4CudaExpertSet*set,int e,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds,
        Dsv4CudaTensor**gate,Dsv4CudaTensor**up,Dsv4CudaTensor**down,int use_aux){
    if(!set||e<0||e>=set->count||!gw||!gs||!uw||!us||!dw||!ds||!gate||!up||!down||*gate||*up||*down||
       !ok(cudaSetDevice(set->device),"select expert bank upload device"))return 0;
    int H=set->H,I=set->I;size_t w1=(size_t)I*(H/2),s1=(size_t)I*(H/32),w2=(size_t)H*(I/2),s2=(size_t)H*(I/32);
    uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
    uint8_t*fc2=set->bank_fc2+(size_t)e*w2,*fc2s=set->bank_fc2_scale+(size_t)e*s2;
    Dev*c=ctx(set->device);if(!c)return 0;
    /* use_aux: the double-buffer prefetch worker uploads layer L+1 on the
     * aux stream WHILE the compute stream chews layer L. Everything —
     * copies, DeepGEMM scale packing, the pointer table — rides the chosen
     * stream and drains it before returning, so the caller may release the
     * host slab immediately and the compute stream never stalls. Pageable
     * sources are safe under cudaMemcpyAsync (the copy is staged before it
     * returns with respect to the host buffer). */
    cudaStream_t s=use_aux?c->aux:c->stream;
    if(!ok(cudaMemcpyAsync(fc1,gw,w1,cudaMemcpyHostToDevice,s),"expert gate upload")||
       !ok(cudaMemcpyAsync(fc1+w1,uw,w1,cudaMemcpyHostToDevice,s),"expert up upload")||
       !ok(cudaMemcpyAsync(fc1s,gs,s1,cudaMemcpyHostToDevice,s),"expert gate scale upload")||
       !ok(cudaMemcpyAsync(fc1s+s1,us,s1,cudaMemcpyHostToDevice,s),"expert up scale upload")||
       !ok(cudaMemcpyAsync(fc2,dw,w2,cudaMemcpyHostToDevice,s),"expert down upload")||
       !ok(cudaMemcpyAsync(fc2s,ds,s2,cudaMemcpyHostToDevice,s),"expert down scale upload"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    dg_pack_weight_scale<<<(2*I*(H/32/4)+255)/256,256,0,s>>>(set->bank_fc1_dg_scale,fc1s,e,2*I,H/32);
    dg_pack_weight_scale<<<(H*(I/32/4)+255)/256,256,0,s>>>(set->bank_fc2_dg_scale,fc2s,e,H,I/32);
    if(!ok(cudaGetLastError(),"expert DeepGEMM scale packing"))return 0;
#endif
    Dsv4CudaTensor*g=expert_view(fc1,fc1s,I,H,set->device),*u=expert_view(fc1+w1,fc1s+s1,I,H,set->device),*d=expert_view(fc2,fc2s,H,I,set->device);
    if(!g||!u||!d){free(g);free(u);free(d);return 0;}
    ExpertPtr p[3]={{(uint8_t*)g->w,g->scale},{(uint8_t*)u->w,u->scale},{(uint8_t*)d->w,d->scale}};
    if(!ok(cudaMemcpyAsync(set->table+e,p,sizeof(*p),cudaMemcpyHostToDevice,s),"expert gate table upload")||
       !ok(cudaMemcpyAsync(set->table+set->count+e,p+1,sizeof(*p),cudaMemcpyHostToDevice,s),"expert up table upload")||
       !ok(cudaMemcpyAsync(set->table+2*set->count+e,p+2,sizeof(*p),cudaMemcpyHostToDevice,s),"expert down table upload")||
       !ok(cudaStreamSynchronize(s),"expert upload drain")){
        free(g);free(u);free(d);return 0;
    }
    *gate=g;*up=u;*down=d;return 1;
}

extern "C" int dsv4_cuda_expert_bank_upload(Dsv4CudaExpertSet*set,int e,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds,
        Dsv4CudaTensor**gate,Dsv4CudaTensor**up,Dsv4CudaTensor**down){
    return bank_upload_stream(set,e,gw,gs,uw,us,dw,ds,gate,up,down,0);
}

/* Double-buffer prefetch: same upload, aux stream (overlaps the compute
 * stream), drained before return. */
extern "C" int dsv4_cuda_expert_bank_upload_aux(Dsv4CudaExpertSet*set,int e,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds,
        Dsv4CudaTensor**gate,Dsv4CudaTensor**up,Dsv4CudaTensor**down){
    return bank_upload_stream(set,e,gw,gs,uw,us,dw,ds,gate,up,down,1);
}
extern "C" int dsv4_cuda_expert_bank_upload_tp2(Dsv4CudaExpertSet*set,int e,int rank,
        const uint8_t*gw,const uint8_t*gs,const uint8_t*uw,const uint8_t*us,const uint8_t*dw,const uint8_t*ds){
    if(!set||e<0||e>=set->count||(rank!=0&&rank!=1)||!gw||!gs||!uw||!us||!dw||!ds||
       !ok(cudaSetDevice(set->device),"select TP2 expert bank upload device"))return 0;
    int H=set->H,J=set->I,I=2*J;size_t row_w=H/2,row_s=H/32;
    size_t w1=(size_t)J*row_w,s1=(size_t)J*row_s,w2=(size_t)H*(J/2),s2=(size_t)H*(J/32);
    uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
    uint8_t*fc2=set->bank_fc2+(size_t)e*w2,*fc2s=set->bank_fc2_scale+(size_t)e*s2;
    Dev*tc=ctx(set->device);if(!tc)return 0;cudaStream_t ts=tc->stream;   /* ordered on the compute stream, see h2d */
    if(!ok(cudaMemcpyAsync(fc1,gw+(size_t)rank*w1,w1,cudaMemcpyHostToDevice,ts),"TP2 expert gate upload")||
       !ok(cudaMemcpyAsync(fc1+w1,uw+(size_t)rank*w1,w1,cudaMemcpyHostToDevice,ts),"TP2 expert up upload")||
       !ok(cudaMemcpyAsync(fc1s,gs+(size_t)rank*s1,s1,cudaMemcpyHostToDevice,ts),"TP2 expert gate scale upload")||
       !ok(cudaMemcpyAsync(fc1s+s1,us+(size_t)rank*s1,s1,cudaMemcpyHostToDevice,ts),"TP2 expert up scale upload")||
       !ok(cudaMemcpy2DAsync(fc2,J/2,dw+(size_t)rank*(J/2),I/2,J/2,H,cudaMemcpyHostToDevice,ts),"TP2 expert down upload")||
       !ok(cudaMemcpy2DAsync(fc2s,J/32,ds+(size_t)rank*(J/32),I/32,J/32,H,cudaMemcpyHostToDevice,ts),"TP2 expert down scale upload")||
       !ok(cudaStreamSynchronize(ts),"TP2 expert upload drain"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    dg_pack_weight_scale<<<(2*J*(H/32/4)+255)/256,256>>>(set->bank_fc1_dg_scale,fc1s,e,2*J,H/32);
    dg_pack_weight_scale<<<(H*(J/32/4)+255)/256,256>>>(set->bank_fc2_dg_scale,fc2s,e,H,J/32);
    if(!ok(cudaGetLastError(),"TP2 expert DeepGEMM scale packing"))return 0;
#endif
    return 1;
}
extern "C" void dsv4_cuda_expert_set_free(Dsv4CudaExpertSet*set){if(!set)return;cudaSetDevice(set->device);if(set->table)cudaFree(set->table);if(set->hash)cudaFree(set->hash);if(set->bank_fc1)cudaFree(set->bank_fc1);if(set->bank_fc1_scale)cudaFree(set->bank_fc1_scale);if(set->bank_fc2)cudaFree(set->bank_fc2);if(set->bank_fc2_scale)cudaFree(set->bank_fc2_scale);if(set->bank_fc1_dg_scale)cudaFree(set->bank_fc1_dg_scale);if(set->bank_fc2_dg_scale)cudaFree(set->bank_fc2_dg_scale);free(set);}
/* Rebind the shared-expert mirrors on a bank set. The bank is a per-layer
 * streaming buffer during batched prefill, but the shared experts differ per
 * layer and are already resident as the layer's fp8 mirrors — rebinding them
 * is free, re-creating the ~2 GiB bank per layer is not. Same shape contract
 * as dsv4_cuda_expert_bank_create. */
extern "C" int dsv4_cuda_expert_bank_set_shared(Dsv4CudaExpertSet*set,Dsv4CudaTensor*sg,Dsv4CudaTensor*su,Dsv4CudaTensor*sd){
    if(!set)return 0;
    if(!sg&&!su&&!sd){set->sg=set->su=set->sd=nullptr;return 1;}
    int H=set->H,J=sg?sg->O:0;
    if(!sg||!su||!sd||sg->device!=set->device||su->device!=set->device||sd->device!=set->device||
       sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->I!=H||su->I!=H||su->O!=J||sd->I!=J||sd->O!=H)return 0;
    set->sg=sg;set->su=su;set->sd=sd;return 1;
}
extern "C" int dsv4_cuda_expert_set_upload_hash(Dsv4CudaExpertSet*set,const int64_t*map,int vocab,int topk){
    if(!set||!map||vocab<1||topk!=6||!ok(cudaSetDevice(set->device),"select hash router device"))return 0;
    size_t bytes=(size_t)vocab*topk*sizeof(*map);if(set->hash)cudaFree(set->hash);set->hash=nullptr;set->vocab=0;set->topk=0;
    Dev*hc=ctx(set->device);if(!hc||!ok(cudaMalloc(&set->hash,bytes),"hash router allocation")||!h2d(hc,set->hash,map,bytes,"hash router upload"))return 0;
    set->vocab=vocab;set->topk=topk;return 1;
}
extern "C" int dsv4_cuda_route_moe(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,
        int token,float scale,Dsv4CudaExpertSet*set,float limit,Dsv4CudaActivation*output){
    Dev*c=input?ctx(input->device):nullptr;if(!c||!gate||!set||!output||set->device!=input->device||output->device!=input->device||
       gate->fmt!=32||gate->O!=set->count||gate->I!=set->H||gate->device!=input->device||input->elements<set->H||output->elements<set->H||
       (bias&&(bias->fmt!=32||bias->O*bias->I<set->count||bias->device!=input->device))||set->count!=256||token<0||
       (set->hash&&token>=set->vocab)||
       !ok(cudaSetDevice(input->device),"select routed MoE device"))return 0;
    int H=set->H,I=set->I,K=6;size_t hb=(size_t)H*4,ib=(size_t)K*I*4,parts=(size_t)K*H*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,(size_t)set->count*4)||!buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||
       !buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,parts)||
       !buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)3*K*sizeof(MvDesc))||
       !buf((void**)&c->expert_weights,&c->expert_weightscap,(size_t)K*sizeof(float))||
       !buf((void**)&c->expert_ids,&c->expert_idcap,(size_t)K*sizeof(int)))return 0;
    mv_f32<<<set->count,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,set->count,H);
    if(!ok(cudaMemsetAsync(c->expert_ids,0,(size_t)K*sizeof(int),c->stream),"routed MoE ids clear"))return 0;
    if(set->hash)load_hash_ids<<<1,K,0,c->stream>>>(c->expert_ids,set->hash,c->decode_state,set->topk);
    if(set->hash)route_fixed6<<<1,K,0,c->stream>>>(c->expert_ids,c->expert_weights,c->dy,scale);
    else route_top6<<<1,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,0,scale);
    if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"routed MoE input copy"))return 0;
#ifdef COLI_DSV4_DEEPGEMM
    static int deepgemm=-1;if(deepgemm<0)deepgemm=getenv("DSV4_CUDA_DEEPGEMM")?atoi(getenv("DSV4_CUDA_DEEPGEMM")):1;
    if(deepgemm){
        static int announced;if(!announced++){fprintf(stderr,"[DSV4 CUDA] DeepGEMM routed experts active\n");}
        int ab=getenv("DSV4_CUDA_DG_AB")&&atoi(getenv("DSV4_CUDA_DG_AB"));
        if(ab){
            if(!buf((void**)&c->dgref,&c->dgrefcap,hb))return 0;
            fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);
            build_moe_desc<<<1,K,0,c->stream>>>(c->mvdesc,set->table,c->expert_ids,c->dx,c->p1,c->p2,c->p3,c->p4,K,set->count,H,I);
            run_fp4_grouped(c->mvdesc,2*K,I,H,c->stream);
            expert_act_grouped<<<(K*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,K,I);
            fp8_sim<<<K*((I+127)/128),128,0,c->stream>>>(c->p3,K,I);
            run_fp4_grouped(c->mvdesc+2*K,K,H,I,c->stream);
            expert_reduce<<<(H+255)/256,256,0,c->stream>>>(c->dgref,c->p4,K,H);
            if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"DeepGEMM A/B input restore"))return 0;
        }
        cudaEvent_t dg_begin=nullptr,dg_end=nullptr,shared_end=nullptr;int profile=getenv("DSV4_CUDA_DG_PROFILE")&&atoi(getenv("DSV4_CUDA_DG_PROFILE"));
        if(profile){cudaEventCreate(&dg_begin);cudaEventCreate(&dg_end);cudaEventCreate(&shared_end);cudaEventRecord(dg_begin,c->stream);}
        static int masked=-1;if(masked<0)masked=getenv("DSV4_CUDA_MOE_MASKED")?atoi(getenv("DSV4_CUDA_MOE_MASKED")):0;
        int routed=set->count==256?
            (masked?dg_moe<256>(c,set,c->expert_weights,limit,output->data):dg_moe_contiguous_fast<256>(c,set,c->expert_weights,limit,output->data)):
            set->count==128?(masked?dg_moe<128>(c,set,c->expert_weights,limit,output->data):dg_moe_contiguous_fast<128>(c,set,c->expert_weights,limit,output->data)):0;
        if(!routed)return 0;
        if(!dg_dump_once(c))return 0;
        if(ab){float hs[6];
            dg_compare_fc1<<<1,256,0,c->stream>>>(c->p1,c->p2,c->dgmm1,c->dy);
            dg_compare_fc2<<<1,256,0,c->stream>>>(c->p4,c->dgmm2,c->dy+2);
            dg_compare<<<1,256,0,c->stream>>>(c->dgref,output->data,c->dy+4,H);
            if(!ok(cudaMemcpyAsync(hs,c->dy,sizeof(hs),cudaMemcpyDeviceToHost,c->stream),"DeepGEMM A/B stats")||!ok(cudaStreamSynchronize(c->stream),"DeepGEMM A/B sync"))return 0;
            fprintf(stderr,"[DSV4 DeepGEMM A/B] fc1 max=%.6g mae=%.6g fc2 max=%.6g mae=%.6g routed max=%.6g mae=%.6g\n",hs[0],hs[1],hs[2],hs[3],hs[4],hs[5]);}
        if(profile)cudaEventRecord(dg_end,c->stream);
        int dense_dg=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG"));
        if(dense_dg){if(!dg_dense_2048x4096(c,set->sg,c->dx,c->p1)||!dg_dense_2048x4096(c,set->su,c->dx,c->p2))return 0;}
        else{fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);run_mv<8>((uint8_t*)set->sg->w,set->sg->scale,c->dx,c->p1,I,H,1,c->stream);run_mv<8>((uint8_t*)set->su->w,set->su->scale,c->dx,c->p2,I,H,1,c->stream);}
        expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);
        if(dense_dg){if(!dg_dense_4096x2048(c,set->sd,c->p3,c->p4))return 0;}else{fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);run_mv<8>((uint8_t*)set->sd->w,set->sd->scale,c->p3,c->p4,H,I,1,c->stream);}
        moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
        if(profile){float routed_ms=0,shared_ms=0;cudaEventRecord(shared_end,c->stream);cudaEventSynchronize(shared_end);cudaEventElapsedTime(&routed_ms,dg_begin,dg_end);cudaEventElapsedTime(&shared_ms,dg_end,shared_end);fprintf(stderr,"[DSV4 DeepGEMM] routed=%.3fms shared=%.3fms\n",routed_ms,shared_ms);cudaEventDestroy(dg_begin);cudaEventDestroy(dg_end);cudaEventDestroy(shared_end);}
        return ok(cudaGetLastError(),"DeepGEMM routed and shared MoE launch");
    }
#endif
    fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);
    build_moe_desc<<<1,K,0,c->stream>>>(c->mvdesc,set->table,c->expert_ids,c->dx,c->p1,c->p2,c->p3,c->p4,K,set->count,H,I);
    run_fp4_grouped(c->mvdesc,2*K,I,H,c->stream);
    expert_act_grouped<<<(K*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,K,I);
    fp8_sim<<<K*((I+127)/128),128,0,c->stream>>>(c->p3,K,I);
    run_fp4_grouped(c->mvdesc+2*K,K,H,I,c->stream);
    expert_reduce<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,K,H);
    run_mv<8>((uint8_t*)set->sg->w,set->sg->scale,c->dx,c->p1,I,H,1,c->stream);
    run_mv<8>((uint8_t*)set->su->w,set->su->scale,c->dx,c->p2,I,H,1,c->stream);
    expert_act<<<(I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,I);fp8_sim<<<(I+127)/128,128,0,c->stream>>>(c->p3,1,I);
    run_mv<8>((uint8_t*)set->sd->w,set->sd->scale,c->p3,c->p4,H,I,1,c->stream);moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
    return ok(cudaGetLastError(),"routed MoE launch");
}
extern "C" int dsv4_cuda_route_moe_batch(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,const int*tokens,int count,float scale,Dsv4CudaExpertSet*set,float limit,Dsv4CudaActivation*output){
#ifdef COLI_DSV4_DEEPGEMM
    Dev*c=input?ctx(input->device):nullptr;if(!c||!gate||!set||!output||count<1||count>128||set->device!=input->device||output->device!=input->device||gate->fmt!=32||gate->O!=256||gate->I!=4096||gate->device!=input->device||set->count!=256||set->H!=4096||set->I!=2048||input->elements<(long long)count*4096||output->elements<(long long)count*4096||(bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||(set->hash&&!tokens)||!ok(cudaSetDevice(input->device),"select batched MoE device"))return 0;
    int routes=count*6;size_t logits=(size_t)count*256*sizeof(float),ids=(size_t)routes*sizeof(int),weights=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->dy,&c->ycap,logits)||!buf((void**)&c->expert_ids,&c->expert_idcap,ids)||!buf((void**)&c->expert_weights,&c->expert_weightscap,weights)||
       (set->hash&&!buf((void**)&c->aux3,&c->aux3cap,(size_t)count*sizeof(int))))return 0;
    route_logits_batch<<<count*256,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,count,256,4096);
    if(set->hash){if(!ok(cudaMemcpyAsync(c->aux3,tokens,(size_t)count*sizeof(int),cudaMemcpyHostToDevice,c->stream),"batch token upload"))return 0;route_hash_batch<<<(routes+255)/256,256,0,c->stream>>>(c->expert_ids,set->hash,(int*)c->aux3,count);route_fixed_weights_batch<<<count,6,0,c->stream>>>(c->expert_weights,c->expert_ids,c->dy,count,scale);}
    else route_top6_batch<<<count,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,count,scale);
    /* DSV4_CUDA_MOE_PROF=1: wall-time the three segments (forced syncs, so
     * only for diagnosis) and print the first few calls. */
    static int prof=-1;static int prof_calls;if(prof<0){const char*e=getenv("DSV4_CUDA_MOE_PROF");prof=e&&atoi(e);}
    double t_expert=0,t_shared=0;std::chrono::steady_clock::time_point ta,tb;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);ta=std::chrono::steady_clock::now();}
    if(!dg_moe_batch_contiguous(c,set,input->data,c->expert_ids,c->expert_weights,count,limit,output->data))return 0;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_expert=std::chrono::duration<double>(tb-ta).count();ta=tb;}
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;if(!sg||!su||!sd||sg->O!=2048||su->O!=2048||sd->I!=2048)return 0;
    size_t mid=(size_t)count*2048*sizeof(float),hb=(size_t)count*4096*sizeof(float);if(!buf((void**)&c->p1,&c->p1cap,mid)||!buf((void**)&c->p2,&c->p2cap,mid)||!buf((void**)&c->p3,&c->p3cap,mid)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!dg_dense_batch_dispatch(c,sg,input->data,c->p1,count)||!dg_dense_batch_dispatch(c,su,input->data,c->p2,count))return 0;swiglu_rows<<<((long long)count*2048+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,limit,count,2048);if(!dg_dense_batch_dispatch(c,sd,c->p3,c->p4,count))return 0;add_rows<<<((long long)count*4096+255)/256,256,0,c->stream>>>(output->data,c->p4,count*4096);
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_shared=std::chrono::duration<double>(tb-ta).count();
        fprintf(stderr,"[DSV4 CUDA] moe-batch prof count=%d experts=%.3fs shared=%.3fs\n",count,t_expert,t_shared);prof_calls++;}
    return ok(cudaGetLastError(),"batched MoE launch");
#else
    (void)input;(void)gate;(void)bias;(void)tokens;(void)count;(void)scale;(void)set;(void)limit;(void)output;return 0;
#endif
}
/* Batched top-6 routing only: logits from the f32 gate mirror, ids/weights
 * downloaded to the host so the caller can stage exactly the routed experts
 * before running the expert GEMMs. */
extern "C" int dsv4_cuda_route_top6_batch(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,int count,float scale,int*ids,float*weights){
    Dev*c=input?ctx(input->device):nullptr;
    if(!c||!gate||!ids||!weights||count<1||count>128||gate->fmt!=32||gate->O!=256||gate->I!=4096||gate->device!=input->device||
       input->elements<(long long)count*4096||(bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||
       !ok(cudaSetDevice(input->device),"select batched route device"))return 0;
    int routes=count*6;size_t logits=(size_t)count*256*sizeof(float),ib=(size_t)routes*sizeof(int),wb=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->dy,&c->ycap,logits)||!buf((void**)&c->expert_ids,&c->expert_idcap,ib)||!buf((void**)&c->expert_weights,&c->expert_weightscap,wb))return 0;
    route_logits_batch<<<count*256,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,count,256,4096);
    route_top6_batch<<<count,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,count,scale);
    return ok(cudaGetLastError(),"batched route launch")&&
           ok(cudaMemcpyAsync(ids,c->expert_ids,ib,cudaMemcpyDeviceToHost,c->stream),"batched route id download")&&
           ok(cudaMemcpyAsync(weights,c->expert_weights,wb,cudaMemcpyDeviceToHost,c->stream),"batched route weight download")&&
           ok(cudaStreamSynchronize(c->stream),"batched route sync");
}
/* Batched MoE over host-provided routes (ids/weights per token, 6 each).
 * The caller guarantees every id's bank slot holds that expert's weights;
 * routed sum and the shared expert land in output like route_moe_batch. */
extern "C" int dsv4_cuda_route_moe_ids_batch(const Dsv4CudaActivation*input,const int*ids,const float*weights,int count,Dsv4CudaExpertSet*set,float limit,Dsv4CudaActivation*output){
#ifdef COLI_DSV4_DEEPGEMM
    Dev*c=input?ctx(input->device):nullptr;
    if(!c||!ids||!weights||!set||!output||count<1||count>128||set->device!=input->device||output->device!=input->device||
       set->count!=256||set->H!=4096||set->I!=2048||!set->bank_fc1||
       input->elements<(long long)count*4096||output->elements<(long long)count*4096||
       !ok(cudaSetDevice(input->device),"select batched MoE device"))return 0;
    int routes=count*6;
    for(int r=0;r<routes;r++)if(ids[r]<0||ids[r]>=set->count)return 0;
    size_t ib=(size_t)routes*sizeof(int),wb=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->expert_ids,&c->expert_idcap,ib)||!buf((void**)&c->expert_weights,&c->expert_weightscap,wb))return 0;
    if(!ok(cudaMemcpyAsync(c->expert_ids,ids,ib,cudaMemcpyHostToDevice,c->stream),"batched MoE id upload")||
       !ok(cudaMemcpyAsync(c->expert_weights,weights,wb,cudaMemcpyHostToDevice,c->stream),"batched MoE weight upload"))return 0;
    static int prof=-1;static int prof_calls;if(prof<0){const char*e=getenv("DSV4_CUDA_MOE_PROF");prof=e&&atoi(e);}
    double t_expert=0,t_shared=0;std::chrono::steady_clock::time_point ta,tb;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);ta=std::chrono::steady_clock::now();}
    if(!dg_moe_batch_contiguous(c,set,input->data,c->expert_ids,c->expert_weights,count,limit,output->data,ids))return 0;
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_expert=std::chrono::duration<double>(tb-ta).count();ta=tb;}
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;if(!sg||!su||!sd||sg->O!=2048||su->O!=2048||sd->I!=2048)return 0;
    size_t mid=(size_t)count*2048*sizeof(float),hb=(size_t)count*4096*sizeof(float);
    if(!buf((void**)&c->p1,&c->p1cap,mid)||!buf((void**)&c->p2,&c->p2cap,mid)||!buf((void**)&c->p3,&c->p3cap,mid)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    if(!dg_dense_batch_dispatch(c,sg,input->data,c->p1,count)||!dg_dense_batch_dispatch(c,su,input->data,c->p2,count))return 0;
    swiglu_rows<<<((long long)count*2048+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,limit,count,2048);
    if(!dg_dense_batch_dispatch(c,sd,c->p3,c->p4,count))return 0;
    add_rows<<<((long long)count*4096+255)/256,256,0,c->stream>>>(output->data,c->p4,count*4096);
    if(prof&&prof_calls<6){cudaStreamSynchronize(c->stream);tb=std::chrono::steady_clock::now();t_shared=std::chrono::duration<double>(tb-ta).count();
        fprintf(stderr,"[DSV4 CUDA] moe-ids prof count=%d experts=%.3fs shared=%.3fs\n",count,t_expert,t_shared);prof_calls++;}
    return ok(cudaGetLastError(),"batched MoE ids launch");
#else
    /* GENERIC batched MoE (no DeepGEMM): the decode kernels applied to every
     * token of the chunk in one grouped launch each — fp32 accumulate over
     * fp4-decoded expert rows from the same VRAM bank, then the fp8 shared
     * experts through the generic batch matmul. Slower than the tensor-core
     * path (~10-20 ms per 128-token chunk-layer) but runs on any sm_80+ GPU,
     * so prefill stays GPU-side instead of falling back to the CPU union. */
    Dev*c=input?ctx(input->device):nullptr;
    /* Any bank of up to 256 slots: the grouped kernels index slots by expert
     * id, so expert-pruned checkpoints use a bank sized to their count. */
    if(!c||!ids||!weights||!set||!output||count<1||count>128||set->device!=input->device||output->device!=input->device||
       set->count<1||set->count>256||set->H!=4096||set->I!=2048||!set->bank_fc1||
       input->elements<(long long)count*4096||output->elements<(long long)count*4096||
       !ok(cudaSetDevice(input->device),"select generic batched MoE device"))return 0;
    const int H=4096,I=2048,K=6;int routes=count*K;
    for(int r=0;r<routes;r++)if(ids[r]<0||ids[r]>=set->count)return 0;
    Dsv4CudaTensor*sg=set->sg,*su=set->su,*sd=set->sd;
    if(!sg||!su||!sd||sg->fmt!=8||su->fmt!=8||sd->fmt!=8||sg->O!=I||su->O!=I||sd->I!=I||sg->I!=H||sd->O!=H)return 0;
    size_t w1=(size_t)I*(H/2),s1=(size_t)I*(H/32),w2=(size_t)H*(I/2),s2=(size_t)H*(I/32);
    size_t rb=(size_t)routes*I*sizeof(float),pb=(size_t)routes*H*sizeof(float),wb=(size_t)routes*sizeof(float);
    if(!buf((void**)&c->p1,&c->p1cap,rb)||!buf((void**)&c->p2,&c->p2cap,rb)||!buf((void**)&c->p3,&c->p3cap,rb)||
       !buf((void**)&c->p4,&c->p4cap,pb)||!buf((void**)&c->expert_weights,&c->expert_weightscap,wb))return 0;
    if(!ok(cudaMemcpyAsync(c->expert_weights,weights,wb,cudaMemcpyHostToDevice,c->stream),"generic MoE weight upload"))return 0;
    /* Default: stream each routed expert once per chunk (mv_fp4_expert_major,
     * bit-identical); DSV4_MOE_EXPERT_MAJOR=0 keeps one matvec per route.
     * DSV4_MOE_EXPERT_MAJOR_VERIFY=1 runs both and counts bit mismatches (the
     * per-route results are kept, so it also runs over an explicit =0, which
     * it says once). Routes per pass (TB): 2 while the chunk averages fewer
     * than 4 routes per routed expert (small prompts, tail chunks), else 4.
     * On sm_86 TB=4 took ~1.4x TB=2's time at 1-2 routes per expert and
     * pulled ahead from ~3-4.5 (more skewed routing = later); every TB is
     * bit-identical. DSV4_MOE_EM_TB=2|3|4|6 forces one. */
    static int em_mode=-1,em_mode_env=1,em_verify=0,em_tb=0;static unsigned long long em_calls=0,em_bad1=0,em_bad2=0;
    if(em_mode<0){const char*v=getenv("DSV4_MOE_EXPERT_MAJOR");em_mode_env=v?atoi(v)!=0:1;em_mode=em_mode_env;
        v=getenv("DSV4_MOE_EM_TB");if(v&&*v){em_tb=atoi(v);
            if(em_tb!=2&&em_tb!=3&&em_tb!=4&&em_tb!=6){fprintf(stderr,"[DSV4 CUDA] DSV4_MOE_EM_TB=%s ignored (2, 3, 4 or 6): TB by route density\n",v);em_tb=0;}}
        v=getenv("DSV4_MOE_EXPERT_MAJOR_VERIFY");em_verify=v&&atoi(v)!=0;
        if(em_verify&&!em_mode){em_mode=1;fprintf(stderr,"[DSV4 CUDA] DSV4_MOE_EXPERT_MAJOR_VERIFY=1 overrides DSV4_MOE_EXPERT_MAJOR=0: the expert-major kernel runs beside the per-route one, whose results are kept\n");}}
    int nex=0;const int*em_ex=nullptr,*em_off=nullptr,*em_rt=nullptr;
    if(em_mode){
        int cnt[256]={0},slot_of[256];for(int r=0;r<routes;r++)cnt[ids[r]]++;
        std::vector<int> hm((size_t)set->count+1+set->count+routes);int*hex=hm.data(),*hoff=hex+set->count,*hrt=hoff+set->count+1;
        hoff[0]=0;for(int e=0;e<set->count;e++)if(cnt[e]){slot_of[e]=nex;hex[nex]=e;hoff[nex+1]=hoff[nex]+cnt[e];nex++;}
        std::vector<int> fill(hoff,hoff+nex);for(int r=0;r<routes;r++)hrt[fill[slot_of[ids[r]]]++]=r;
        /* pack as [experts nex][offsets nex+1][routes] */
        std::vector<int> packed((size_t)nex+nex+1+routes);std::copy(hex,hex+nex,packed.begin());std::copy(hoff,hoff+nex+1,packed.begin()+nex);
        std::copy(hrt,hrt+routes,packed.begin()+2*nex+1);
        if(!buf((void**)&c->emaj,&c->emajcap,packed.size()*sizeof(int))||
           !ok(cudaMemcpyAsync(c->emaj,packed.data(),packed.size()*sizeof(int),cudaMemcpyHostToDevice,c->stream),"expert-major route lists"))return 0;
        em_ex=c->emaj;em_off=c->emaj+nex;em_rt=c->emaj+2*nex+1;
        /* verify is a diagnostic: without memory for its buffers it switches
         * itself off for good and DSV4_MOE_EXPERT_MAJOR's own setting applies
         * again (an overridden 0 sends this call down the per-route path) */
        if(em_verify&&(!buf((void**)&c->emv1,&c->emv1cap,rb)||!buf((void**)&c->emv2,&c->emv2cap,rb)||!buf((void**)&c->emv4,&c->emv4cap,pb)||
           (!c->emcount&&!alloc_ok(cudaMalloc(&c->emcount,2*sizeof(unsigned long long)),"expert-major verify counter")))){
            em_verify=0;em_mode=em_mode_env;fprintf(stderr,"[DSV4 CUDA] DSV4_MOE_EXPERT_MAJOR_VERIFY off: no memory for its buffers\n");}
    }
    const int tb=em_tb?em_tb:routes<4*nex?2:4;
    /* the per-route kernel's descriptor tables: per-route mode and verify only */
    const bool per_route=!em_mode||em_verify;std::vector<MvDesc> hd;
    if(per_route){
        if(!buf((void**)&c->mvdesc,&c->mvdesccap,(size_t)routes*2*sizeof(MvDesc)))return 0;
        hd.resize((size_t)routes*2);
        for(int r=0;r<routes;r++){int e=ids[r],t=r/K;
            const uint8_t*fc1=set->bank_fc1+(size_t)e*2*w1,*fc1s=set->bank_fc1_scale+(size_t)e*2*s1;
            const float*x=input->data+(long long)t*H;
            hd[2*r]={fc1,fc1s,x,c->p1+(long long)r*I};
            hd[2*r+1]={fc1+w1,fc1s+s1,x,c->p2+(long long)r*I};}
        if(!ok(cudaMemcpyAsync(c->mvdesc,hd.data(),(size_t)routes*2*sizeof(MvDesc),cudaMemcpyHostToDevice,c->stream),"generic MoE fc1 descriptors"))return 0;
        mv_fp4_grouped<4><<<dim3((I+3)/4,routes*2),256,0,c->stream>>>(c->mvdesc,routes*2,I,H);
    }
    if(em_mode){
        float*y0=em_verify?c->emv1:c->p1,*y1=em_verify?c->emv2:c->p2;
        run_fp4_expert_major(tb,set->bank_fc1,set->bank_fc1_scale,2*w1,2*s1,
            em_ex,em_off,em_rt,nex,input->data,H,K,y0,y1,I,I,2*I,H,c->stream);
        if(em_verify){
            if(!ok(cudaMemsetAsync(c->emcount,0,2*sizeof(unsigned long long),c->stream),"verify clear"))return 0;
            count_bit_mismatch<<<((long long)routes*I+255)/256,256,0,c->stream>>>(c->p1,c->emv1,(long long)routes*I,c->emcount);
            count_bit_mismatch<<<((long long)routes*I+255)/256,256,0,c->stream>>>(c->p2,c->emv2,(long long)routes*I,c->emcount);
        }
    }
    expert_act_grouped<<<((long long)routes*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,c->expert_weights,limit,routes,I);
    fp8_sim<<<routes*((I+127)/128),128,0,c->stream>>>(c->p3,routes,I);
    if(per_route){
        /* down: descriptors must not be overwritten while fc1 still reads them —
         * the copy is stream-ordered after the launches above, so reuse is safe. */
        for(int r=0;r<routes;r++){int e=ids[r];
            hd[r]={set->bank_fc2+(size_t)e*w2,set->bank_fc2_scale+(size_t)e*s2,c->p3+(long long)r*I,c->p4+(long long)r*H};}
        if(!ok(cudaMemcpyAsync(c->mvdesc,hd.data(),(size_t)routes*sizeof(MvDesc),cudaMemcpyHostToDevice,c->stream),"generic MoE down descriptors"))return 0;
        mv_fp4_grouped<4><<<dim3((H+3)/4,routes),256,0,c->stream>>>(c->mvdesc,routes,H,I);
    }
    if(em_mode){
        float*y=em_verify?c->emv4:c->p4;
        run_fp4_expert_major(tb,set->bank_fc2,set->bank_fc2_scale,w2,s2,
            em_ex,em_off,em_rt,nex,c->p3,I,1,y,y,H,H,H,I,c->stream);
        if(em_verify){
            count_bit_mismatch<<<((long long)routes*H+255)/256,256,0,c->stream>>>(c->p4,c->emv4,(long long)routes*H,c->emcount+1);
            unsigned long long bad[2]={0,0};
            if(!ok(cudaMemcpyAsync(bad,c->emcount,sizeof(bad),cudaMemcpyDeviceToHost,c->stream),"verify read")||
               !ok(cudaStreamSynchronize(c->stream),"verify sync"))return 0;
            em_calls++;em_bad1+=bad[0];em_bad2+=bad[1];
            if(bad[0]||bad[1]||em_calls%100==0)
                fprintf(stderr,"moe-expert-major verify calls=%llu fc1_mismatch=%llu fc2_mismatch=%llu (this call %llu/%llu of %lld/%lld)\n",
                        em_calls,em_bad1,em_bad2,bad[0],bad[1],(long long)routes*I*2,(long long)routes*H);
        }
    }
    expert_reduce_rows<<<((long long)count*H+255)/256,256,0,c->stream>>>(output->data,c->p4,count,K,H);
    /* shared experts (fp8 dense) over the whole chunk, then the same combine
     * as decode: routed = bf16(routed + bf16(shared)). */
    size_t mid=(size_t)count*I*sizeof(float),hb=(size_t)count*H*sizeof(float);
    if(!buf((void**)&c->p1,&c->p1cap,mid)||!buf((void**)&c->p2,&c->p2cap,mid)||!buf((void**)&c->p3,&c->p3cap,mid)||!buf((void**)&c->p4,&c->p4cap,hb))return 0;
    run_mm_batch(sg,input->data,c->p1,count,c->stream);run_mm_batch(su,input->data,c->p2,count,c->stream);
    expert_act_rows1<<<((long long)count*I+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,limit,count*I);
    fp8_sim<<<count*((I+127)/128),128,0,c->stream>>>(c->p3,count,I);
    run_mm_batch(sd,c->p3,c->p4,count,c->stream);
    moe_combine<<<((long long)count*H+255)/256,256,0,c->stream>>>(output->data,c->p4,count*H);
    return ok(cudaGetLastError(),"generic batched MoE launch");
#endif
}
extern "C" int dsv4_cuda_route_moe_ep2(const Dsv4CudaActivation*input,Dsv4CudaTensor*gate,Dsv4CudaTensor*bias,
        const Dsv4CudaActivation*peer_input,Dsv4CudaTensor*peer_gate,Dsv4CudaTensor*peer_bias,int token,float scale,
        Dsv4CudaExpertSet*local,Dsv4CudaExpertSet*peer,float limit,Dsv4CudaActivation*output,Dsv4CudaActivation*peer_output){
#ifndef COLI_DSV4_DEEPGEMM
    (void)input;(void)gate;(void)bias;(void)peer_input;(void)peer_gate;(void)peer_bias;(void)token;(void)scale;(void)local;(void)peer;(void)limit;(void)output;(void)peer_output;return 0;
#else
    Dev*c=input?ctx(input->device):nullptr,*p=peer?ctx(peer->device):nullptr;
    int replicated=getenv("DSV4_CUDA_REPLICATED_TP2")&&atoi(getenv("DSV4_CUDA_REPLICATED_TP2"));
    int moe_tp2=getenv("DSV4_CUDA_MOE_TP2")&&atoi(getenv("DSV4_CUDA_MOE_TP2"));
    if(!c||!p||!gate||!local||!peer||!output||!peer_output||
       (moe_tp2?(!replicated||local->count!=256||peer->count!=256||local->I!=1024):
                (local->count!=128||peer->count!=128))||local->H!=peer->H||local->I!=peer->I||
       gate->fmt!=32||gate->O!=256||gate->I!=local->H||gate->device!=input->device||output->device!=input->device||peer_output->device!=peer->device||
       (bias&&(bias->fmt!=32||bias->O*bias->I<256||bias->device!=input->device))||
       (replicated&&(!peer_input||!peer_gate||peer_input->device!=peer->device||peer_gate->device!=peer->device||peer_gate->fmt!=32||peer_gate->O!=256||peer_gate->I!=local->H||
       (peer_bias&&(peer_bias->device!=peer->device||peer_bias->fmt!=32||peer_bias->O*peer_bias->I<256))))||token<0||(local->hash&&token>=local->vocab))return 0;
    int H=local->H,I=local->I,K=6;size_t hb=(size_t)H*4,ib=(size_t)K*I*4;
    if(!ok(cudaSetDevice(c->id),"select EP2 primary")||!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->dy,&c->ycap,256*4)||
       !buf((void**)&c->p1,&c->p1cap,ib)||!buf((void**)&c->p2,&c->p2cap,ib)||!buf((void**)&c->p3,&c->p3cap,ib)||!buf((void**)&c->p4,&c->p4cap,hb)||
       !buf((void**)&c->expert_ids,&c->expert_idcap,K*sizeof(int))||!buf((void**)&c->expert_weights,&c->expert_weightscap,K*sizeof(float))||
       !buf((void**)&c->aux1,&c->aux1cap,hb))return 0;
    mv_f32<<<256,256,0,c->stream>>>((float*)gate->w,input->data,c->dy,256,H);
    if(!ok(cudaMemsetAsync(c->expert_ids,0,K*sizeof(int),c->stream),"EP2 ids clear"))return 0;
    if(local->hash)load_hash_ids<<<1,K,0,c->stream>>>(c->expert_ids,local->hash,c->decode_state,local->topk);
    if(local->hash)route_fixed6<<<1,K,0,c->stream>>>(c->expert_ids,c->expert_weights,c->dy,scale);
    else route_top6<<<1,32,0,c->stream>>>(c->expert_ids,c->expert_weights,bias?(float*)bias->w:nullptr,c->dy,0,scale);
    if(!ok(cudaMemcpyAsync(c->dx,input->data,hb,cudaMemcpyDeviceToDevice,c->stream),"EP2 primary input")||
       !ok(cudaSetDevice(p->id),"select EP2 peer")||!buf((void**)&p->dx,&p->xcap,hb)||!buf((void**)&p->dy,&p->ycap,256*4)||
       !buf((void**)&p->expert_ids,&p->expert_idcap,K*sizeof(int))||!buf((void**)&p->expert_weights,&p->expert_weightscap,K*sizeof(float))||
       !buf((void**)&p->p1,&p->p1cap,(size_t)I*4)||!buf((void**)&p->p2,&p->p2cap,(size_t)I*4)||
       !buf((void**)&p->p3,&p->p3cap,(size_t)I*4)||!buf((void**)&p->p4,&p->p4cap,hb))return 0;
    if(replicated){mv_f32<<<256,256,0,p->stream>>>((float*)peer_gate->w,peer_input->data,p->dy,256,H);
        if(!ok(cudaMemsetAsync(p->expert_ids,0,K*sizeof(int),p->stream),"EP2 peer ids clear"))return 0;
        if(peer->hash)load_hash_ids<<<1,K,0,p->stream>>>(p->expert_ids,peer->hash,p->decode_state,peer->topk);
        if(peer->hash)route_fixed6<<<1,K,0,p->stream>>>(p->expert_ids,p->expert_weights,p->dy,scale);
        else route_top6<<<1,32,0,p->stream>>>(p->expert_ids,p->expert_weights,peer_bias?(float*)peer_bias->w:nullptr,p->dy,0,scale);
        if(!ok(cudaMemcpyAsync(p->dx,peer_input->data,hb,cudaMemcpyDeviceToDevice,p->stream),"EP2 peer local input"))return 0;}
#ifdef COLI_DSV4_NCCL
    if(!replicated&&(!c->ep_comm||!p->ep_comm||!ok(cudaSetDevice(c->id),"select EP2 broadcast root")||!nok(ncclGroupStart(),"EP2 broadcast group start")||
       !nok(ncclBroadcast(c->dx,c->dx,H,ncclFloat,0,c->ep_comm,c->stream),"EP2 input root")||
       !nok(ncclBroadcast(c->expert_ids,c->expert_ids,K,ncclInt,0,c->ep_comm,c->stream),"EP2 ids root")||
       !nok(ncclBroadcast(c->expert_weights,c->expert_weights,K,ncclFloat,0,c->ep_comm,c->stream),"EP2 weights root")||
       !ok(cudaSetDevice(p->id),"select EP2 broadcast peer")||
       !nok(ncclBroadcast(p->dx,p->dx,H,ncclFloat,0,p->ep_comm,p->stream),"EP2 input peer")||
       !nok(ncclBroadcast(p->expert_ids,p->expert_ids,K,ncclInt,0,p->ep_comm,p->stream),"EP2 ids peer")||
       !nok(ncclBroadcast(p->expert_weights,p->expert_weights,K,ncclFloat,0,p->ep_comm,p->stream),"EP2 weights peer")||
       !nok(ncclGroupEnd(),"EP2 broadcast group end")))return 0;
#else
    if(!replicated&&(!ok(cudaEventRecord(c->ready,c->stream),"EP2 route ready")||!ok(cudaStreamWaitEvent(p->stream,c->ready,0),"EP2 peer route wait")||
       !ok(cudaMemcpyPeerAsync(p->dx,p->id,c->dx,c->id,hb,p->stream),"EP2 input transfer")||
       !ok(cudaMemcpyPeerAsync(p->expert_ids,p->id,c->expert_ids,c->id,K*sizeof(int),p->stream),"EP2 ids transfer")||
       !ok(cudaMemcpyPeerAsync(p->expert_weights,p->id,c->expert_weights,c->id,K*sizeof(float),p->stream),"EP2 weights transfer")))return 0;
#endif
    int contiguous=getenv("DSV4_CUDA_MOE_CONTIGUOUS")&&atoi(getenv("DSV4_CUDA_MOE_CONTIGUOUS"));
    if(moe_tp2){sort_routes6<<<1,1,0,p->stream>>>(p->expert_ids,p->expert_weights);if(!dg_moe_tp2(p,peer,p->expert_weights,limit,peer_output->data)){fprintf(stderr,"[DSV4 TP2] peer routed failed\n");return 0;}}
    else{ep2_localize<<<1,6,0,p->stream>>>(p->expert_ids,p->expert_weights,128);
        if(!(contiguous?dg_moe_contiguous_fast<128>(p,peer,p->expert_weights,limit,peer_output->data,0):dg_moe<128>(p,peer,p->expert_weights,limit,peer_output->data))){fprintf(stderr,"[DSV4 EP2] peer routed failed\n");return 0;}}
    int shared_tp=peer->sg!=nullptr;
    if(shared_tp){int PJ=peer->sg->O;if(!dg_attention_projection(p,peer->sg,p->dx,p->p1)||
       !dg_attention_projection(p,peer->su,p->dx,p->p2)){fprintf(stderr,"[DSV4 EP2] peer shared up failed\n");return 0;}
        expert_act<<<(PJ+255)/256,256,0,p->stream>>>(p->p3,p->p1,p->p2,1.f,limit,PJ);
        if(!dg_attention_projection(p,peer->sd,p->p3,p->p4)){fprintf(stderr,"[DSV4 EP2] peer shared down failed\n");return 0;}
        moe_combine<<<(H+255)/256,256,0,p->stream>>>(peer_output->data,p->p4,H);}
    if(!ok(cudaSetDevice(c->id),"restore EP2 primary"))return 0;
    if(moe_tp2){sort_routes6<<<1,1,0,c->stream>>>(c->expert_ids,c->expert_weights);if(!dg_moe_tp2(c,local,c->expert_weights,limit,output->data)){fprintf(stderr,"[DSV4 TP2] primary routed failed\n");return 0;}}
    else{ep2_localize<<<1,6,0,c->stream>>>(c->expert_ids,c->expert_weights,0);
        if(!(contiguous?dg_moe_contiguous_fast<128>(c,local,c->expert_weights,limit,output->data,0):dg_moe<128>(c,local,c->expert_weights,limit,output->data))){fprintf(stderr,"[DSV4 EP2] primary routed failed\n");return 0;}}
    int dense=getenv("DSV4_CUDA_DENSE_DG")&&atoi(getenv("DSV4_CUDA_DENSE_DG")),LJ=local->sg?local->sg->O:0;
    if(shared_tp){if(!LJ||!dg_attention_projection(c,local->sg,c->dx,c->p1)||!dg_attention_projection(c,local->su,c->dx,c->p2)){fprintf(stderr,"[DSV4 EP2] primary shared up failed\n");return 0;}}
    else if(dense){if(!dg_dense_2048x4096(c,local->sg,c->dx,c->p1)||!dg_dense_2048x4096(c,local->su,c->dx,c->p2))return 0;}
    else{fp8_sim<<<(H+127)/128,128,0,c->stream>>>(c->dx,1,H);run_mv<8>((uint8_t*)local->sg->w,local->sg->scale,c->dx,c->p1,LJ,H,1,c->stream);run_mv<8>((uint8_t*)local->su->w,local->su->scale,c->dx,c->p2,LJ,H,1,c->stream);}
    expert_act<<<(LJ+255)/256,256,0,c->stream>>>(c->p3,c->p1,c->p2,1.f,limit,LJ);
    if(shared_tp){if(!dg_attention_projection(c,local->sd,c->p3,c->p4)){fprintf(stderr,"[DSV4 EP2] primary shared down failed\n");return 0;}}
    else if(dense){if(!dg_dense_4096x2048(c,local->sd,c->p3,c->p4))return 0;}
    else{fp8_sim<<<(LJ+127)/128,128,0,c->stream>>>(c->p3,1,LJ);run_mv<8>((uint8_t*)local->sd->w,local->sd->scale,c->p3,c->p4,H,LJ,1,c->stream);}
    if(shared_tp)moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
#ifdef COLI_DSV4_NCCL
    if(!ok(cudaSetDevice(c->id),"select EP2 reduce root")||!nok(ncclGroupStart(),"EP2 reduce group start")||
       !nok(ncclAllReduce(output->data,output->data,H,ncclFloat,ncclSum,c->ep_comm,c->stream),"EP2 primary reduce")||
       !ok(cudaSetDevice(p->id),"select EP2 reduce peer")||
       !nok(ncclAllReduce(peer_output->data,peer_output->data,H,ncclFloat,ncclSum,p->ep_comm,p->stream),"EP2 peer reduce")||
       !nok(ncclGroupEnd(),"EP2 reduce group end"))return 0;
    if(!ok(cudaSetDevice(c->id),"restore EP2 combine device"))return 0;
    if(!shared_tp)moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
#else
    if(!ok(cudaEventRecord(p->ready,p->stream),"EP2 peer compute ready")||!ok(cudaStreamWaitEvent(c->stream,p->ready,0),"EP2 peer compute wait")||
       !ok(cudaMemcpyPeerAsync(c->aux1,c->id,peer_output->data,p->id,hb,c->stream),"EP2 result transfer"))return 0;
    add_peer<<<(H+255)/256,256,0,c->stream>>>(output->data,c->aux1,H);
    if(!shared_tp)moe_combine<<<(H+255)/256,256,0,c->stream>>>(output->data,c->p4,H);
#endif
    return ok(cudaGetLastError(),"EP2 routed MoE launch");
#endif
}

extern "C" int dsv4_cuda_qkv(Dsv4CudaTensor *qa,Dsv4CudaTensor *qn,Dsv4CudaTensor *qb,Dsv4CudaTensor *kv,float eps,float *qo,float *ko,const float *x){
    Dev*c=qa?ctx(qa->device):nullptr;if(!c||!qn||!qb||!kv||qa->fmt!=8||qn->fmt!=16||qb->fmt!=8||kv->fmt!=8||
       qn->device!=qa->device||qb->device!=qa->device||kv->device!=qa->device||!ok(cudaSetDevice(qa->device),"select qkv device"))return 0;
    int H=qa->I,R=qa->O,Q=qb->O,K=kv->O;if(qb->I!=R||kv->I!=H||qn->I!=R)return 0;
    size_t hb=(size_t)H*4,rb=(size_t)R*4,qbbytes=(size_t)Q*4,kb=(size_t)K*4;
    if(!buf((void**)&c->dx,&c->xcap,hb)||!buf((void**)&c->p1,&c->p1cap,rb)||
       !buf((void**)&c->p2,&c->p2cap,qbbytes)||!buf((void**)&c->p3,&c->p3cap,kb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,x,hb,cudaMemcpyHostToDevice,c->stream),"qkv activation upload"))return 0;
    run_mv<8>((uint8_t*)qa->w,qa->scale,c->dx,c->p1,R,H,1,c->stream);bf16_round<<<(R+255)/256,256,0,c->stream>>>(c->p1,R);
    rmsnorm_seq<<<1,256,0,c->stream>>>(c->p1,(__nv_bfloat16*)qn->w,R,eps);fp8_sim<<<(R+127)/128,128,0,c->stream>>>(c->p1,1,R);
    run_mv<8>((uint8_t*)qb->w,qb->scale,c->p1,c->p2,Q,R,1,c->stream);bf16_round<<<(Q+255)/256,256,0,c->stream>>>(c->p2,Q);
    run_mv<8>((uint8_t*)kv->w,kv->scale,c->dx,c->p3,K,H,1,c->stream);bf16_round<<<(K+255)/256,256,0,c->stream>>>(c->p3,K);
    return ok(cudaGetLastError(),"qkv launch")&&ok(cudaMemcpyAsync(qo,c->p2,qbbytes,cudaMemcpyDeviceToHost,c->stream),"q download")&&
           ok(cudaMemcpyAsync(ko,c->p3,kb,cudaMemcpyDeviceToHost,c->stream),"kv download")&&ok(cudaStreamSynchronize(c->stream),"qkv sync");
}
extern "C" int dsv4_cuda_wo(Dsv4CudaTensor *wa,Dsv4CudaTensor *wb,int groups,float *out,const float *context){
    Dev*c=wa?ctx(wa->device):nullptr;if(!c||!wb||wa->fmt!=9||wb->fmt!=8||groups<1||wa->O%groups||
       wb->device!=wa->device||wb->I!=wa->O||!ok(cudaSetDevice(wa->device),"select wo device"))return 0;
    int C=wa->I*groups,R=wa->O,H=wb->O;size_t cb=(size_t)C*4,rb=(size_t)R*4,hb=(size_t)H*4;
    if(!buf((void**)&c->dx,&c->xcap,cb)||!buf((void**)&c->p1,&c->p1cap,rb)||!buf((void**)&c->p2,&c->p2cap,hb))return 0;
    if(!ok(cudaMemcpyAsync(c->dx,context,cb,cudaMemcpyHostToDevice,c->stream),"wo context upload"))return 0;
    run_mv<9>((uint8_t*)wa->w,wa->scale,c->dx,c->p1,R,wa->I,groups,c->stream);bf16_round<<<(R+255)/256,256,0,c->stream>>>(c->p1,R);
    fp8_sim<<<(R+127)/128,128,0,c->stream>>>(c->p1,1,R);run_mv<8>((uint8_t*)wb->w,wb->scale,c->p1,c->p2,H,R,1,c->stream);
    bf16_round<<<(H+255)/256,256,0,c->stream>>>(c->p2,H);
    return ok(cudaGetLastError(),"wo launch")&&ok(cudaMemcpyAsync(out,c->p2,hb,cudaMemcpyDeviceToHost,c->stream),"wo result download")&&ok(cudaStreamSynchronize(c->stream),"wo sync");
}
extern "C" void dsv4_cuda_tensor_free(Dsv4CudaTensor*t){if(!t)return;cudaSetDevice(t->device);if(t->ev)cudaEventDestroy(t->ev);if(t->block)cudaFree(t->block);if(t->block2)cudaFree(t->block2);if(t->own_w&&t->w)cudaFree(t->w);if(t->bf16_cache)cudaFree(t->bf16_cache);if(t->own_scale&&t->scale)cudaFree(t->scale);if(t->own_dg_scale&&t->dg_scale)cudaFree(t->dg_scale);free(t);}
extern "C" long long dsv4_cuda_tensor_bytes(const Dsv4CudaTensor*t){return t?t->bytes:0;}
extern "C" int dsv4_cuda_tensor_device(const Dsv4CudaTensor*t){return t?t->device:-1;}
