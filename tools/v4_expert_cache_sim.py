"""Offline expert-cache policy simulation over V4_EXPERT_TRACE lines.
usage: python cache_sim.py <err file> [warm_tokens]"""
import sys, re, collections
path = sys.argv[1]; warm = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tokens = []  # list of list of (layer, expert, vram, ram)
cur = []; last_layer = -1
for line in open(path, 'r', errors='replace'):
    if not line.startswith('xtrace '): continue
    m = re.match(r'xtrace layer=(\d+) ids=(.*)', line.strip())
    layer = int(m.group(1))
    if layer <= last_layer and cur: tokens.append(cur); cur = []
    last_layer = layer
    for item in m.group(2).split(','):
        if not item: continue
        e, st = item.split(':'); cur.append((layer, int(e), int(st[0]), int(st[1])))
if cur: tokens.append(cur)
T = len(tokens); req = sum(len(t) for t in tokens)
print(f'tokens={T} requests={req} per_token={req/max(T,1):.1f}')
meas = [sum(x[2] for x in t) / len(t) for t in tokens]
print(f'measured vram hit: all={sum(meas)/T:.3f} last_half={sum(meas[T//2:])/(T-T//2):.3f}; ram-cached among vram misses: '
      f'{sum(1 for t in tokens for x in t if not x[2] and x[3]) / max(1, sum(1 for t in tokens for x in t if not x[2])):.3f}')
keys = [[(l, e) for (l, e, v, r) in t] for t in tokens]
distinct = len(set(k for t in keys for k in t)); print(f'distinct experts touched={distinct}')
# overlap with previous token
ov = [len(set(keys[i]) & set(keys[i-1])) / len(keys[i]) for i in range(1, T)]
print(f'token-to-token overlap={sum(ov)/len(ov):.3f}')

def lru(cap, victim=0):
    cache = collections.OrderedDict(); ram = collections.OrderedDict()
    hits = miss = ramhit = 0
    for ti, t in enumerate(keys):
        for k in t:
            if k in cache:
                cache.move_to_end(k)
                if ti >= warm: hits += 1
            else:
                if ti >= warm:
                    miss += 1
                    if k in ram: ramhit += 1
                if k in ram: del ram[k]
                cache[k] = 1
                if len(cache) > cap:
                    old, _ = cache.popitem(last=False)
                    if victim:
                        ram[old] = 1
                        if len(ram) > victim: ram.popitem(last=False)
    return hits / max(1, hits + miss), ramhit / max(1, miss)

def lfu_decay(cap, half_every=64):
    cache = {}; count = collections.Counter(); hits = miss = 0; n = 0
    for ti, t in enumerate(keys):
        for k in t:
            n += 1
            if n % (half_every * 258) == 0:
                for kk in list(count): count[kk] = count[kk] / 2
            count[k] += 1
            if k in cache:
                if ti >= warm: hits += 1
            else:
                if ti >= warm: miss += 1
                if len(cache) >= cap:
                    victim = min(cache, key=lambda kk: (count[kk], cache[kk]))
                    del cache[victim]
                cache[k] = n
    return hits / max(1, hits + miss)

def static_top(cap, fit_tokens):
    c = collections.Counter(k for t in keys[:fit_tokens] for k in t)
    hot = set(k for k, _ in c.most_common(cap))
    tot = hit = 0
    for t in keys[fit_tokens:]:
        for k in t:
            tot += 1; hit += k in hot
    return hit / max(1, tot)

for cap in (688, 900, 1200, 1600, 2400):
    h, r = lru(cap, victim=700)
    print(f'LRU cap={cap}: hit={h:.3f}  (+RAM victim 700: {r:.3f} of misses)  LFU-decay: {lfu_decay(cap):.3f}')
half = T // 2
for cap in (688, 1200):
    print(f'static top-{cap} fitted on first {half} tokens, scored on the rest: {static_top(cap, half):.3f}')

def two_cards(cap1, hot_n, fit_tokens):
    c = collections.Counter(k for t in keys[:fit_tokens] for k in t)
    hot = set(k for k, _ in c.most_common(hot_n))
    cache = collections.OrderedDict(); h1 = h2 = miss = 0
    for ti, t in enumerate(keys):
        for k in t:
            if k in hot:
                if ti >= fit_tokens: h2 += 1
                continue
            if k in cache:
                cache.move_to_end(k)
                if ti >= fit_tokens: h1 += 1
            else:
                if ti >= fit_tokens: miss += 1
                cache[k] = 1
                if len(cache) > cap1: cache.popitem(last=False)
    tot = max(1, h1 + h2 + miss)
    return h1 / tot, h2 / tot, miss / tot
for hot_n in (500, 700):
    p1, p2, m = two_cards(688, hot_n, half)
    print(f'primary LRU 688 + static hot {hot_n} on card 2 (fitted first half): primary={p1:.3f} card2={p2:.3f} miss={m:.3f}')
