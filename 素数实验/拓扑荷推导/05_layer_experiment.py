import math, statistics, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 10_000
K = 50
M = 300

SQRT_N = int(N**0.5)
INTERVAL_SIZE = N // K

def primes_upto(limit):
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if is_p[p]:
            step = p
            start = p * p
            is_p[start:limit+1:step] = b'\x00' * ((limit - start) // step + 1)
    return [i for i in range(2, limit + 1) if is_p[i]]

ALL_PRIMES = primes_upto(N)
TARGET = len(ALL_PRIMES)
REAL_PRIMES_SET = set(ALL_PRIMES)

seed_primes_all = [p for p in ALL_PRIMES if p <= SQRT_N]

LAYER_SEEDS = {
    "L1": seed_primes_all[:1],
    "L2": seed_primes_all[:5],
    "L3": seed_primes_all[:15],
    "L4": seed_primes_all,
}

def li_approx(x):
    if x <= 2: return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))

def li_density(x):
    if x <= 2: return 0.0
    return 1.0 / math.log(x)

ideal_counts = [li_approx(b+1) - li_approx(a+1)
                for a, b in [(i*INTERVAL_SIZE, (i+1)*INTERVAL_SIZE-1)
                             for i in range(K)]]

def sieve_with_seeds(seeds):
    is_comp = bytearray(N + 1)
    for s in seeds:
        start = s * 2
        if start <= N:
            n_marks = (N - start) // s + 1
            is_comp[start:N+1:s] = b'\x01' * n_marks
    return {i for i in range(2, N + 1) if is_comp[i] == 0}

def compute_S(point_set):
    n_total = len(point_set)
    if n_total == 0:
        return 0.0
    scale = n_total / TARGET
    SP_vals = []
    for i in range(K):
        a = i * INTERVAL_SIZE + 1
        b = (i + 1) * INTERVAL_SIZE
        obs = sum(1 for x in range(a, b + 1) if x in point_set)
        v_star = ideal_counts[i] * scale
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        SP_vals.append(math.exp(-(obs - v_star)**2 / (2 * sigma**2)))
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if K > 1 else 0.0
    return SP_mean / (1 + delta_SP)

S_POISSON_THEORY = 1.0 / math.sqrt(2)

print(f"参数: N={N}  K={K}  M={M}  sqrt(N)={SQRT_N}")
print(f"筛种子素数: {len(seed_primes_all)} 个 = {seed_primes_all}")
print()

results_det = {}

for label, seeds in [("L1", LAYER_SEEDS["L1"]),
                      ("L2", LAYER_SEEDS["L2"]),
                      ("L3", LAYER_SEEDS["L3"]),
                      ("L4", LAYER_SEEDS["L4"])]:
    start = time.time()
    pts = sieve_with_seeds(seeds)
    count = len(pts)
    S_val = compute_S(pts)
    seed_str = "{" + ",".join(str(s) for s in seeds[:6]) + (
        ",..." if len(seeds) > 6 else "") + "}"
    print(f"  {label}: seeds={seed_str} (n={len(seeds)})"
          f"  count={count}  S={S_val:.6f}  ({time.time()-start:.3f}s)")
    results_det[label] = {"S": S_val, "count": count, "seeds": seeds}

print()
print(f"  真素数 (L4 = 全筛验证): "
      f"count={TARGET}, expected={TARGET}, "
      f"match={'OK' if count == TARGET else 'MISMATCH!'}")
print()

print("=" * 65)
print("L0: 纯密度 Poisson (M={} 次采样)".format(M))
print("=" * 65)

import random
random.seed(42)

def gen_L0(c):
    return {x for x in range(2, N + 1)
            if random.random() < c * li_density(x)}

def tune_L0():
    lo, hi = 0.1, 3.0
    for _ in range(15):
        mid = (lo + hi) / 2
        avg = sum(len(gen_L0(mid)) for _ in range(30)) / 30
        if avg > TARGET:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

c0 = tune_L0()

start = time.time()
S_L0_vals = []
cnt_L0 = []
for _ in range(M):
    pts = gen_L0(c0)
    cnt_L0.append(len(pts))
    S_L0_vals.append(compute_S(pts))

mu0 = statistics.mean(S_L0_vals)
sd0 = statistics.stdev(S_L0_vals)
avg0 = statistics.mean(cnt_L0)
print(f"  c={c0:.4f}  avg_count={avg0:.0f}  S={mu0:.6f} ± {sd0:.6f}"
      f"  ({time.time()-start:.1f}s)")

L0_S = mu0
L1_S = results_det["L1"]["S"]
L2_S = results_det["L2"]["S"]
L3_S = results_det["L3"]["S"]
L4_S = results_det["L4"]["S"]

all_S = [L0_S, L1_S, L2_S, L3_S, L4_S]
all_counts = [avg0, results_det["L1"]["count"], results_det["L2"]["count"],
              results_det["L3"]["count"], results_det["L4"]["count"]]
all_n_seeds = [0, 1, 5, 15, 25]
all_labels = ["L0\nNo Sieve", "L1\n{2}", "L2\n{2,3,5,7,11}",
              "L3\n15 primes", "L4\n25 primes"]

total_gap = L4_S - L0_S

print()
print("=" * 72)
print("  乘法结构逐层引入 — 确定性筛种子递增")
print("=" * 72)
print(f"  {'层级':<36s} {'S':>8s}  {'count':>6s}  {'n_seeds':>7s}  {'Δ from L0':>10s}")
print(f"  {'─'*72}")
for i, (label, s, cnt, ns) in enumerate(zip(
    ["L0: 纯密度 Poisson (无筛, 纯噪声)",
     "L1: 筛去偶数 ({2} 单种子)",
     "L2: 前 5 素数筛 {2,3,5,7,11}",
     "L3: 前 15 素数筛 {2,...,47}",
     "L4: 全筛 (25 素数, 即真素数)"],
    all_S, all_counts, all_n_seeds)):
    delta = s - L0_S if i > 0 else 0
    print(f"  {label:<36s} {s:>8.4f}  {cnt:>5.0f}  {ns:>7d}  {delta:>+10.4f}")
print("=" * 72)

print()
print(f"  纯 Poisson 理论基线:  {S_POISSON_THEORY:.4f}")
print(f"  L0 实测 (纯噪声):     {L0_S:.4f}")
print(f"  L1 ({2}):             {L1_S:.4f}")
print(f"  L2 (5 素数):          {L2_S:.4f}")
print(f"  L3 (15 素数):         {L3_S:.4f}")
print(f"  L4 (25 素数, 真素数): {L4_S:.4f}")
print(f"  总增长 (L4 - L0):     {total_gap:+.4f}")
print()

deltas = []
for i in range(1, len(all_S)):
    d = all_S[i] - all_S[i-1]
    if d > 0:
        deltas.append((all_labels[i], d, all_S[i] - L0_S))

print("  各层边际贡献 (增量):")
for label, d, cumulative in deltas:
    pct = d / total_gap * 100 if total_gap > 0 else 0
    print(f"    {label:<20s}  +{d:.4f}  ({pct:.0f}% of total gap)")

remaining = L4_S - L3_S
if total_gap > 0:
    print(f"\n  最终一跳 (L3→L4):     +{remaining:.4f}  "
          f"({remaining/total_gap*100:.0f}% of total gap)")
    print(f"  前 15 素数已解释:     {L3_S-L0_S:+.4f}  "
          f"({(L3_S-L0_S)/total_gap*100:.0f}% of total gap)")

plt.figure(figsize=(13, 5.5))
xs = range(len(all_labels))
colors = ['#95a5a6', '#e67e22', '#3498db', '#2ecc71', '#e74c3c']

plt.bar(xs, all_S, color=colors, edgecolor='white', width=0.6)
if sd0:
    plt.errorbar(0, L0_S, yerr=sd0, fmt='none', ecolor='black', capsize=5)

for x, s in zip(xs, all_S):
    plt.text(x, s + 0.012, f'{s:.4f}', ha='center', fontsize=9, fontweight='bold')

for i in range(1, len(all_S)):
    delta = all_S[i] - all_S[i-1]
    if abs(delta) > 0.0005:
        mid = all_S[i-1] + delta / 2
        plt.annotate(f'{delta:+.4f}', xy=(i-0.42, mid), fontsize=7,
                     color='#2c3e50',
                     bbox=dict(boxstyle='round,pad=0.15',
                               facecolor='#ecf0f1', alpha=0.85))

plt.xticks(xs, all_labels, fontsize=8)
plt.ylabel('S (with scaled v*)')
plt.axhline(y=S_POISSON_THEORY, color='gray', linestyle='--', linewidth=0.8,
            label=f'Poisson theory: {S_POISSON_THEORY:.4f}')
plt.legend(fontsize=8)
plt.title(f'Multiplicative Structure: Deterministic Sieve with Increasing Primes\n'
          f'N={N}  K={K}  |  '
          f'L0={L0_S:.3f}  ->  L4={L4_S:.3f}  (total +{total_gap:+.3f})',
          fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('../figures/13_layer_by_layer_v4.png', dpi=150)
plt.close()
print()
print("图已保存: ../figures/13_layer_by_layer_v4.png")
