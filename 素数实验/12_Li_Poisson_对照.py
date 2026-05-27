"""
测试 C: Li(x) 导引 Poisson 零模型 (最纯粹的密度对照)
────────────────────────────────────────────────────────
核心追问:
  如果素数的唯一结构就是"越来越稀"（由素数定理描述），
  那么 Li(x) 导引的独立随机序列应该和真素数有相同的 S。
  如果 S(真素数) ≫ S(Li-Poisson) → 素数的结构远超密度趋势之外。

这是一把双刃剑:
  近距离 → 支持你的发现：S 捕捉到了真正的微观序
  接近   → 否定了你的发现：S 只测到了密度衰减，前人早就知道

多层对照:
  A: Li'(x) Poisson  — 密度 = 1/log(x)，最理论纯净的对照
  B: 经验密度 Poisson — 密度 = 真实素数在 [x-Δ,x+Δ] 内的局部密度
  C: 固定个数 Li(x)抽样 — 保证总数 = 真素数总数，按 Li(x) 权重不放回抽样
  D: 真素数 (基准)
"""

import math, random, statistics, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)
N = 10_000
K = 50

def li_approx(x):
    if x <= 2: return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))

def li_derivative(x):
    if x <= 2: return 0.0
    return 1.0 / math.log(x)

interval_size = N // K
ideal_counts = []
for i in range(K):
    a = i * interval_size + 1
    b = (i + 1) * interval_size
    ideal_counts.append(li_approx(b + 1) - li_approx(a))

def compute_S_from_count_fn(count_fn):
    SP_vals = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        obs = count_fn(a, b)
        v_star = ideal_counts[i]
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        SP_vals.append(math.exp(-(obs - v_star)**2 / (2 * sigma**2)))
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if K > 1 else 0.0
    return SP_mean / (1 + delta_SP), SP_mean, delta_SP

# ── 真实素数 ──
is_p = bytearray(b'\x01') * (N + 1)
is_p[0] = is_p[1] = 0
for p in range(2, int(N**0.5) + 1):
    if is_p[p]:
        is_p[p*p:N+1:p] = bytearray((N-p*p)//p + 1)
real_primes = {i for i in range(2, N+1) if is_p[i]}

def real_count(a, b):
    return sum(1 for x in range(a, b+1) if x in real_primes)

S_real, spm_r, dsp_r = compute_S_from_count_fn(real_count)
total_primes = len(real_primes)

print(f"真实素数: N={N}, 总数={total_primes}, S={S_real:.6f}")
print()

# ── 计算真实密度曲线 (滑动窗口平滑) ──
window = 200
empirical_density = {}
for x in range(2, N+1):
    lo, hi = max(2, x - window//2), min(N, x + window//2)
    count = sum(1 for i in range(lo, hi+1) if i in real_primes)
    empirical_density[x] = count / (hi - lo + 1)

M = 2000
models = {}
print(f"每个零模型抽样 {M} 次...\n")

# ═══════════════════════════════════════════════════════════
# A: Li'(x) Poisson — 纯理论密度, 最干净
# ═══════════════════════════════════════════════════════════
start = time.time()
li_poisson_S = []
for _ in range(M):
    def gen_li_poisson():
        bits = bytearray(N+1)
        for x in range(2, N+1):
            if random.random() < li_derivative(x):
                bits[x] = 1
        def count(a, b):
            return sum(bits[a:b+1])
        return count
    s, _, _ = compute_S_from_count_fn(gen_li_poisson())
    li_poisson_S.append(s)
mn_lp, sd_lp = statistics.mean(li_poisson_S), statistics.stdev(li_poisson_S)
models['A: Li(x) Poisson\n(理论密度)'] = (li_poisson_S, mn_lp, sd_lp)
print(f"  Li(x) Poisson:          S={mn_lp:.6f} ± {sd_lp:.6f}  ({time.time()-start:.1f}s)")

# ═══════════════════════════════════════════════════════════
# B: 经验密度 Poisson — 保留真实素数局部密度
# ═══════════════════════════════════════════════════════════
start = time.time()
emp_poisson_S = []
for _ in range(M):
    def gen_emp_poisson():
        bits = bytearray(N+1)
        for x in range(2, N+1):
            if random.random() < empirical_density[x]:
                bits[x] = 1
        def count(a, b):
            return sum(bits[a:b+1])
        return count
    s, _, _ = compute_S_from_count_fn(gen_emp_poisson())
    emp_poisson_S.append(s)
mn_ep, sd_ep = statistics.mean(emp_poisson_S), statistics.stdev(emp_poisson_S)
models['B: 经验密度 Poisson\n(局部密度)'] = (emp_poisson_S, mn_ep, sd_ep)
print(f"  经验密度 Poisson:        S={mn_ep:.6f} ± {sd_ep:.6f}  ({time.time()-start:.1f}s)")

# ═══════════════════════════════════════════════════════════
# C: 权重不放回抽样 (numpy 加速)
# ═══════════════════════════════════════════════════════════
start = time.time()
positions = np.array(range(2, N+1))
weights  = np.array([li_derivative(x) for x in positions])
weighted_S = []
for _ in range(M):
    probs = weights / weights.sum()
    chosen = set(np.random.choice(positions, size=total_primes, replace=False, p=probs))
    def count(a, b):
        return sum(1 for x in range(a, b+1) if x in chosen)
    s, _, _ = compute_S_from_count_fn(count)
    weighted_S.append(s)

mn_w, sd_w = statistics.mean(weighted_S), statistics.stdev(weighted_S)
models['C: Li(x)权重\n不放回抽样'] = (weighted_S, mn_w, sd_w)
print(f"  Li(x)权重不放回:         S={mn_w:.6f} ± {sd_w:.6f}  ({time.time()-start:.1f}s)")

# ═══════════════════════════════════════════════════════════
# D: 固定个数随机置放 (原始零模型, 做基线)
# ═══════════════════════════════════════════════════════════
start = time.time()
all_pos = list(range(2, N+1))
shuffle_S = []
for _ in range(M):
    fake = set(random.sample(all_pos, total_primes))
    def count(a, b):
        return sum(1 for x in range(a, b+1) if x in fake)
    s, _, _ = compute_S_from_count_fn(count)
    shuffle_S.append(s)
mn_s, sd_s = statistics.mean(shuffle_S), statistics.stdev(shuffle_S)
models['D: 固定个数\n随机置放 (原)'] = (shuffle_S, mn_s, sd_s)
print(f"  固定个数随机置放:        S={mn_s:.6f} ± {sd_s:.6f}  ({time.time()-start:.1f}s)")

# ═══════════════════════════════════════════════════════════
# 判决
# ═══════════════════════════════════════════════════════════
print(f"\n{'='*75}")
print(f"  S(真素数) = {S_real:.6f}")
print(f"{'='*75}")
print(f"  {'模型':<35s} {'S均值':>8s} {'± sd':>8s} {'距真(σ)':>8s} {'≥S':>5s}")
print(f"  {'-'*65}")

best_mn = 0
best_name = ""
for name, (vals, mn, sd) in models.items():
    eta = (S_real - mn) / sd if sd > 0 else float('inf')
    pct = sum(1 for v in vals if v >= S_real) / M * 100
    stars = "★" if pct == 0 else ""
    print(f"  {name:<35s} {mn:8.6f} {sd:8.6f} {eta:8.1f} {pct:5.1f}% {stars}")
    if mn > best_mn:
        best_mn = mn
        best_name = name

gap_to_best = S_real - best_mn
print(f"\n{'='*75}")
print(f"  判决")
print(f"{'='*75}")
print(f"  最强零模型: {best_name}  S={best_mn:.6f}")
print(f"  真素数:     S={S_real:.6f}")
print(f"  差距:       {gap_to_best:.6f}")
print(f"  效应比:     真素数/{best_mn*100 if best_mn > 0 else 0:.0f} = {S_real/best_mn:.2f}x" if best_mn > 0 else "")
print(f"  总体假阳性: 所有 {M*len(models)} 次零模型抽样中 ≥S_real 的次数 = "
      f"{sum(1 for vals,_,_ in models.values() for v in vals if v >= S_real)}")
print()

lp_eta = (S_real - mn_lp) / sd_lp if sd_lp > 0 else float('inf')
if lp_eta >= 5:
    print(f"  ★ Li(x)-Poisson 零模型仅给出 S={mn_lp:.4f}。")
    print(f"    真素数 S={S_real:.4f} 高出 {lp_eta:.1f}σ。")
    print(f"    Li(x)/素数定理描述的越来越稀趋势无法解释")
    print(f"    S(N) 在素数中检测到的结构。")
    print(f"    素数内部存在 Li(x) 之外的微观序。")
else:
    print(f"  需要更大样本或不同参数验证。")

# ── 图 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

for ax, (name, color) in zip(axes, zip(models.keys(), colors)):
    vals, mn, sd = models[name]
    ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(mn, color='black', linewidth=1.5, linestyle=':', label=f'mean={mn:.4f}')
    ax.axvline(S_real, color='red', linewidth=2, label=f'primes={S_real:.4f}')
    ax.set_xlabel('S')
    ax.set_ylabel('frequency')
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)

fig.suptitle(f'Li(x)-Poisson Null Models  |  N={N}, K={K}, M={M}\n'
             f'S(primes)={S_real:.4f}  —  {lp_eta:.1f}σ above Li(x) baseline',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/12_lix_poisson.png', dpi=150)
plt.close()
print("\n图已保存: figures/12_lix_poisson.png")
