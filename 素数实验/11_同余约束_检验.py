"""
测试 A: 同余约束零模型 (Congruence-Constrained Null Models)
────────────────────────────────────────────────────────
核心追问:
  你的 η=899 到底有多少来自"非随机尾数分布"这个已知事实？
  如果把尾数约束还给零模型，效应还剩多少？

约束层级 (逐级收紧, 从"不公平"到"公平"):
  L0: 无约束    — 完全随机置放, 连奇数位都没保留      ← 原来的零模型
  L1: mod 2    — 只放奇数位 (素数>2 全是奇数)        ← 最小公平性修正
  L2: mod 3    — 按模3余数分层 (保留 mod3 分布)
  L3: mod 5    — 按模5余数分层  
  L4: mod 10   — 按个位 1/3/7/9 分层 (最强约束)      ← 终极公平对照

判决逻辑:
  如果 S(真素数) ≫ S(L4约束) → 素数结构远超同余约束
  如果 S(真素数) ≈ S(L4约束) → 你发现的就是尾数分布, 前人早知道了
"""

import math, random, statistics, time
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

interval_size = N // K
ideal_counts = []
for i in range(K):
    a = i * interval_size + 1
    b = (i + 1) * interval_size
    ideal_counts.append(li_approx(b + 1) - li_approx(a))

def compute_S_from_set(prime_set):
    SP_vals = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        obs = sum(1 for x in range(a, b + 1) if x in prime_set)
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
S_real, spm_r, dsp_r = compute_S_from_set(real_primes)
total = len(real_primes)

print(f"真实素数: N={N}, 总数={total}, S={S_real:.6f}")
print()

# ── 构建分层索引 ──
all_positions = list(range(2, N+1))

def build_strata(positions, mod_func):
    strata = {}
    for x in positions:
        r = mod_func(x)
        strata.setdefault(r, []).append(x)
    return strata

mod2_strata  = build_strata(all_positions, lambda x: x % 2)
mod3_strata  = build_strata(all_positions, lambda x: x % 3)
mod5_strata  = build_strata(all_positions, lambda x: x % 5)
mod10_strata = build_strata(all_positions, lambda x: x % 10)

def count_in_strata(prime_set, strata):
    counts = {}
    for r, positions in strata.items():
        counts[r] = sum(1 for p in prime_set if p in positions)
    return counts

real_mod2  = count_in_strata(real_primes, mod2_strata)
real_mod3  = count_in_strata(real_primes, mod3_strata)
real_mod5  = count_in_strata(real_primes, mod5_strata)
real_mod10 = count_in_strata(real_primes, mod10_strata)

print("── 素数在各模下的分布 ──")
print(f"  mod 2:  {sorted(real_mod2.items())}  ← 除2外全在奇数位")
print(f"  mod 3:  {sorted(real_mod3.items())}  ← 3本身是唯一余0的")
print(f"  mod 5:  {sorted(real_mod5.items())}  ← 5本身是唯一余0的")
print(f"  mod 10: {sorted(real_mod10.items())}  ← 只有1,3,7,9 (及{2}和{5})")
print()

# ── 抽样函数 ──
def sample_constrained(strata, real_counts):
    """从各层按真实计数随机抽取，生成伪素数集"""
    fake = set()
    for r, positions in strata.items():
        n_needed = real_counts.get(r, 0)
        if n_needed > 0 and len(positions) >= n_needed:
            fake.update(random.sample(positions, n_needed))
    return fake

M = 2000
models = {}
print(f"每个约束层级抽样 {M} 次...\n")

def run_model(name, strata, real_counts):
    start = time.time()
    vals = []
    for _ in range(M):
        fake = sample_constrained(strata, real_counts)
        s, _, _ = compute_S_from_set(fake)
        vals.append(s)
    mn, sd = statistics.mean(vals), statistics.stdev(vals)
    eta = (S_real - mn) / sd if sd > 0 else float('inf')
    pct = sum(1 for v in vals if v >= S_real) / M * 100
    models[name] = (vals, mn, sd)
    print(f"  {name:>30s}  S={mn:.6f} ± {sd:.6f}  "
          f"距真素数 {eta:.1f}σ  ≥S_real: {pct:.1f}%  "
          f"({time.time()-start:.1f}s)")

# L0: 无约束 — 特殊处理: 从全位置随机抽 total 个
start = time.time()
l0_vals = []
all_pos = list(range(2, N+1))
for _ in range(M):
    fake = set(random.sample(all_pos, total))
    s, _, _ = compute_S_from_set(fake)
    l0_vals.append(s)
mn_l0, sd_l0 = statistics.mean(l0_vals), statistics.stdev(l0_vals)
models["L0 无约束 (全随机)"] = (l0_vals, mn_l0, sd_l0)
print(f"  {'L0 无约束 (全随机)':>30s}  S={mn_l0:.6f} ± {sd_l0:.6f}  "
      f"距真素数 {(S_real-mn_l0)/sd_l0:.1f}σ  ≥S_real: {sum(1 for v in l0_vals if v>=S_real)/M*100:.1f}%  "
      f"({time.time()-start:.1f}s)")

# L1: mod 2
run_model("L1 mod2约束 (只放奇数位)", mod2_strata, real_mod2)

# L2: mod 3  
run_model("L2 mod3约束 (保留mod3分布)", mod3_strata, real_mod3)

# L3: mod 5
run_model("L3 mod5约束 (保留mod5分布)", mod5_strata, real_mod5)

# L4: mod 10
run_model("L4 mod10约束 (保留尾数分布)", mod10_strata, real_mod10)

print(f"\n{'='*75}")
print(f"  S(真素数) = {S_real:.6f}")
print(f"{'='*75}")
print(f"  {'层级':<30s} {'S均值':>8s} {'± 标准差':>10s} {'距真(σ)':>8s} {'≥S_real':>7s}")
print(f"  {'-'*65}")
for name, (vals, mn, sd) in models.items():
    eta = (S_real - mn) / sd if sd > 0 else float('inf')
    pct = sum(1 for v in vals if v >= S_real) / M * 100
    print(f"  {name:<30s} {mn:8.6f} {sd:10.6f} {eta:8.1f} {pct:6.1f}%")

# ── 关键判决 ──
l4_name = "L4 mod10约束 (保留尾数分布)"
l0_name = "L0 无约束 (全随机)"
l4_mn, l4_sd = models[l4_name][1], models[l4_name][2]
l0_mn, l0_sd = models[l0_name][1], models[l0_name][2]
effect_from_constraints = l4_mn - l0_mn
effect_remaining = S_real - l4_mn
l4_eta = (S_real - l4_mn) / l4_sd

print(f"\n{'='*75}")
print(f"  判决")
print(f"{'='*75}")
print(f"  无约束零模型 S:           {l0_mn:.6f}  (基线)")
print(f"  +mod10约束后 S:           {l4_mn:.6f}  (+{effect_from_constraints:.6f})")
print(f"  尾数约束贡献:             {effect_from_constraints:.6f}  "
      f"({effect_from_constraints/(S_real-l0_mn)*100:.0f}% of gap)")
print(f"  剩余差距 (远超尾数):      {effect_remaining:.6f}  "
      f"({effect_remaining/(S_real-l0_mn)*100:.0f}% of gap)")
print(f"  距 mod10 约束:            {l4_eta:.1f}σ")
print()

if l4_eta >= 5:
    print(f"  ★ S(真素数) 比最强的同余约束零模型高出 {l4_eta:.1f}σ。")
    print(f"    同余约束仅解释了 {(S_real-l0_mn-effect_remaining)/(S_real-l0_mn)*100:.0f}% 的效应。")
    print(f"    S(N) 捕捉到的结构远在已知的尾数分布之上。")
elif l4_eta >= 3:
    print(f"  S(真素数) 依然显著高于最强约束零模型 ({l4_eta:.1f}σ)。")
    print(f"    同余约束解释了部分效应，但远非全部。")
else:
    print(f"  ⚠ S(真素数) 与最强约束零模型差距仅 {l4_eta:.1f}σ。")
    print(f"    效应可能主要由已知的尾数分布驱动。需要更大 N 验证。")

# ── 图 ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
colors = ['#95a5a6', '#e67e22', '#3498db', '#2ecc71', '#e74c3c']

for ax, (name, color) in zip(axes[:5], zip(
    [l0_name, "L1 mod2约束 (只放奇数位)", "L2 mod3约束 (保留mod3分布)",
     "L3 mod5约束 (保留mod5分布)", l4_name], colors)):
    vals, mn, sd = models[name]
    ax.hist(vals, bins=50, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(mn, color='black', linewidth=1.5, linestyle=':', label=f'mean={mn:.4f}')
    ax.axvline(S_real, color='red', linewidth=2, label=f'primes={S_real:.4f}')
    ax.set_xlabel('S')
    ax.set_ylabel('freq')
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7)

# 第六张：汇总对比
ax = axes[5]
names_short = ['L0\n无约束', 'L1\nmod2', 'L2\nmod3', 'L3\nmod5', 'L4\nmod10']
xs = range(len(names_short))
means = [models[n][1] for n in models]
sds   = [models[n][2] for n in models]
bars = ax.bar(xs, means, yerr=sds, color=colors, edgecolor='white', capsize=5)
ax.axhline(S_real, color='red', linewidth=2, linestyle='--', label=f'S(primes)={S_real:.4f}')
ax.set_xticks(xs)
ax.set_xticklabels(names_short, fontsize=8)
ax.set_ylabel('S')
ax.set_title('Constraint Hierarchy', fontsize=10)
ax.legend(fontsize=8)
for bar, mn in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{mn:.4f}', ha='center', fontsize=7)

fig.suptitle(f'Congruence-Constrained Null Models  |  N={N}, K={K}, M={M}\n'
             f'S(primes)={S_real:.4f}  —  {l4_eta:.1f}σ above strongest congruence constraint',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_congruence_constraints.png', dpi=150)
plt.close()
print("\n图已保存: figures/11_congruence_constraints.png")
