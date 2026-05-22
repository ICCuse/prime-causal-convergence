import math
import random
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scale_N = [10000, 20000, 50000, 100000]
K_base = 50
M = 200
random.seed(42)

max_N = max(scale_N)
print(f"正在筛法生成素数表，最大范围到 {max_N}...")
t0 = time.time()

is_prime = [True] * (max_N + 1)
is_prime[0] = is_prime[1] = False
for p in range(2, int(max_N**0.5) + 1):
    if is_prime[p]:
        for multiple in range(p * p, max_N + 1, p):
            is_prime[multiple] = False

print(f"筛法完成，耗时 {time.time() - t0:.1f} 秒")


def li(x):
    if x <= 2:
        return 0.0
    steps = max(2000, int((x - 2) * 20))
    dt = (x - 2) / steps
    integral = 0.0
    for i in range(steps):
        t1 = 2 + i * dt
        t2 = t1 + dt
        integral += (1.0 / math.log(t1) + 1.0 / math.log(t2)) * dt / 2
    return integral


def compute_S_for_range(N, K):
    interval_size = N // K
    ideal_counts = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        expected = li(b + 1) - li(a)
        ideal_counts.append(expected)
    SP_vals = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        obs = sum(1 for x in range(a, b + 1) if is_prime[x])
        v_star = ideal_counts[i]
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        diff = obs - v_star
        SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
        SP_vals.append(SP_i)
    SP_mean = statistics.mean(SP_vals)
    if K > 1:
        delta_SP = statistics.variance(SP_vals)
    else:
        delta_SP = 0.0
    S = SP_mean / (1 + delta_SP)
    return S, SP_mean, delta_SP


def perturb_and_compute_S(N, K, total_primes, M_samples):
    S_list = []
    interval_size = N // K
    ideal_counts = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        expected = li(b + 1) - li(a)
        ideal_counts.append(expected)
    for _ in range(M_samples):
        fake_positions = random.sample(range(2, N + 1), total_primes)
        fake_bools = [False] * (N + 1)
        for pos in fake_positions:
            fake_bools[pos] = True
        SP_vals = []
        for i in range(K):
            a = i * interval_size + 1
            b = (i + 1) * interval_size
            obs = sum(1 for x in range(a, b + 1) if fake_bools[x])
            v_star = ideal_counts[i]
            sigma = math.sqrt(v_star) if v_star > 0 else 1.0
            diff = obs - v_star
            SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
            SP_vals.append(SP_i)
        SP_mean = statistics.mean(SP_vals)
        delta_SP = statistics.variance(SP_vals) if K > 1 else 0.0
        S = SP_mean / (1 + delta_SP)
        S_list.append(S)
    return S_list


print("\n========== 跨尺度检验 ==========")
print(f"{'N':<8} {'K':<6} {'总素数':<8} {'S_real':<10} {'S_扰动均值':<12} {'S_扰动std':<12} {'η':<8} {'p值':<8}")
print("-" * 80)

results = []

for N in scale_N:
    K = K_base * (N // 10000)
    prime_list = [i for i in range(2, N + 1) if is_prime[i]]
    total_primes = len(prime_list)
    t_start = time.time()
    S_real, SP_mean, delta_SP = compute_S_for_range(N, K)
    S_perturbed = perturb_and_compute_S(N, K, total_primes, M)
    mean_pert = statistics.mean(S_perturbed)
    std_pert = statistics.stdev(S_perturbed)
    count_higher = sum(1 for s in S_perturbed if s >= S_real)
    p_val = (count_higher + 1) / (M + 1)
    eta = (S_real - mean_pert) / std_pert if std_pert > 0 else float('inf')
    elapsed = time.time() - t_start
    print(f"{N:<8} {K:<6} {total_primes:<8} {S_real:<10.6f} {mean_pert:<12.6f} {std_pert:<12.6f} {eta:<8.2f} {p_val:<8.4f}")
    results.append({
        'N': N,
        'K': K,
        'total_primes': total_primes,
        'S_real': S_real,
        'SP_mean': SP_mean,
        'delta_SP': delta_SP,
        'S_pert_mean': mean_pert,
        'S_pert_std': std_pert,
        'eta': eta,
        'p_value': p_val
    })

print("\n========== 跨尺度趋势分析 ==========")
print(f"{'N':<8} {'S_real':<10} {'SP_mean':<10} {'delta_SP':<10}")
for r in results:
    print(f"{r['N']:<8} {r['S_real']:<10.6f} {r['SP_mean']:<10.6f} {r['delta_SP']:<10.6f}")

print("\n--- 尺度行为观察 ---")
S_vals = [r['S_real'] for r in results]
N_vals = [r['N'] for r in results]
for i in range(1, len(S_vals)):
    delta_S = S_vals[i] - S_vals[i - 1]
    ratio_N = N_vals[i] / N_vals[i - 1]
    print(f"N: {N_vals[i-1]} → {N_vals[i]} (x{ratio_N}), S: {S_vals[i-1]:.6f} → {S_vals[i]:.6f}, ΔS = {delta_S:+.6f}")

N_list = [r['N'] for r in results]
S_list = [r['S_real'] for r in results]
S_pert_mean = [r['S_pert_mean'] for r in results]
S_pert_std = [r['S_pert_std'] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.errorbar(N_list, S_list, yerr=S_pert_std, fmt='o-', color='darkblue',
             capsize=5, capthick=1.5, markersize=8, linewidth=2, label='S(N)')
ax1.fill_between(N_list, [m - s for m, s in zip(S_pert_mean, S_pert_std)],
                 [m + s for m, s in zip(S_pert_mean, S_pert_std)],
                 alpha=0.2, color='grey', label='Null ±1σ')
ax1.set_xscale('log')
ax1.set_xlabel('N (log scale)')
ax1.set_ylabel('S')
ax1.set_title('S(N) Convergence Trend')
ax1.legend()
ax1.grid(True, alpha=0.3)

eta_vals = [r['eta'] for r in results]
ax2.bar(range(len(eta_vals)), eta_vals, color='coral', edgecolor='darkred', alpha=0.8)
ax2.set_xticks(range(len(eta_vals)))
ax2.set_xticklabels([f'{n//1000}K' for n in N_list], rotation=45)
ax2.set_ylabel('Effect Size η (σ)')
ax2.set_title('Statistical Significance by Scale')
ax2.axhline(y=5, color='grey', linestyle='--', alpha=0.5, label='5σ')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('03_cross_scale_S.png', dpi=150)
plt.close()
print("图已保存: 03_cross_scale_S.png")
