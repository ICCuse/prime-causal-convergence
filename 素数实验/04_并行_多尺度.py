import math
import random
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_max = 5_000_000
E_target = 50
M_perturb = 30
random.seed(42)

scale_points = [10_000, 20_000, 50_000,
                100_000, 200_000, 500_000,
                1_000_000, 2_000_000, 5_000_000]

print(f"全尺度扫描：{len(scale_points)} 个尺度点，最大 N = {N_max:,}")
print(f"E = {E_target}, M = {M_perturb}")

print(f"正在生成素数表...")
t0 = time.time()

is_prime = [True] * (N_max + 1)
is_prime[0] = is_prime[1] = False
for p in range(2, int(N_max**0.5) + 1):
    if is_prime[p]:
        is_prime[p * p: N_max + 1: p] = [False] * ((N_max - p * p) // p + 1)

print(f"筛法完成，耗时 {time.time() - t0:.2f} 秒。")


def li_approx(x):
    if x <= 2:
        return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))


def compute_S_from_counts(obs_counts, ideal_counts):
    SP_vals = []
    for obs, v_star in zip(obs_counts, ideal_counts):
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        diff = obs - v_star
        SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
        SP_vals.append(SP_i)
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if len(SP_vals) > 1 else 0.0
    S = SP_mean / (1 + delta_SP)
    return S, SP_mean, delta_SP


results = []

for N in scale_points:
    total_primes_est = N / math.log(N)
    K = max(10, round(total_primes_est / E_target))
    interval_size = N // K

    ideal_counts = []
    obs_counts = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size if i < K - 1 else N
        ideal = li_approx(b + 1) - li_approx(a)
        ideal_counts.append(ideal)
        obs = sum(is_prime[a: b + 1])
        obs_counts.append(obs)

    S_real, SP_real, delta_SP_real = compute_S_from_counts(obs_counts, ideal_counts)
    total_primes = sum(obs_counts)

    perturbed_S_list = []
    for _ in range(M_perturb):
        fake_positions = random.sample(range(2, N + 1), total_primes)
        fake_is_prime = [False] * (N + 1)
        for pos in fake_positions:
            fake_is_prime[pos] = True
        fake_obs = []
        for i in range(K):
            a = i * interval_size + 1
            b = (i + 1) * interval_size if i < K - 1 else N
            fake_obs.append(sum(fake_is_prime[a: b + 1]))
        S_fake, _, _ = compute_S_from_counts(fake_obs, ideal_counts)
        perturbed_S_list.append(S_fake)

    mean_pert = statistics.mean(perturbed_S_list)
    std_pert = statistics.stdev(perturbed_S_list)
    count_higher = sum(1 for s in perturbed_S_list if s >= S_real)
    p_val = (count_higher + 1) / (M_perturb + 1)
    eta = (S_real - mean_pert) / std_pert if std_pert > 0 else float('inf')

    results.append({
        'N': N, 'K': K, 'total_primes': total_primes,
        'S_real': S_real, 'SP_real': SP_real, 'delta_SP_real': delta_SP_real,
        'S_pert_mean': mean_pert, 'S_pert_std': std_pert,
        'eta': eta, 'p_value': p_val
    })
    print(f"N={N:>9,}  K={K:>5}  S={S_real:.6f}  SP={SP_real:.6f}  δSP={delta_SP_real:.6f}  η={eta:.2f}  p={p_val:.4f}")

print("\n" + "=" * 80)
print("全尺度扫描结果汇总")
print("=" * 80)
print(f"{'N':<12} {'S_real':<10} {'SP':<10} {'δSP':<10} {'η':<8} {'p':<8}")
print("-" * 60)
for r in results:
    print(f"{r['N']:<12,} {r['S_real']:<10.6f} {r['SP_real']:<10.6f} {r['delta_SP_real']:<10.6f} {r['eta']:<8.2f} {r['p_value']:<8.4f}")

S_vals = [r['S_real'] for r in results]
diffs = [S_vals[i+1] - S_vals[i] for i in range(len(S_vals)-1)]
sign_changes = sum(1 for i in range(len(diffs)-1) if diffs[i] * diffs[i+1] < 0)
print(f"\n符号变化次数：{sign_changes}（共 {len(diffs)} 个相邻差分）")
if sign_changes > 0:
    print("结论：S(N) 不是单调函数，存在波动。")
else:
    print("结论：S(N) 单调变化。")

N_list = [r['N'] for r in results]
S_list = [r['S_real'] for r in results]
SP_list = [r['SP_real'] for r in results]
delta_list = [r['delta_SP_real'] for r in results]
eta_list = [r['eta'] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(N_list, S_list, 'o-', color='darkblue', linewidth=2, markersize=8, label='S(N)')
ax1.set_xscale('log')
ax1.set_xlabel('N (log scale)')
ax1.set_ylabel('S')
ax1.set_title(f'S(N) from 10K to {N_max//1000000}M')
ax1.grid(True, alpha=0.3)

for i, (n, s) in enumerate(zip(N_list, S_list)):
    ax1.annotate(f'{s:.4f}', (n, s), textcoords="offset points", xytext=(0, 10),
                ha='center', fontsize=7, color='darkblue')

ax2.plot(N_list, SP_list, 's-', color='forestgreen', label='SP_mean', linewidth=1.5)
ax2.plot(N_list, delta_list, '^-', color='darkorange', label='delta_SP', linewidth=1.5)
ax2.set_xscale('log')
ax2.set_xlabel('N (log scale)')
ax2.set_title('SP_mean and delta_SP Decomposition')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_multiscale_S_decomposition.png', dpi=150)
plt.close()
print("图已保存: 04_multiscale_S_decomposition.png")
