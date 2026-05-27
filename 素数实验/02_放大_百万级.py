import math
import random
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 1_000_000
K = 500
M = 100
random.seed(42)

print(f"N = {N}, K = {K}, M = {M}")

print(f"正在生成 1 到 {N} 的素数表...")
t0 = time.time()

is_prime = [True] * (N + 1)
is_prime[0] = is_prime[1] = False
for p in range(2, int(N**0.5) + 1):
    if is_prime[p]:
        is_prime[p * p: N + 1: p] = [False] * ((N - p * p) // p + 1)

prime_positions = [i for i in range(2, N + 1) if is_prime[i]]
total_primes = len(prime_positions)
print(f"筛法完成，耗时 {time.time() - t0:.2f} 秒。共找到 {total_primes} 个素数。")


def li_approx(x):
    if x <= 2:
        return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))


interval_size = N // K
ideal_counts = []
obs_counts = []

for i in range(K):
    a = i * interval_size + 1
    b = (i + 1) * interval_size
    ideal = li_approx(b + 1) - li_approx(a)
    ideal_counts.append(ideal)
    obs = sum(is_prime[a: b + 1])
    obs_counts.append(obs)


def compute_S_from_counts(obs_list, ideal_list):
    SP_vals = []
    for obs, v_star in zip(obs_list, ideal_list):
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        diff = obs - v_star
        SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
        SP_vals.append(SP_i)
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if K > 1 else 0.0
    S = SP_mean / (1 + delta_SP)
    return S, SP_mean, delta_SP


print("正在计算真实序列 S...")
t_calc = time.time()
S_real, SP_mean_real, delta_SP_real = compute_S_from_counts(obs_counts, ideal_counts)
print(f"计算完成，耗时 {time.time() - t_calc:.2f} 秒。")
print(f"S_real = {S_real:.6f} (SP={SP_mean_real:.6f}, δSP={delta_SP_real:.6f})")

print(f"正在生成 {M} 条扰动序列...")
t_pert = time.time()
perturbed_S = []

for idx in range(M):
    fake_positions = random.sample(range(2, N + 1), total_primes)
    fake_is_prime = [False] * (N + 1)
    for pos in fake_positions:
        fake_is_prime[pos] = True
    fake_obs = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        count = sum(fake_is_prime[a: b + 1])
        fake_obs.append(count)
    S_fake, _, _ = compute_S_from_counts(fake_obs, ideal_counts)
    perturbed_S.append(S_fake)
    if (idx + 1) % 20 == 0:
        print(f"  已生成 {idx + 1}/{M} 条扰动序列...")

print(f"扰动序列生成完成，耗时 {time.time() - t_pert:.2f} 秒。")

mean_perturbed = statistics.mean(perturbed_S)
stdev_perturbed = statistics.stdev(perturbed_S)
count_higher = sum(1 for s in perturbed_S if s >= S_real)
p_value = (count_higher + 1) / (M + 1)
eta = (S_real - mean_perturbed) / stdev_perturbed if stdev_perturbed > 0 else float('inf')

print(f"\n总素数: {total_primes}")
print(f"K: {K}")
print(f"S_real: {S_real:.6f}")
print(f"SP: {SP_mean_real:.6f}")
print(f"δSP: {delta_SP_real:.6f}")
print(f"扰动均值: {mean_perturbed:.6f}")
print(f"扰动标准差: {stdev_perturbed:.6f}")
print(f"η: {eta:.2f}")
print(f"p: {p_value:.4f}")
print(f"S >= S_real: {count_higher}/{M}")
if p_value <= 0.05:
    print("结论：显著")
else:
    print("结论：不显著")

print(f"\nN=10000: S=0.9077")
print(f"N=20000: S=0.8760")
print(f"N=50000: S=0.8582")
print(f"N=100000: S=0.8380")
print(f"N=1000000: S={S_real:.4f}")

plt.figure(figsize=(8, 5))
plt.hist(perturbed_S, bins=30, color='lightblue', edgecolor='grey', alpha=0.8, label='Perturbed S')
plt.axvline(S_real, color='red', linewidth=2, linestyle='--', label=f'S_real = {S_real:.4f}')
plt.axvline(mean_perturbed, color='blue', linewidth=1.5, linestyle=':', label=f'Null mean = {mean_perturbed:.4f}')
plt.xlabel('S')
plt.ylabel('Frequency')
plt.title(f'S Distribution (N=1,000,000, K=500)\nη = {eta:.2f}σ, p = {p_value:.4f}')
plt.legend()
plt.tight_layout()
plt.savefig('02_S_distribution_1M.png', dpi=150)
plt.close()
print("图已保存: 02_S_distribution_1M.png")
