import math
import random
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a, b = 1500, 1800
K = 10
M_perturb = 100
random.seed(42)


def sieve_range(start, end):
    is_prime = [True] * (end + 1)
    is_prime[0] = is_prime[1] = False
    limit = int(end**0.5) + 1
    for p in range(2, limit):
        if is_prime[p]:
            for multiple in range(max(p * p, (start // p) * p), end + 1, p):
                if multiple >= start:
                    is_prime[multiple] = False
    return [i for i in range(start, end + 1) if is_prime[i]]


primes_in_window = sieve_range(a, b)
total_primes = len(primes_in_window)
print(f"窗口 [{a}, {b}] 内素数个数: {total_primes}")


def li_approx(x):
    if x <= 2:
        return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))


ideal_counts = []
interval_size = (b - a + 1) // K
for i in range(K):
    left = a + i * interval_size
    right = left + interval_size - 1
    if i == K - 1:
        right = b
    ideal = li_approx(right + 1) - li_approx(left)
    ideal_counts.append(ideal)


def compute_S_local(prime_list):
    obs_counts = []
    for i in range(K):
        left = a + i * interval_size
        right = left + interval_size - 1
        if i == K - 1:
            right = b
        obs = sum(1 for p in prime_list if left <= p <= right)
        obs_counts.append(obs)
    SP_vals = []
    for obs, v_star in zip(obs_counts, ideal_counts):
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        diff = obs - v_star
        SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
        SP_vals.append(SP_i)
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if K > 1 else 0.0
    S = SP_mean / (1 + delta_SP)
    return S, obs_counts, SP_vals


S_real, obs_real, SP_real = compute_S_local(primes_in_window)
print(f"\n真实局部序列 S_local = {S_real:.6f}")

perturbed_S = []
all_numbers = list(range(a, b + 1))
for _ in range(M_perturb):
    fake_primes = random.sample(all_numbers, total_primes)
    S_fake, _, _ = compute_S_local(fake_primes)
    perturbed_S.append(S_fake)

mean_pert = statistics.mean(perturbed_S)
std_pert = statistics.stdev(perturbed_S)
count_higher = sum(1 for s in perturbed_S if s >= S_real)
p_val = (count_higher + 1) / (M_perturb + 1)
eta = (S_real - mean_pert) / std_pert if std_pert > 0 else float('inf')

print(f"扰动序列 S 均值: {mean_pert:.6f}")
print(f"扰动序列 S 标准差: {std_pert:.6f}")
print(f"效应量 η: {eta:.2f}")
print(f"p 值: {p_val:.4f}")
print(f"扰动序列中 S >= S_real 的条数: {count_higher}/{M_perturb}")
print(f"\n--- 对比全局极限 S_inf ---")
print(f"局部 S = {S_real:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.hist(perturbed_S, bins=25, color='lightblue', edgecolor='grey', alpha=0.8, label='Perturbed S')
ax1.axvline(S_real, color='red', linewidth=2, linestyle='--', label=f'S_local = {S_real:.4f}')
ax1.axvline(mean_pert, color='blue', linewidth=1.5, linestyle=':', label=f'Null mean = {mean_pert:.4f}')
ax1.set_xlabel('S')
ax1.set_ylabel('Frequency')
ax1.set_title(f'Local S Distribution [{a}, {b}]\nη = {eta:.2f}σ, p = {p_val:.4f}')
ax1.legend()

obs_list = obs_real
ideal_list = ideal_counts
items = [f'Int{i+1}' for i in range(K)]
x = range(K)
width = 0.35
ax2.bar([i - width/2 for i in x], obs_list, width, color='steelblue', label='Observed')
ax2.bar([i + width/2 for i in x], ideal_list, width, color='coral', alpha=0.7, label='Ideal (Li)')
ax2.set_xlabel('Interval')
ax2.set_ylabel('Prime Count')
ax2.set_title('Observed vs Ideal Per Interval')
ax2.set_xticks(x)
ax2.set_xticklabels(items, rotation=45, fontsize=7)
ax2.legend()

plt.tight_layout()
plt.savefig('07_local_window_euler.png', dpi=150)
plt.close()
print("图已保存: 07_local_window_euler.png")
