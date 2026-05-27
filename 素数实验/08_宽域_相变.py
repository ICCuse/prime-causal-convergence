import math
import random
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

a, b = 1700, 3000
K = 20
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


interval_size = (b - a + 1) // K
ideal_counts = []
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
    return S, obs_counts


S_real, obs_real = compute_S_local(primes_in_window)
print(f"真实序列 S_local = {S_real:.6f}")

perturbed_S = []
all_numbers = list(range(a, b + 1))
for _ in range(M_perturb):
    fake_primes = random.sample(all_numbers, total_primes)
    S_fake, _ = compute_S_local(fake_primes)
    perturbed_S.append(S_fake)

mean_pert = statistics.mean(perturbed_S)
std_pert = statistics.stdev(perturbed_S)
count_higher = sum(1 for s in perturbed_S if s >= S_real)
p_val = (count_higher + 1) / (M_perturb + 1)
eta = (S_real - mean_pert) / std_pert if std_pert > 0 else float('inf')

print(f"\n扰动序列 S 均值: {mean_pert:.6f}")
print(f"扰动序列 S 标准差: {std_pert:.6f}")
print(f"效应量 η: {eta:.2f}")
print(f"p 值: {p_val:.4f}")
print(f"扰动序列中 S >= S_real 的条数: {count_higher}/{M_perturb}")

print(f"\n--- 逐区间素数分布 ---")
print(f"{'区间':<8} {'范围':<16} {'实际素数':<10} {'理想预测':<10} {'局部SP':<10}")
print("-" * 60)
for i in range(K):
    left = a + i * interval_size
    right = left + interval_size - 1
    if i == K - 1:
        right = b
    v_star = ideal_counts[i]
    sigma = math.sqrt(v_star) if v_star > 0 else 1.0
    diff = obs_real[i] - v_star
    SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
    print(f"区间{i+1:<3}  [{left:>4}, {right:>4}]  {obs_real[i]:<10}  {v_star:<10.2f}  {SP_i:<10.4f}")

SP_local_list = []
for i in range(K):
    v_star = ideal_counts[i]
    sigma = math.sqrt(v_star) if v_star > 0 else 1.0
    diff = obs_real[i] - v_star
    SP_local_list.append(math.exp(- (diff ** 2) / (2 * sigma ** 2)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

items = [f'[{a + i * interval_size}, {a + (i+1) * interval_size - 1}]' for i in range(K)]
x = range(K)
width = 0.35
ax1.bar([i - width/2 for i in x], obs_real, width, color='steelblue', label='Observed Primes')
ax1.bar([i + width/2 for i in x], ideal_counts, width, color='coral', alpha=0.7, label='Ideal (Li)')
ax1.set_xlabel('Interval')
ax1.set_ylabel('Prime Count')
ax1.set_title(f'S_local = {S_real:.4f}  |  η = {eta:.2f}σ  |  Window [{a}, {b}]')
ax1.set_xticks(x)
ax1.set_xticklabels(items, rotation=30, fontsize=7)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

colors = ['forestgreen' if sp >= 0.99 else 'orange' if sp >= 0.95 else 'crimson' for sp in SP_local_list]
bars = ax2.bar(range(K), SP_local_list, color=colors, edgecolor='darkgrey', alpha=0.85)
ax2.axhline(y=0.99, color='grey', linestyle='--', alpha=0.5, label='SP = 0.99')
ax2.axhline(y=0.95, color='grey', linestyle=':', alpha=0.5, label='SP = 0.95')
ax2.set_xlabel('Interval')
ax2.set_ylabel('Local SP')
ax2.set_title('Phase-Change Pattern: Interval-level SP')
ax2.set_xticks(x)
ax2.set_xticklabels(items, rotation=30, fontsize=7)
ax2.set_ylim(0.8, 1.02)
ax2.legend()

plt.tight_layout()
plt.savefig('08_wide_phase_transition.png', dpi=150)
plt.close()
print("图已保存: 08_wide_phase_transition.png")
