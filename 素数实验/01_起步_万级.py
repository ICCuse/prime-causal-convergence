import math
import random
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 10000
K = 50
M = 1000
random.seed(42)

def sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            for multiple in range(p * p, limit + 1, p):
                is_prime[multiple] = False
    return is_prime

is_prime = sieve(N)
prime_positions = [i for i in range(2, N + 1) if is_prime[i]]
total_primes = len(prime_positions)

def li(x):
    if x < 2:
        return 0.0
    steps = max(1000, int((x - 2) * 10))
    dt = (x - 2) / steps
    integral = 0.0
    for i in range(steps):
        t1 = 2 + i * dt
        t2 = t1 + dt
        integral += (1.0 / math.log(t1) + 1.0 / math.log(t2)) * dt / 2
    return integral

interval_size = N // K
ideal_counts = []
for i in range(K):
    a = i * interval_size + 1
    b = (i + 1) * interval_size
    expected = li(b + 1) - li(a)
    ideal_counts.append(expected)

def compute_S(prime_bools):
    SP_vals = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        obs = sum(1 for x in range(a, b + 1) if prime_bools[x])
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
    return S

prime_bools = [False] * (N + 1)
for p in prime_positions:
    prime_bools[p] = True

S_real = compute_S(prime_bools)
print(f"S_real = {S_real:.6f}")

perturbed_S = []
print(f"正在生成 {M} 条扰动序列...")
for _ in range(M):
    fake_positions = random.sample(range(2, N + 1), total_primes)
    fake_bools = [False] * (N + 1)
    for pos in fake_positions:
        fake_bools[pos] = True
    S_fake = compute_S(fake_bools)
    perturbed_S.append(S_fake)

mean_perturbed = statistics.mean(perturbed_S)
stdev_perturbed = statistics.stdev(perturbed_S)
count_higher = sum(1 for s in perturbed_S if s >= S_real)
p_value = (count_higher + 1) / (M + 1)
eta = (S_real - mean_perturbed) / stdev_perturbed if stdev_perturbed > 0 else float('inf')

print(f"扰动序列 S 均值: {mean_perturbed:.6f}")
print(f"扰动序列 S 标准差: {stdev_perturbed:.6f}")
print(f"S_real: {S_real:.6f}")
print(f"效应量 η: {eta:.2f}")
print(f"p 值: {p_value:.4f}")
print(f"扰动序列中 S >= S_real: {count_higher}/{M}")
if p_value <= 0.05:
    print("结论：显著")
else:
    print("结论：不显著")

plt.figure(figsize=(8, 5))
plt.hist(perturbed_S, bins=40, color='lightblue', edgecolor='grey', alpha=0.8, label='Perturbed S')
plt.axvline(S_real, color='red', linewidth=2, linestyle='--', label=f'S_real = {S_real:.4f}')
plt.axvline(mean_perturbed, color='blue', linewidth=1.5, linestyle=':', label=f'Null mean = {mean_perturbed:.4f}')
plt.xlabel('S')
plt.ylabel('Frequency')
plt.title(f'S Distribution (N=10,000, K=50)\nη = {eta:.2f}σ, p = {p_value:.4f}')
plt.legend()
plt.tight_layout()
plt.savefig('01_S_distribution_10K.png', dpi=150)
plt.close()
print("图已保存: 01_S_distribution_10K.png")
