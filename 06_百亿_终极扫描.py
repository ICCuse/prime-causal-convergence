import math
import random
import statistics
import time
from multiprocessing import Pool, freeze_support, cpu_count
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_max = 10_000_000_000

scale_points = [
    100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000, 100_000_000,
    200_000_000, 500_000_000,
    1_000_000_000, 2_000_000_000, 5_000_000_000, 10_000_000_000
]

E_target = 50
M_perturb = 100
BLOCK_SIZE = 100_000_000
NUM_WORKERS = max(2, int(cpu_count() * 0.7))

_small_primes_global = None


def _init_worker(small_primes):
    global _small_primes_global
    _small_primes_global = small_primes


def li_approx(x):
    if x <= 2:
        return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))


def compute_S(args):
    obs_list, ideal_list = args
    SP_vals = []
    for obs, v_star in zip(obs_list, ideal_list):
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        diff = obs - v_star
        SP_vals.append(math.exp(-(diff ** 2) / (2 * sigma ** 2)))
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if len(SP_vals) > 1 else 0.0
    S = SP_mean / (1 + delta_SP)
    return S, SP_mean, delta_SP


def generate_small_primes(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if is_prime[p]:
            is_prime[p * p: limit + 1: p] = bytearray((limit - p * p) // p + 1)
    return [i for i in range(2, limit + 1) if is_prime[i]]


def sieve_block(block_idx_and_params):
    bi, block_start, block_end, interval_size, K = block_idx_and_params
    small_primes = _small_primes_global
    block_len = block_end - block_start + 1
    block = bytearray(b'\x01') * block_len
    for p in small_primes:
        start = max(p * p, ((block_start + p - 1) // p) * p)
        if start > block_end:
            continue
        for m in range(start, block_end + 1, p):
            block[m - block_start] = 0
    if block_start == 1:
        block[0] = 0
    local_counts = [0] * K
    for i in range(block_len):
        if block[i]:
            n = block_start + i
            idx = (n - 1) // interval_size
            if idx >= K:
                idx = K - 1
            local_counts[idx] += 1
    return local_counts


def perturb_one(seed_and_params):
    seed, ideal_counts, total_primes = seed_and_params
    random.seed(seed)
    K = len(ideal_counts)
    obs = [max(0, round(random.gauss(v, math.sqrt(v)))) for v in ideal_counts]
    total = sum(obs)
    if total > 0:
        scale = total_primes / total
        obs = [max(0, round(o * scale)) for o in obs]
    S, _, _ = compute_S((obs, ideal_counts))
    return S


if __name__ == '__main__':
    freeze_support()

    print(f"=== 因果变换：百亿级素数扫描（{NUM_WORKERS} 核并行）===")
    print(f"目标：绘制从 {scale_points[0]:,} 到 {N_max:,} 的 S(N) 曲线")
    print("=" * 80)

    sqrt_Nmax = int(N_max**0.5) + 1
    print(f"\n预生成小素数表 (2 ~ {sqrt_Nmax:,})...")
    t0 = time.time()
    small_primes = generate_small_primes(sqrt_Nmax)
    print(f"完成，{len(small_primes):,} 个小素数，耗时 {time.time() - t0:.2f}s")

    results = []
    total_start = time.time()

    for si, N in enumerate(scale_points):
        total_primes_est = N / math.log(N)
        K = max(10, round(total_primes_est / E_target))
        interval_size = N // K

        ideal_counts = []
        for i in range(K):
            a = i * interval_size + 1
            b = (i + 1) * interval_size if i < K - 1 else N
            ideal_counts.append(li_approx(b + 1) - li_approx(a))

        num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        print(f"\n[{si + 1}/{len(scale_points)}] N = {N:>12,}  | K = {K:>7}  | {num_blocks} 块 x {NUM_WORKERS} 核并行...")
        t0 = time.time()

        block_tasks = [(bi, bi * BLOCK_SIZE + 1, min((bi + 1) * BLOCK_SIZE, N), interval_size, K) for bi in range(num_blocks)]

        with Pool(processes=NUM_WORKERS, initializer=_init_worker, initargs=(small_primes,)) as pool:
            block_results = pool.map(sieve_block, block_tasks)

        total_counts = [0] * K
        for local in block_results:
            for i in range(K):
                total_counts[i] += local[i]

        t_sieve = time.time() - t0
        total_primes = sum(total_counts)
        print(f"  筛法: {t_sieve:.2f}s  |  素数: {total_primes:,}  |  Li(N) ~ {li_approx(N):,.0f}")

        S_real, SP_real, delta_SP_real = compute_S((total_counts, ideal_counts))
        print(f"  S_real = {S_real:.6f}  |  SP = {SP_real:.6f}  |  δSP = {delta_SP_real:.6f}")

        perturb_tasks = [(hash((42, N, i)), ideal_counts, total_primes) for i in range(M_perturb)]
        with Pool(processes=NUM_WORKERS) as pool:
            perturbed_S_list = pool.map(perturb_one, perturb_tasks)

        mean_pert = statistics.mean(perturbed_S_list)
        std_pert = statistics.stdev(perturbed_S_list)
        count_higher = sum(1 for s in perturbed_S_list if s >= S_real)
        p_val = (count_higher + 1) / (M_perturb + 1)
        eta = (S_real - mean_pert) / std_pert if std_pert > 0 else float('inf')

        results.append({
            'N': N, 'K': K, 'total_primes': total_primes,
            'S_real': S_real, 'SP_real': SP_real, 'delta_SP_real': delta_SP_real,
            'eta': eta, 'p_value': p_val, 't_sieve': t_sieve
        })
        print(f"  S_null = {mean_pert:.6f}  |  η = {eta:.2f}  |  p = {p_val:.4f}")

    print("\n" + "=" * 85)
    print("百亿级扫描结果：素数因果连接强度 S(N)")
    print("=" * 85)
    print(f"{'N':<16} {'S_real':<10} {'SP':<10} {'δSP':<10} {'η':<10} {'p':<10} {'筛时':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['N']:<16,} {r['S_real']:<10.6f} {r['SP_real']:<10.6f} "
              f"{r['delta_SP_real']:<10.6f} {r['eta']:<10.2f} {r['p_value']:<10.4f} "
              f"{r['t_sieve']:<10.2f}s")

    print(f"\n总耗时: {time.time() - total_start:.2f}s")

    N_list = np.array([r['N'] for r in results], dtype=float)
    S_list = np.array([r['S_real'] for r in results])
    eta_list = np.array([r['eta'] for r in results])

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.semilogx(N_list, S_list, 'o-', color='darkblue', linewidth=2.5, markersize=9,
                markerfacecolor='white', markeredgewidth=2, label='S(N) measured')
    for n, s in zip(N_list, S_list):
        ax.annotate(f'{s:.4f}', (n, s), textcoords="offset points",
                     xytext=(0, -16), ha='center', fontsize=7.5, color='darkblue')
    ax.set_xlabel('N (log scale)', fontsize=12)
    ax.set_ylabel('S(N)', fontsize=12)
    ax.set_title(f'S(N) Causal Convergence: 16 points from {N_list[0]/1000:.0f}K to 10B\n'
                 f'S(10B) = {S_list[-1]:.6f}  |  η_max = {max(eta_list):.0f}σ', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('06_10B_scan.png', dpi=150)
    plt.close()
    print("图已保存: 06_10B_scan.png")
