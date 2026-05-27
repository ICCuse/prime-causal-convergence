import math
import random
import statistics
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_max = 100_000_000
E_target = 50
M_perturb = 50

scale_points = [
    100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000, 100_000_000
]

print(f"=== 因果变换：终极素数扫描 ===")
print(f"目标：从 {scale_points[0]:,} 到 {N_max:,} 的 S(N) 曲线")
print(f"引擎：PyPy + {os.cpu_count()} 核并行")
print("=" * 80)

print(f"\n正在生成 1 到 {N_max:,} 的素数表...")
t0 = time.time()

is_prime = bytearray(b'\x01') * (N_max + 1)
is_prime[0] = is_prime[1] = 0
limit = int(N_max**0.5) + 1
for p in range(2, limit):
    if is_prime[p]:
        step = p
        start = p * p
        is_prime[start: N_max + 1: step] = bytearray((N_max - start) // step + 1)

print(f"素数表生成完成，耗时 {time.time() - t0:.2f} 秒。")


def li_approx(x):
    if x <= 2:
        return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))


def compute_S_from_counts(obs_list, ideal_list):
    SP_vals = []
    for obs, v_star in zip(obs_list, ideal_list):
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        diff = obs - v_star
        SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
        SP_vals.append(SP_i)
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if len(SP_vals) > 1 else 0.0
    S = SP_mean / (1 + delta_SP)
    return S, SP_mean, delta_SP


def generate_fake_S(args):
    N, K, total_primes, ideal_counts = args
    random.seed((os.getpid() * int(time.time() * 1000)) % (2**32))
    fake_positions = random.sample(range(2, N + 1), total_primes)
    fake_is_prime = [False] * (N + 1)
    for pos in fake_positions:
        fake_is_prime[pos] = True
    interval_size = N // K
    fake_obs = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size if i < K - 1 else N
        fake_obs.append(sum(fake_is_prime[a: b + 1]))
    SP_vals = []
    for obs, v_star in zip(fake_obs, ideal_counts):
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        diff = obs - v_star
        SP_i = math.exp(- (diff ** 2) / (2 * sigma ** 2))
        SP_vals.append(SP_i)
    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if len(SP_vals) > 1 else 0.0
    S = SP_mean / (1 + delta_SP)
    return S


if __name__ == '__main__':
    multiprocessing.freeze_support()
    results = []
    num_workers = max(1, int(os.cpu_count() * 0.75))

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

        print(f"\nN = {N:>10,}  | K = {K:>6}  | 素数 = {total_primes:>8} | 并行生成 {M_perturb} 条扰动序列...")
        t_pert_start = time.time()

        args_list = [(N, K, total_primes, ideal_counts) for _ in range(M_perturb)]
        perturbed_S_list = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(generate_fake_S, args) for args in args_list]
            for future in as_completed(futures):
                perturbed_S_list.append(future.result())

        t_pert = time.time() - t_pert_start
        print(f"  完成，耗时 {t_pert:.2f} 秒（并行 {num_workers} 核）")

        mean_pert = statistics.mean(perturbed_S_list)
        std_pert = statistics.stdev(perturbed_S_list)
        count_higher = sum(1 for s in perturbed_S_list if s >= S_real)
        p_val = (count_higher + 1) / (M_perturb + 1)
        eta = (S_real - mean_pert) / std_pert if std_pert > 0 else float('inf')

        results.append({
            'N': N, 'K': K, 'total_primes': total_primes,
            'S_real': S_real, 'SP_real': SP_real, 'delta_SP_real': delta_SP_real,
            'eta': eta, 'p_value': p_val
        })
        print(f"  S_real = {S_real:.6f}  |  SP = {SP_real:.6f}  |  δSP = {delta_SP_real:.6f}")
        print(f"  效应量 η = {eta:.2f}  |  p = {p_val:.4f}")

    print("\n" + "=" * 80)
    print("终极扫描结果：素数因果连接强度 S(N)")
    print("=" * 80)
    print(f"{'N':<12} {'S_real':<10} {'SP':<10} {'δSP':<10} {'η':<8} {'p':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['N']:<12,} {r['S_real']:<10.6f} {r['SP_real']:<10.6f} {r['delta_SP_real']:<10.6f} {r['eta']:<8.2f} {r['p_value']:<8.4f}")

    N_list = [r['N'] for r in results]
    S_list = [r['S_real'] for r in results]
    eta_list = [r['eta'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(N_list, S_list, 'o-', color='darkblue', linewidth=2, markersize=8)
    ax1.set_xscale('log')
    ax1.fill_between(N_list, min(S_list) - 0.005, [S_list[-1]] * len(S_list),
                     alpha=0.15, color='lightblue')
    ax1.set_xlabel('N (log scale)')
    ax1.set_ylabel('S(N)')
    ax1.set_title(f'S(N) from 100K to 100M\nη_max = {max(eta_list):.0f}σ')
    ax1.grid(True, alpha=0.3)
    for n, s in zip(N_list, S_list):
        ax1.annotate(f'{s:.4f}', (n, s), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=7, color='darkblue')

    ax2.bar(range(len(eta_list)), eta_list, color='steelblue', edgecolor='navy', alpha=0.8)
    ax2.set_xticks(range(len(eta_list)))
    ax2.set_xticklabels([f'{n//1000}K' if n < 1e6 else f'{n//1000000}M' for n in N_list],
                        rotation=45, fontsize=7)
    ax2.set_ylabel('η (σ)')
    ax2.set_title('Effect Size Growth')
    ax2.axhline(y=5, color='grey', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('05_100M_scan.png', dpi=150)
    plt.close()
    print("图已保存: 05_100M_scan.png")
