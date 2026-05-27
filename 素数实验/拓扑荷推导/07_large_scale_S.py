import math, statistics, time, sys
import numpy as np

K = 50
N_VALS = [10**5, 10**6, 10**7, 10**8, 10**9]

def primes_up_to(limit):
    is_p = bytearray(b'\x01') * (limit + 1)
    if limit >= 0: is_p[0] = 0
    if limit >= 1: is_p[1] = 0
    for p in range(2, int(limit**0.5) + 1):
        if is_p[p]:
            step = p
            start = p * p
            is_p[start:limit+1:step] = b'\x00' * ((limit - start) // step + 1)
    return [i for i in range(2, limit + 1) if is_p[i]]

def segmented_prime_counts(N, small_primes, K=K):
    interval_size = N // K
    counts = []
    for i in range(K):
        lo = i * interval_size + 1
        hi = (i + 1) * interval_size
        if i == K - 1:
            hi = N
        if hi < 2:
            counts.append(0)
            continue
        lo = max(lo, 2)

        seg_len = hi - lo + 1
        segment = bytearray(b'\x01') * seg_len

        for p in small_primes:
            if p * p > hi:
                break
            start = ((lo + p - 1) // p) * p
            if start < p * p:
                start = p * p
            if start > hi:
                continue
            start_idx = start - lo
            step = p
            segment[start_idx:seg_len:step] = b'\x00' * ((seg_len - 1 - start_idx) // step + 1)

        cnt = sum(1 for b in segment if b == 1)
        counts.append(cnt)

    return counts

def li_approx(x):
    if x <= 2: return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1 / log_x + 2 / (log_x**2))

def zeta_count(T):
    if T <= 0: return 0.0
    return T / (2 * math.pi) * math.log(T / (2 * math.pi * math.e)) + 7/8

def zeta_density(T):
    if T <= 0: return 0.0
    return math.log(T / (2 * math.pi)) / (2 * math.pi)

def zeta_zero_sum(power, T_max):
    total = 0.0
    ranges = [
        (10.0, 30.0, 4000),
        (30.0, 100.0, 4000),
        (100.0, 300.0, 4000),
        (300.0, 1000.0, 4000),
        (1000.0, 10000.0, 4000),
        (10000.0, 100000.0, 4000),
        (100000.0, T_max, 4000),
    ]
    for lo, hi, n in ranges:
        if hi > T_max:
            hi = T_max
        if hi <= lo:
            continue
        dT = (hi - lo) / n
        for i in range(n):
            T_mid = lo + (i + 0.5) * dT
            dens = zeta_density(T_mid)
            if dens <= 0:
                continue
            total += dens * 2.0 / (0.25 + T_mid**2)**power * dT
    return total

print("=" * 70)
print("  大尺度 S(N) 外推 — 分段筛 + 零点求和")
print("=" * 70)
print()

N_max = max(N_VALS)
sqrt_N_max = int(N_max**0.5)
print(f"生成 sqrt({N_max}) = {sqrt_N_max} 以内的小素数...")
t0 = time.time()
small_primes = primes_up_to(sqrt_N_max)
print(f"  {len(small_primes)} 个小素数, 耗时 {time.time()-t0:.1f}s")
print()

results = []

for N in N_VALS:
    print(f"--- N = {N:,} ({int(math.log10(N))} 位) ---")
    t1 = time.time()

    counts = segmented_prime_counts(N, small_primes, K)
    total_primes = sum(counts)

    interval_size = N // K
    ideal = []
    for i in range(K):
        a = i * interval_size + 1
        b = (i + 1) * interval_size
        if i == K - 1: b = N
        ideal.append(li_approx(b) - li_approx(a - 1))

    SP_vals = []
    for i in range(K):
        obs = counts[i]
        v_star = ideal[i]
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        SP_vals.append(math.exp(-(obs - v_star)**2 / (2 * sigma**2)))

    SP_mean = statistics.mean(SP_vals)
    delta_SP = statistics.variance(SP_vals) if K > 1 else 0.0
    S_val = SP_mean / (1 + delta_SP)

    T_eff = total_primes * 2 * math.pi
    lo, hi = 1.0, 1e12
    for _ in range(80):
        mid = (lo + hi) / 2
        if zeta_count(mid) > total_primes:
            hi = mid
        else:
            lo = mid
    T_eff = (lo + hi) / 2

    sum_k10 = zeta_zero_sum(0.50, T_eff)
    sum_k15 = zeta_zero_sum(0.75, T_eff)
    sum_k2  = zeta_zero_sum(1.0,  T_eff)

    dt = time.time() - t1
    print(f"  prime_count={total_primes}  S={S_val:.6f}  T_eff={T_eff:.0f}  "
          f"sum_k1.0={sum_k10:.6f}  sum_k1.5={sum_k15:.6f}  ({dt:.1f}s)")

    results.append({
        "N": N, "primes": total_primes, "S": S_val,
        "T_eff": T_eff, "sum_k10": sum_k10, "sum_k15": sum_k15, "sum_k2": sum_k2
    })

print()
print("=" * 70)
print("  公式检验: S(N) = exp(-c * Σ 1/|ρ|^k)")
print("=" * 70)

print(f"  {'N':>10s} {'S(N)':>10s} {'c (k=1.0)':>10s} {'c (k=1.5)':>10s} "
      f"{'c (k=2.0)':>10s}")
print(f"  {'-'*55}")

c_vals_k10 = []
c_vals_k15 = []
c_vals_k2 = []

for r in results:
    N = r["N"]
    S = r["S"]
    s10 = r["sum_k10"]
    s15 = r["sum_k15"]
    s2 = r["sum_k2"]

    c10 = -math.log(S) / s10 if s10 > 0 and S > 0 else 0
    c15 = -math.log(S) / s15 if s15 > 0 and S > 0 else 0
    c2  = -math.log(S) / s2  if s2 > 0 and S > 0 else 0

    c_vals_k10.append(c10)
    c_vals_k15.append(c15)
    c_vals_k2.append(c2)

    print(f"  {N:>10,d} {S:>10.6f} {c10:>10.4f} {c15:>10.4f} {c2:>10.4f}")

print()

if len(c_vals_k15) >= 2:
    c10_mean = statistics.mean(c_vals_k10)
    c10_sd = statistics.stdev(c_vals_k10)
    c15_mean = statistics.mean(c_vals_k15)
    c15_sd = statistics.stdev(c_vals_k15)
    c2_mean = statistics.mean(c_vals_k2)
    c2_sd = statistics.stdev(c_vals_k2)

    c10_cv = c10_sd / c10_mean * 100 if c10_mean > 0 else 0
    c15_cv = c15_sd / c15_mean * 100 if c15_mean > 0 else 0
    c2_cv = c2_sd / c2_mean * 100 if c2_mean > 0 else 0

    print(f"  k=1.0 系数: mean={c10_mean:.4f}  sd={c10_sd:.4f}  CV={c10_cv:.1f}%")
    print(f"  k=1.5 系数: mean={c15_mean:.4f}  sd={c15_sd:.4f}  CV={c15_cv:.1f}%")
    print(f"  k=2.0 系数: mean={c2_mean:.4f}  sd={c2_sd:.4f}  CV={c2_cv:.1f}%")
    print()

    best_cv = min(c10_cv, c15_cv, c2_cv)
    best_k_for_cv = "1.0" if best_cv == c10_cv else ("1.5" if best_cv == c15_cv else "2.0")
    print(f"  ✓ k={best_k_for_cv} 系数最稳定 (CV {best_cv:.1f}%)")

    T_large = 1e8
    sum_k10_full = zeta_zero_sum(0.50, T_large)
    sum_k15_full = zeta_zero_sum(0.75, T_large)
    sum_k2_full  = zeta_zero_sum(1.0,  T_large)

    S_pred_k10 = math.exp(-c10_mean * sum_k10_full)
    S_pred_k15 = math.exp(-c15_mean * sum_k15_full)
    S_pred_k2  = math.exp(-c2_mean * sum_k2_full)

    print()
    print(f"  ── S_infty 外推 (T_max={T_large:.0e}) ──")
    print(f"  k=1.0: S_infty = exp(-{c10_mean:.4f}*{sum_k10_full:.6f}) = {S_pred_k10:.6f}")
    print(f"  k=1.5: S_infty = exp(-{c15_mean:.4f}*{sum_k15_full:.6f}) = {S_pred_k15:.6f}")
    print(f"  k=2.0: S_infty = exp(-{c2_mean:.4f}*{sum_k2_full:.6f}) = {S_pred_k2:.6f}")

print()
print("=" * 70)
print("  结果保存")
print("=" * 70)

S_POISSON_THEORY = 1.0 / math.sqrt(2)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Ns = [r["N"] for r in results]
Ss = [r["S"] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.semilogx(Ns, Ss, 'o-', color='#e74c3c', markersize=8, linewidth=2,
             label='S(N) from segmented sieve')
ax1.axhline(y=S_POISSON_THEORY, color='gray', linestyle='--',
            label=f'Poisson theory: {S_POISSON_THEORY:.4f}')
ax1.set_xlabel('N')
ax1.set_ylabel('S(N)')
ax1.set_title('S(N) Convergence (unscaled v* = pure Li(x))')
ax1.legend()
ax1.grid(True, alpha=0.3)

log_Ns = [math.log10(r["N"]) for r in results]
ax2.plot(log_Ns, c_vals_k10, '^--', color='#95a5a6', markersize=8,
         label=f'c (k=1.0), mean={c10_mean:.4f}')
ax2.plot(log_Ns, c_vals_k15, 'o-', color='#3498db', markersize=8,
         label=f'c (k=1.5), mean={c15_mean:.4f}')
ax2.plot(log_Ns, c_vals_k2, 's--', color='#e67e22', markersize=8,
         label=f'c (k=2.0), mean={c2_mean:.4f}')
ax2.set_xlabel('log10(N)')
ax2.set_ylabel('Coefficient c')
ax2.set_title('Coefficient Stability')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/14_large_scale_SN.png', dpi=150)
plt.close()
print("  图已保存: ../figures/14_large_scale_SN.png")

print()
print("  数据点:")
for r in results:
    print(f"    N={r['N']:>10,d}  S={r['S']:.6f}  T_eff={r['T_eff']:.0f}  "
          f"sum_k1.0={r['sum_k10']:.6f}  sum_k1.5={r['sum_k15']:.6f}")
