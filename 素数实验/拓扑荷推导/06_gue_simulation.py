import numpy as np
import math, statistics, time

np.random.seed(42)

N_ZEROS = 200
N_SAMPLES = 80
K = 50

print("=" * 70)
print("  GUE 间距扰动实验: 检验 repulsion 对 Σ 1/|ρ|^k 的影响")
print("=" * 70)
print(f"  N_zeros={N_ZEROS}  N_samples={N_SAMPLES}")
print()

def zeta_count(T):
    if T <= 0: return 0.0
    return T / (2 * math.pi) * math.log(T / (2 * math.pi * math.e)) + 7.0/8.0

def zeta_zero_position(n):
    lo, hi = 2.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        if zeta_count(mid) > n:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2

base_zeros = np.array([zeta_zero_position(n) for n in range(1, N_ZEROS + 1)])
base_spacings = np.diff(base_zeros)
mean_base_spacing = np.mean(base_spacings)

def generate_gue_spacings(n_eigs):
    H_real = np.random.normal(0, 1, (n_eigs, n_eigs))
    H_imag = np.random.normal(0, 1, (n_eigs, n_eigs))
    H = np.zeros((n_eigs, n_eigs), dtype=complex)
    for i in range(n_eigs):
        H[i, i] = np.random.normal(0, 1)
    for i in range(n_eigs):
        for j in range(i + 1, n_eigs):
            val = (H_real[i, j] + 1j * H_imag[i, j]) / math.sqrt(2)
            H[i, j] = val
            H[j, i] = np.conj(val)
    eigs = np.sort(np.linalg.eigvalsh(H))
    return np.diff(eigs)

print("方案 A: 用 GUE 间距替换 zeta 相邻间距 (保留 zeta 位置骨架)")
print("-" * 50)

gue_eig_count = N_ZEROS + 10

start = time.time()
gue_replacement_zeros = []

for s in range(N_SAMPLES):
    gue_sp = generate_gue_spacings(gue_eig_count)
    gue_sp = gue_sp[:N_ZEROS - 1]
    gue_sp_mean = np.mean(gue_sp)
    if gue_sp_mean > 0:
        gue_sp = gue_sp / gue_sp_mean * mean_base_spacing

    new_zeros = np.zeros(N_ZEROS)
    new_zeros[0] = base_zeros[0]
    for i in range(1, N_ZEROS):
        new_zeros[i] = new_zeros[i-1] + gue_sp[i-1]

    gue_replacement_zeros.append(new_zeros)

print("  Σ 1/|ρ|^k 比较:")
for k_label, k in [("k=1.0", 1.0), ("k=1.5", 1.5), ("k=2.0", 2.0),
                    ("k=1.3", 1.3), ("k=1.7", 1.7)]:
    power = k / 2.0
    zeta_sum = np.sum(2.0 / (0.25 + base_zeros**2)**power)
    gue_vals = [np.sum(2.0 / (0.25 + gz**2)**power) for gz in gue_replacement_zeros]
    gue_mean = statistics.mean(gue_vals)
    gue_sd = statistics.stdev(gue_vals)
    ratio = gue_mean / zeta_sum
    print(f"    {k_label:<6s}  zeta={zeta_sum:.6f}  GUE={gue_mean:.6f}±{gue_sd:.6f}  "
          f"ratio={ratio:.4f}")

print(f"\n  k 扫描 (使 |GUE/ζeta - 1| 最小):")
k_scan = np.linspace(0.8, 2.4, 33)
best_k, best_dev = None, float('inf')
for k in k_scan:
    power = k / 2.0
    zs = np.sum(2.0 / (0.25 + base_zeros**2)**power)
    gs = [np.sum(2.0 / (0.25 + gz**2)**power) for gz in gue_replacement_zeros]
    gm = statistics.mean(gs)
    dev = abs(math.log(gm / zs)) if gm > 0 and zs > 0 else float('inf')
    if dev < best_dev:
        best_dev = dev
        best_k = k
    if 1.2 <= k <= 1.8 and abs(k - round(k*4)/4) < 0.001:
        print(f"    k={k:.2f}  GUE={gm:.6f}  zeta={zs:.6f}  ratio={gm/zs:.4f}  "
              f"|log(r)|={dev:.4f}")

print(f"\n  最优 k = {best_k:.3f} (GUE/ζeta ratio = {math.exp(best_dev):.4f})")

print(f"\n  方案 A 的 S 统计:")
T_LO, T_HI = base_zeros[0] - 2, base_zeros[-1] + 10
bin_edges = np.linspace(T_LO, T_HI, K + 1)
ideal_per_bin = np.array([zeta_count(b) - zeta_count(a) for a, b in
                          zip(bin_edges[:-1], bin_edges[1:])])

nz = N_ZEROS
scale_z = nz / (zeta_count(T_HI) - zeta_count(T_LO))
SP_z = []
for i in range(K):
    obs = np.sum((base_zeros >= bin_edges[i]) & (base_zeros < bin_edges[i+1]))
    v_star = ideal_per_bin[i] * scale_z
    sigma = math.sqrt(v_star) if v_star > 0 else 1.0
    SP_z.append(math.exp(-(obs - v_star)**2 / (2 * sigma**2)))
S_zeta_base = statistics.mean(SP_z) / (1 + statistics.variance(SP_z) if K > 1 else 1.0)

S_A_vals = []
for gz in gue_replacement_zeros:
    ng = len(gz)
    scale_g = ng / (zeta_count(T_HI) - zeta_count(T_LO))
    SP_g = []
    for i in range(K):
        obs = np.sum((gz >= bin_edges[i]) & (gz < bin_edges[i+1]))
        v_star = ideal_per_bin[i] * scale_g
        sigma = math.sqrt(v_star) if v_star > 0 else 1.0
        SP_g.append(math.exp(-(obs - v_star)**2 / (2 * sigma**2)))
    v = statistics.variance(SP_g) if K > 1 else 0.0
    S_A_vals.append(statistics.mean(SP_g) / (1 + v))

S_A = statistics.mean(S_A_vals)
S_A_sd = statistics.stdev(S_A_vals)
S_POISSON = 1.0 / math.sqrt(2)

print(f"    基准 zeta S  = {S_zeta_base:.6f}")
print(f"    GUE 间距 S   = {S_A:.6f} ± {S_A_sd:.6f}")
print(f"    Poisson 理论  = {S_POISSON:.6f}")
print(f"    elapsed: {time.time()-start:.0f}s")

print()
print("方案 B: 间距分布统计比较")
print("-" * 50)

all_gue_spacings_raw = []
for _ in range(50):
    sp = generate_gue_spacings(gue_eig_count)
    sp_mean = np.mean(sp)
    if sp_mean > 0:
        sp = sp / sp_mean
    all_gue_spacings_raw.extend(sp)

zeta_sp_norm = base_spacings / mean_base_spacing

print(f"  zeta 归一化间距: mean={np.mean(zeta_sp_norm):.4f}  var={np.var(zeta_sp_norm):.4f}")
print(f"  GUE  归一化间距: mean={np.mean(all_gue_spacings_raw):.4f}  "
      f"var={np.var(all_gue_spacings_raw):.4f}")
print(f"  Poisson 理论: var=1.0")
print(f"  GUE 理论:     var≈0.422 (Wigner surmise)")

print()
print("方案 C: zeta 零点求和 — 解析 vs 渐近 vs 完整积分")
print("-" * 50)

def zeta_zero_sum_by_integral(k, T_min=14.0, T_max=1e6, n_steps=50000):
    power = k / 2.0
    dT = (T_max - T_min) / n_steps
    total = 0.0
    for i in range(n_steps):
        T = T_min + (i + 0.5) * dT
        dens = math.log(T / (2 * math.pi)) / (2 * math.pi)
        if dens < 0:
            dens = 0
        total += dens * 2.0 / (0.25 + T**2)**power * dT
    return total

sum_integral = {}
for k in [1.0, 1.5, 2.0, 1.3, 1.7]:
    sum_integral[k] = zeta_zero_sum_by_integral(k)

for k in [1.0, 1.5, 2.0]:
    power = k / 2.0
    finite = np.sum(2.0 / (0.25 + base_zeros**2)**power)
    T_cut = base_zeros[-1]
    tail = zeta_zero_sum_by_integral(k, T_min=T_cut, T_max=1e6, n_steps=20000)
    full = finite + tail
    print(f"  k={k:.1f}: finite(N={N_ZEROS})={finite:.6f}  tail_int={tail:.6f}  "
          f"full={full:.6f}  ∞-integral={sum_integral[k]:.6f}")

print()
print("=" * 70)
print("  结论")
print("=" * 70)
print(f"  间距方案: GUE 间距替换 zeta 间距")
print(f"    最优 k = {best_k:.3f}")
if abs(best_k - 1.5) < 0.15:
    print(f"    ✓ k ≈ 1.5 = 3/2, 间距 repulsion 确定最优幂次为 3/2!")
elif best_k > 1.5:
    print(f"    → k > 1.5, repulsion 效应偏好更高幂次")
else:
    print(f"    → k < 1.5, repulsion 效应偏好更低幂次")
print()
print(f"  S 值: zeta={S_zeta_base:.4f}  GUE_spacing={S_A:.4f}  Poisson={S_POISSON:.4f}")
