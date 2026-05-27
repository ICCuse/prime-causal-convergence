import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N_data = np.array([
    100_000, 200_000, 500_000,
    1_000_000, 2_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000,
    100_000_000, 200_000_000, 500_000_000,
    1_000_000_000, 2_000_000_000, 5_000_000_000,
    10_000_000_000
])

S_data = np.array([
    0.882086, 0.861785, 0.837151,
    0.825675, 0.809556, 0.800386,
    0.798170, 0.791757, 0.783501,
    0.777618, 0.770859, 0.764888,
    0.760839, 0.756222, 0.752226,
    0.748854
])

lnN = np.log(N_data)
lnlnN = np.log(lnN)


def model_powerlaw(N, S_inf, A, alpha):
    return S_inf + A / (np.log(N) ** alpha)


def model_oscillatory(N, S_inf, A, alpha, B, omega, phi, beta):
    ln_N = np.log(N)
    ln_ln_N = np.log(ln_N)
    trend = S_inf + A / (ln_N ** alpha)
    osc = B * np.cos(omega * ln_ln_N + phi) / (ln_N ** beta)
    return trend + osc


print("=" * 70)
print(" 第一阶段：纯幂律拟合  S(N) = S_∞ + A/(ln N)^α")
print("=" * 70)

p0_A = [0.71, 20.0, 2.0]
popt_A, pcov_A = curve_fit(model_powerlaw, N_data, S_data, p0=p0_A, maxfev=20000)

perr_A = np.sqrt(np.diag(pcov_A))
S_pred_A = model_powerlaw(N_data, *popt_A)
residuals_A = S_data - S_pred_A

ss_res_A = np.sum(residuals_A ** 2)
ss_tot = np.sum((S_data - np.mean(S_data)) ** 2)
r2_A = 1 - ss_res_A / ss_tot

print(f"  S_∞   = {popt_A[0]:.8f}  (± {perr_A[0]:.8f})")
print(f"  A     = {popt_A[1]:.6f}  (± {perr_A[1]:.6f})")
print(f"  α     = {popt_A[2]:.6f}  (± {perr_A[2]:.6f})")
print(f"  R²    = {r2_A:.6f}")
print(f"  MSE   = {np.mean(residuals_A**2):.8f}")

print()
print("=" * 70)
print(" 第二阶段：分析残差振荡，猜测频率参数")
print("=" * 70)

print(f"\n  残差序列 (按 ln(ln N) 排列):")
print(f"  {'N':<16} {'ln(ln N)':<10} {'残差':<12}")
print(f"  " + "-" * 38)
for N, x, r in zip(N_data, lnlnN, residuals_A):
    print(f"  {N:<16,} {x:<10.4f} {r:+.6f}")

abs_res = np.abs(residuals_A)
log_abs_res = np.log(np.maximum(abs_res, 1e-10))
slope, intercept = np.polyfit(lnlnN, log_abs_res, 1)
beta_guess = -slope
B_guess = np.exp(intercept)

print(f"\n  残差包络分析:")
print(f"  斜率 (β 估计): {beta_guess:.4f}")
print(f"  振幅 B 估计:   {B_guess:.6f}")

print()
print("=" * 70)
print(" 第三阶段：振荡拟合")
print("  S(N) = S_∞ + A/(ln N)^α + B·cos(ω·ln(ln N) + φ)/(ln N)^β")
print("=" * 70)

omega_guess = 6.0
phi_guess = 0.5

p0_B = [popt_A[0], popt_A[1], popt_A[2], B_guess, omega_guess, phi_guess, beta_guess]

bounds_lower = [0.6, 0.0, 0.5, 0.0, 3.0, 0.0, 0.0]
bounds_upper = [0.8, 100.0, 4.0, 0.5, 15.0, 6.28, 5.0]

try:
    popt_B, pcov_B = curve_fit(
        model_oscillatory, N_data, S_data,
        p0=p0_B, bounds=(bounds_lower, bounds_upper),
        maxfev=50000, method='trf'
    )
    perr_B = np.sqrt(np.diag(pcov_B))
    S_pred_B = model_oscillatory(N_data, *popt_B)
    residuals_B = S_data - S_pred_B
    ss_res_B = np.sum(residuals_B ** 2)
    r2_B = 1 - ss_res_B / ss_tot

    print(f"\n  最优参数:")
    print(f"  S_∞   = {popt_B[0]:.8f}  (± {perr_B[0]:.8f})")
    print(f"  A     = {popt_B[1]:.6f}  (± {perr_B[1]:.6f})")
    print(f"  α     = {popt_B[2]:.6f}  (± {perr_B[2]:.6f})")
    print(f"  B     = {popt_B[3]:.6f}  (± {perr_B[3]:.6f})")
    print(f"  ω     = {popt_B[4]:.6f}  (± {perr_B[4]:.6f})")
    print(f"  φ     = {popt_B[5]:.6f}  (± {perr_B[5]:.6f})")
    print(f"  β     = {popt_B[6]:.6f}  (± {perr_B[6]:.6f})")
    print(f"  R²    = {r2_B:.6f}")
    print(f"  MSE   = {np.mean(residuals_B**2):.8f}")
    fit_success = True
except Exception as e:
    print(f"\n  全参数拟合失败: {e}")
    print(f"  改用无界拟合...")
    try:
        popt_B, pcov_B = curve_fit(
            model_oscillatory, N_data, S_data,
            p0=p0_B, maxfev=100000
        )
        perr_B = np.sqrt(np.diag(pcov_B))
        S_pred_B = model_oscillatory(N_data, *popt_B)
        residuals_B = S_data - S_pred_B
        ss_res_B = np.sum(residuals_B ** 2)
        r2_B = 1 - ss_res_B / ss_tot
        fit_success = True
        print(f"\n  无界拟合成功!")
        print(f"  S_∞   = {popt_B[0]:.8f}  (± {perr_B[0]:.8f})")
        print(f"  A     = {popt_B[1]:.6f}  (± {perr_B[1]:.6f})")
        print(f"  α     = {popt_B[2]:.6f}  (± {perr_B[2]:.6f})")
        print(f"  B     = {popt_B[3]:.6f}  (± {perr_B[3]:.6f})")
        print(f"  ω     = {popt_B[4]:.6f}  (± {perr_B[4]:.6f})")
        print(f"  φ     = {popt_B[5]:.6f}  (± {perr_B[5]:.6f})")
        print(f"  β     = {popt_B[6]:.6f}  (± {perr_B[6]:.6f})")
        print(f"  R²    = {r2_B:.6f}")
        print(f"  MSE   = {np.mean(residuals_B**2):.8f}")
    except Exception as e2:
        print(f"  无界拟合也失败: {e2}")
        fit_success = False
        popt_B = [popt_A[0], popt_A[1], popt_A[2], 0.001, 6.0, 1.0, 1.0]
        r2_B = r2_A
        residuals_B = residuals_A.copy()

print()
print("=" * 80)
print(" 汇总对比")
print("=" * 80)
print(f"  {'指标':<20} {'纯幂律':<15} {'振荡修正':<15}")
print(f"  " + "-" * 50)
print(f"  {'S_∞':<20} {popt_A[0]:<15.8f} {popt_B[0]:<15.8f}")
print(f"  {'R²':<20} {r2_A:<15.6f} {r2_B:<15.6f}")
print(f"  {'MSE':<20} {np.mean(residuals_A**2):<15.8f} {np.mean(residuals_B**2):<15.8f}")

print()
print("=" * 80)
print(" 逐点对比：实测值 vs 纯幂律 vs 振荡修正")
print("=" * 80)
print(f"  {'N':<16} {'S_实测':<12} {'幂律拟合':<12} {'幂律残差':<12} {'振荡拟合':<12} {'振荡残差':<12}")
print(f"  " + "-" * 76)
for i in range(len(N_data)):
    S_fit_A = S_pred_A[i]
    S_fit_B = model_oscillatory(N_data[i], *popt_B) if fit_success else S_fit_A
    res_A = residuals_A[i]
    res_B = S_data[i] - S_fit_B
    print(f"  {N_data[i]:<16,} {S_data[i]:<12.6f} {S_fit_A:<12.6f} {res_A:+10.6f} {S_fit_B:<12.6f} {res_B:+10.6f}")

print()
print("=" * 80)
print(" 残差符号序列分析")
print("=" * 80)
signs_A = ''.join(['+' if r > 0 else '-' for r in residuals_A])
signs_B = ''.join(['+' if r > 0 else '-' for r in residuals_B]) if fit_success else signs_A
print(f"  纯幂律残差符号: {signs_A}")
print(f"  连续同号分析:")
current = signs_A[0]
run = 1
runs_A = []
for s in signs_A[1:]:
    if s == current:
        run += 1
    else:
        runs_A.append(run)
        current = s
        run = 1
runs_A.append(run)
print(f"  游程长度: {runs_A}  (若随机应混杂, 长游程 = 系统性偏差)")

if fit_success:
    print(f"  振荡修正残差符号: {signs_B}")
    current = signs_B[0]
    run = 1
    runs_B = []
    for s in signs_B[1:]:
        if s == current:
            run += 1
        else:
            runs_B.append(run)
            current = s
            run = 1
    runs_B.append(run)
    print(f"  游程长度: {runs_B}")

print()
print("=" * 80)
print(" 外推预测")
print("=" * 80)
future_N = [1e11, 1e12, 1e13, 1e14, 1e15]
print(f"  {'N':<16} {'纯幂律预测':<14} {'振荡修正预测':<14}")
print(f"  " + "-" * 44)
for Nf in future_N:
    pred_A = model_powerlaw(Nf, *popt_A)
    pred_B = model_oscillatory(Nf, *popt_B) if fit_success else pred_A
    print(f"  {Nf:<16,.0f} {pred_A:<14.8f} {pred_B:<14.8f}")

N_smooth = np.logspace(np.log10(N_data[0]), np.log10(N_data[-1]), 500)
S_smooth_A = model_powerlaw(N_smooth, *popt_A)
S_smooth_B = model_oscillatory(N_smooth, *popt_B) if fit_success else S_smooth_A
trend_smooth = popt_B[0] + popt_B[1] / (np.log(N_smooth) ** popt_B[2]) if fit_success else S_smooth_A
osc_smooth = S_smooth_B - trend_smooth if fit_success else np.zeros_like(S_smooth_A)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

axes[0].semilogx(N_data, S_data, 'o', color='black', markersize=8, label='Data (16 pts)')
axes[0].semilogx(N_smooth, S_smooth_A, '--', color='grey', linewidth=1.5, label=f'Power-law (R²={r2_A:.4f})')
if fit_success:
    axes[0].semilogx(N_smooth, S_smooth_B, '-', color='darkblue', linewidth=2, label=f'Oscillatory (R²={r2_B:.4f})')
axes[0].set_xlabel('N')
axes[0].set_ylabel('S(N)')
axes[0].set_title(f'S(N) Convergence Fit\nS_∞ = {popt_B[0]:.4f}')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].axhline(y=0, color='grey', linestyle='-', alpha=0.3)
axes[1].semilogx(N_data, residuals_A * 1000, 's-', color='grey', markersize=6, label=f'Power-law MSE={np.mean(residuals_A**2):.1e}')
if fit_success:
    axes[1].semilogx(N_data, residuals_B * 1000, 'o-', color='darkblue', markersize=6, label=f'Oscillatory MSE={np.mean(residuals_B**2):.1e}')
axes[1].set_xlabel('N')
axes[1].set_ylabel('Residual (×10⁻³)')
axes[1].set_title('Residuals Comparison')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

if fit_success:
    axes[2].semilogx(N_smooth, trend_smooth, '--', color='coral', linewidth=1.5, label='Trend (S_∞ + A/(ln N)^α)')
    axes[2].semilogx(N_smooth, osc_smooth, '-', color='darkblue', linewidth=1.5, label=f'Oscillation (ω={popt_B[4]:.1f}, β={popt_B[6]:.2f})')
    axes[2].scatter(N_data, np.zeros_like(N_data), color='grey', s=20, zorder=5)
axes[2].set_xlabel('N')
axes[2].set_ylabel('Amplitude')
axes[2].set_title('Oscillation Decomposition in ln(ln N)')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('09_convergence_fit.png', dpi=150)
plt.close()
print("图已保存: 09_convergence_fit.png")
