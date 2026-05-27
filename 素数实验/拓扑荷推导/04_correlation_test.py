import math
import mpmath as mp

mp.mp.dps = 50
S_NUM = 0.748854

print("=" * 72)
print("  S_infty 推导: 为什么系数是 6.57 而非 0.5?")
print("=" * 72)
print()

sum_inv_rho2 = 0.04404553
alpha_from_num = -math.log(S_NUM) / sum_inv_rho2
print(f"  sum 1/|rho|^2   = {sum_inv_rho2:.8f}")
print(f"  -ln(0.748854)   = {-math.log(S_NUM):.8f}")
print(f"  alpha = -ln(S)/sum = {alpha_from_num:.4f}")
print(f"  理论预测 alpha = 0.5")
print(f"  比值 = {alpha_from_num/0.5:.2f}x")
print()

print(f"  {'─'*60}")
print(f"  零点间距分布 vs 理论 Poisson 间距")
print(f"  {'─'*60}")

print("  获取前 200 个零点...")
gammas = []
for n in range(1, 201):
    gammas.append(float(mp.im(mp.zetazero(n))))

spacings = [gammas[i+1] - gammas[i] for i in range(len(gammas)-1)]
mean_spacing = sum(spacings) / len(spacings)
normalized_spacings = [s / mean_spacing for s in spacings]

small_spacings = sum(1 for s in normalized_spacings if s < 0.5) / len(normalized_spacings)
poisson_small = 1 - math.exp(-0.5)

print(f"  总零点数: {len(gammas)}")
print(f"  平均间距:  {mean_spacing:.6f}")
print(f"  P(spacing < 0.5*mean): 实测={small_spacings:.3f}  Poisson={poisson_small:.3f}")
print(f"  排斥程度: 零点间距比 Poisson 预期少 {(poisson_small-small_spacings)/poisson_small*100:.0f}% 的小间距事件")
print()

print(f"  {'─'*60}")
print(f"  模拟: 独立零点 vs 真实零点的 Z_i^2 分布差异")
print(f"  {'─'*60}")

import random
random.seed(42)

N_test = 1000000
delta = 200
num_intervals = N_test // delta

print(f"  N={N_test}, 区间大小={delta}, 区间数={num_intervals}")

nz = 50
use_gammas = gammas[:nz]

def pi_minus_li(x):
    total = 0.0
    log_x = math.log(x)
    if log_x == 0:
        return 0.0
    sqrt_x = math.sqrt(x)
    for gamma_n in use_gammas:
        rho = 0.5 + 1j * gamma_n
        amplitude = sqrt_x / (gamma_n * log_x)
        phase = gamma_n * log_x
        total += -2.0 * amplitude * math.cos(phase - math.pi/2)
    return total

def li_approx(x):
    if x <= 2: return 0.0
    log_x = math.log(x)
    return x / log_x * (1 + 1/log_x + 2/(log_x**2))

real_Z = []
for interval_idx in range(num_intervals):
    a = 2 + interval_idx * delta
    b = min(N_test, a + delta)
    mid = (a + b) / 2
    obs = li_approx(b) - li_approx(a) + pi_minus_li(b) - pi_minus_li(a)
    ideal = (b/math.log(b) if b > 2 else 0) - (a/math.log(a) if a > 2 else 0)
    sigma = math.sqrt(ideal) if ideal > 0 else 1.0
    real_Z.append((obs - ideal) / sigma)

indep_Z = []
for _ in range(num_intervals):
    z = 0.0
    for gamma_n in use_gammas:
        phase = random.random() * 2 * math.pi
        a = 2 + random.uniform(0, N_test)
        log_a = math.log(a) if a > 2 else 1.0
        amplitude = math.sqrt(a) / (gamma_n * log_a)
        amp_norm = amplitude / math.sqrt(delta / log_a)
        z += amp_norm * math.cos(phase)
    indep_Z.append(z)

var_real = sum(z*z for z in real_Z) / len(real_Z)
var_indep = sum(z*z for z in indep_Z) / len(indep_Z)

print(f"  E[Z^2] (真实零点): {var_real:.6f}")
print(f"  E[Z^2] (独立相位): {var_indep:.6f}")
print(f"  比值 (真实/独立): {var_real/var_indep:.2f}x")

S_from_real = math.exp(-var_real / 2)
S_from_indep = math.exp(-var_indep / 2)
print(f"  S ≈ exp(-E[Z^2]/2) (真实): {S_from_real:.6f}")
print(f"  S ≈ exp(-E[Z^2]/2) (独立): {S_from_indep:.6f}")
print(f"  S 数值 (16点拟合):       {S_NUM:.6f}")
print()

correlation_factor = var_real / var_indep if var_indep > 0 else float('inf')
original_alpha = 0.5
adjusted_alpha = original_alpha * correlation_factor

print(f"  ┌──────────────────────────────────────────────────────┐")
print(f"  │ 结论                                                 │")
print(f"  ├──────────────────────────────────────────────────────┤")
print(f"  │ 零点间相位相关性: {correlation_factor:.2f}x 放大        ")
print(f"  │ 原始系数: α = 0.5                                    ")
print(f"  │ 相关性修正后: α ≈ {adjusted_alpha:.3f}                ")
print(f"  │ 实际需要: α = {alpha_from_num:.3f}                    ")
print(f"  │                                                      ")
print(f"  │ 相关性解释了 {(adjusted_alpha/original_alpha-1)/(alpha_from_num/original_alpha-1)*100:.0f}% 的偏差")
print(f"  │ 但仍有 {(alpha_from_num-adjusted_alpha)/alpha_from_num*100:.0f}% 未被解释")
print(f"  └──────────────────────────────────────────────────────┘")
