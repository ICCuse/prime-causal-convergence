import math
import mpmath as mp

mp.mp.dps = 30

print("=" * 72)
print("  zeta 零点求和 -> S_infty 解析预测 (高速版)")
print("=" * 72)
print()

mp.mp.dps = 50

gamma_1 = float(mp.im(mp.zetazero(1)))
print(f"  gamma_1 = {gamma_1:.10f}  (mpmath)")
print()

def N_prime(T):
    log_term = mp.log(T / (2 * mp.pi * mp.e))
    return max(0, log_term / (2 * mp.pi))

def integrand(T):
    return N_prime(T) * 2 / (mp.mpf('0.25') + T**2)

ranges = [
    (gamma_1, 100, 20000),
    (100, 1000, 5000),
    (1000, 10000, 2000),
    (10000, 100000, 1000),
    (100000, 1e6, 500),
    (1e6, 1e8, 200),
    (1e8, 1e12, 100),
]

integral_total = mp.mpf('0')
for lo, hi, steps in ranges:
    dT = (hi - lo) / steps
    for i in range(steps):
        T_mid = lo + (i + 0.5) * dT
        integral_total += integrand(T_mid) * dT

print(f"  数值积分 (gamma_1 到 1e12):")
print(f"    integral N'(T) * 2/(1/4+T^2) dT = {float(integral_total):.10f}")

T_end = 1e12
tail_int = (math.log(T_end / (2 * math.pi * math.e)) + 1) / (math.pi * T_end)
print(f"    tail (T > 1e12):    {tail_int:.12e}")

total_from_integral = float(integral_total) + tail_int
print(f"    total integral:     {total_from_integral:.10f}")
print()

S_pred_integral = math.exp(-0.5 * total_from_integral)
print(f"  S_infty = exp(-1/2 * {total_from_integral:.6f}) = {S_pred_integral:.10f}")
print()

print(f"  {'-'*60}")
print(f"  方案C: 前 200 个零点精确 + 积分续接")
print(f"  {'-'*60}")

exact_sum = mp.mpf('0')
for n in range(1, 201):
    rho_n = mp.zetazero(n)
    gamma_n = float(mp.im(rho_n))
    exact_sum += 2.0 / (0.25 + gamma_n**2)

gamma_200 = float(mp.im(mp.zetazero(200)))

integral_from_200 = mp.mpf('0')
ranges2 = [
    (gamma_200, 500, 5000),
    (500, 5000, 3000),
    (5000, 50000, 2000),
    (50000, 5e5, 1000),
    (5e5, 5e7, 500),
    (5e7, 5e10, 200),
]
for lo, hi, steps in ranges2:
    dT = (hi - lo) / steps
    for i in range(steps):
        T_mid = lo + (i + 0.5) * dT
        integral_from_200 += integrand(T_mid) * dT

tail2 = (math.log(5e10 / (2 * math.pi * math.e)) + 1) / (math.pi * 5e10)

total_C = float(exact_sum + integral_from_200) + tail2
S_pred_C = math.exp(-0.5 * total_C)

print(f"  Exact sum (n=1..200):          {float(exact_sum):.10f}")
print(f"  Integral (gamma_200..5e10):    {float(integral_from_200):.10f}")
print(f"  Tail (T > 5e10):               {tail2:.12e}")
print(f"  Total sum 1/|rho|^2:           {total_C:.10f}")
print(f"  S_infty = exp(-1/2 * sum) =    {S_pred_C:.10f}")
print()

S_num = 0.748854
print(f"  {'='*60}")
print(f"  S_infty (方案A, 纯积分):   {S_pred_integral:.6f}")
print(f"  S_infty (方案C, 200零点+积分): {S_pred_C:.6f}")
print(f"  S_infty (数值, 16点拟合):  {S_num:.6f}")
print(f"  {'='*60}")
print(f"  方案A delta: {abs(S_pred_integral - S_num):.6f}  ({abs(S_pred_integral - S_num)/S_num*100:.2f}%)")
print(f"  方案C delta: {abs(S_pred_C - S_num):.6f}  ({abs(S_pred_C - S_num)/S_num*100:.2f}%)")
print()

print(f"  {'-'*60}")
print(f"  如果公式需要缩放因子...")
for alpha in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
    Sp = math.exp(-alpha * total_from_integral)
    m = " <--" if abs(Sp - S_num) < 0.01 else ""
    print(f"    alpha={alpha:.2f}  S={Sp:.6f}{m}")

if abs(S_pred_integral - S_num) < 0.05 or abs(S_pred_C - S_num) < 0.05:
    print(f"\n  *** 预测与数值结果一致! ***")
elif abs(S_pred_integral - S_num) < 0.15:
    print(f"\n  *** 量级匹配, 需要修正因子 ***")
else:
    print(f"\n  *** 解析形式可能需要重新检验 ***")
