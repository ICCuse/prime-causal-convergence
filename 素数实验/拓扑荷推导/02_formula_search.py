import math
import mpmath as mp

mp.mp.dps = 50

print("获取前 30 个 zeta 零点...")
gammas = []
for n in range(1, 31):
    rho = mp.zetazero(n)
    gammas.append(float(mp.im(rho)))

gamma_30 = gammas[-1]
print(f"  gamma_1..gamma_30: {gammas[0]:.4f} .. {gamma_30:.4f}")

S_NUM = 0.748854
S_NUM_OSC = 0.719

sum_inv_rho2 = 0.0
for g in gammas:
    sum_inv_rho2 += 2.0 / (0.25 + g**2)

def N_prime(T):
    return max(0, float(mp.log(T / (2 * mp.pi * mp.e)) / (2 * mp.pi)))

def tail_integral(T0, power):
    total = mp.mpf('0')
    ranges = [(T0, T0*10, 5000), (T0*10, T0*1000, 2000)]
    for lo, hi, steps in ranges:
        dT = (hi - lo) / steps
        for i in range(steps):
            T_mid = lo + (i + 0.5) * dT
            total += N_prime(T_mid) * 2 / (mp.mpf('0.25') + T_mid**2)**(power/2) * dT
    if power == 2:
        tail = (math.log(T0*1000 / (2*math.pi*math.e)) + 1) / (math.pi * T0*1000)
    elif power == 1:
        tail = 2 * math.log(T0*1000/(2*math.pi*math.e)) / (math.pi * math.sqrt(T0*1000))
    else:
        tail = 0
    return float(total) + tail

sum_inv_rho2_full = sum_inv_rho2 + tail_integral(gamma_30, 2)
sum_inv_rho_full = sum(2.0/math.sqrt(0.25+g**2) for g in gammas) + tail_integral(gamma_30, 1)

print(f"\n  sum 1/|rho|^2  = {sum_inv_rho2_full:.8f}")
print(f"  sum 1/|rho|    = {sum_inv_rho_full:.8f}")
print()

print(f"{'='*72}")
print(f"  候选公式检验")
print(f"{'='*72}")
print(f"  {'公式':<55s} {'S预测':>10s} {'delta':>8s}")
print(f"  {'-'*72}")

def test(name, Spred):
    d = abs(Spred - S_NUM)
    m = " ***" if d < 0.02 else (" **" if d < 0.05 else (" *" if d < 0.10 else ""))
    print(f"  {name:<55s} {Spred:>10.6f} {d:>8.6f}{m}")

S1 = math.exp(-0.5 * sum_inv_rho2_full)
test("S = exp(-1/2 * sum 1/|rho|^2)", S1)

S2 = math.exp(-sum_inv_rho2_full)
test("S = exp(-sum 1/|rho|^2)", S2)

S3 = math.exp(-0.5 * sum_inv_rho_full)
test("S = exp(-1/2 * sum 1/|rho|)", S3)

S3b = math.exp(-sum_inv_rho_full)
test("S = exp(-sum 1/|rho|)", S3b)

best_alpha = -2 * math.log(S_NUM) / sum_inv_rho2_full
test(f"S = exp(-alpha * sum 1/|rho|^2)  alpha={best_alpha:.4f}", S_NUM)

best_alpha2 = -math.log(S_NUM) / sum_inv_rho_full
test(f"S = exp(-alpha * sum 1/|rho|)    alpha={best_alpha2:.4f}", S_NUM)

sum_inv_gamma = sum(1.0/g for g in gammas) + 2*math.log(gamma_30/(2*math.pi*math.e))/(math.pi)
test(f"S = exp(-1/2 * sum 1/gamma)", math.exp(-0.5*sum_inv_gamma))
test(f"S = exp(-sum 1/gamma)", math.exp(-sum_inv_gamma))

print(f"\n  {'-'*72}")
print(f"  如果 S_infty = exp(-c * sum 1/|rho|^k), 扫 c 和 k")
print(f"  {'-'*72}")
print()
print(f"  {'k':>5s} {'c':>8s} {'S':>10s} {'delta':>8s}")
print(f"  {'-'*38}")

best_overall = (None, None, None, float('inf'))
for k in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    sum_k = tail_integral(gamma_30, k)
    for g in gammas:
        sum_k += 2.0 / (0.25 + g**2)**(k/2)
    for c in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
        Sp = math.exp(-c * sum_k)
        d = abs(Sp - S_NUM)
        if d < best_overall[3]:
            best_overall = (k, c, Sp, d)

print(f"\n  Best: S = exp(-{best_overall[1]:.4f} * sum 1/|rho|^{best_overall[0]:.1f}) = {best_overall[2]:.6f}")
print(f"  delta vs 0.748854: {best_overall[3]:.6f}")
