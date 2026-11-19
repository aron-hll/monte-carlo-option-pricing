# ===== Imports =====
from pricing import (
    black_scholes_call, black_scholes_put, simulate_terminal_prices,
    discounted_put_payoffs, antithetic_call_payoffs,
    mc_call_price, mc_put_price, mc_call_price_antithetic,
    confidence_interval, convergence_study, convergence_study_antithetic,
    bs_call_delta, bs_call_vega, mc_call_delta, mc_call_vega,
    mc_call_greeks_pathwise,
)

import matplotlib.pyplot as plt
import numpy as np

# ===== Parameters =====
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1
n_sims = 10000

bs_call = black_scholes_call(S0, K, r, sigma, T)

# ===== Coverage test (call) =====
hits = 0
for seed in range(1000):
    price, std_error = mc_call_price(S0, K, r, sigma, T, n_sims, seed)

    lower, upper = confidence_interval(price, std_error)

    if lower < bs_call < upper:
        hits += 1

print()
print("Coverage test, call (95% confidence interval)")
print("-" * 47)
print(f"{'Hits':<20} {hits:>6} / 1000  ({hits / 1000:.1%})")

# ===== Convergence study =====
n_list = np.array([10, 100, 1000, 10000, 100000, 1000000, 10000000])
prices, errors, abs_errors = convergence_study(S0, K, r, sigma, T, n_list, 42)
prices_anti, errors_anti, abs_errors_anti = convergence_study_antithetic(S0, K, r, sigma, T, n_list, 42)

print()
print("Convergence study")
print(f"{'N':>10} {'Price':>10} {'Std. error':>12} {'Abs. error':>12}")
print("-" * 47)
for n, price, std_error, abs_error in zip(n_list, prices, errors, abs_errors):
    print(f"{n:>10,} {price:>10.4f} {std_error:>12.4f} {abs_error:>12.4f}")

# ===== Convergence rate (slope in the log-log plot) =====
slope_all, _ = np.polyfit(np.log(n_list), np.log(errors), 1)
slope_from_1000, _ = np.polyfit(np.log(n_list[2:]), np.log(errors[2:]), 1)

print()
print("Slope in the log-log plot (theory: -0.5)")
print("-" * 47)
print(f"{'all N':<20} {slope_all:>8.3f}")
print(f"{'N >= 1,000':<20} {slope_from_1000:>8.3f}")

# ===== Put option =====
test_prices = np.array([113.88, 84.37, 131.00])
bs_put = black_scholes_put(S0, K, r, sigma, T)
put_payoffs = discounted_put_payoffs(test_prices, K, r, T)
mc_put, mc_put_error = mc_put_price(S0, K, r, sigma, T, 100000, 42)

print()
print("Put option")
print("-" * 55)
print(f"{'Test':<32} {'Result':>10} {'Expected':>10}")
print(f"{'Black-Scholes put':<32} {bs_put:>10.4f} {5.5735:>10.4f}")
for spot, payoff, expected in zip(test_prices, put_payoffs, [0, 14.8677, 0]):
    label = f"Payoff at S_T = {spot:.2f}"
    print(f"{label:<32} {payoff:>10.4f} {expected:>10.4f}")
print(f"{'MC put (100,000, seed 42)':<32} {mc_put:>10.4f} {5.5735:>10.4f}")
print(f"{'MC put, std. error':<32} {mc_put_error:>10.4f} {0.0275:>10.4f}")

# ===== Put-call parity =====
mc_call, _ = mc_call_price(S0, K, r, sigma, T, 100000, 42)

diff_bs = bs_call - bs_put
diff_mc = mc_call - mc_put
theory = S0 - K * np.exp(-r * T)

terminal_prices = simulate_terminal_prices(S0, r, sigma, T, 100000, 42)
mean_check = np.exp(-r * T) * np.mean(terminal_prices) - K * np.exp(-r * T)

mc_call_seed1, _ = mc_call_price(S0, K, r, sigma, T, 100000, 1)
mc_put_seed2, _ = mc_put_price(S0, K, r, sigma, T, 100000, 2)
diff_mc_other_seeds = mc_call_seed1 - mc_put_seed2

print()
print("Put-call parity: C - P")
print("-" * 47)
print(f"{'Theory S0 - K e^(-rT)':<32} {theory:>12.4f}")
print(f"{'Black-Scholes':<32} {diff_bs:>12.4f}")
print(f"{'MC, same seed':<32} {diff_mc:>12.4f}")
print(f"{'e^(-rT) * mean(S_T) - K e^(-rT)':<32} {mean_check:>12.4f}")
print(f"{'Difference of the last two':<32} {diff_mc - mean_check:>12.2e}")
print(f"{'MC, different seeds':<32} {diff_mc_other_seeds:>12.4f}")

# ===== Variance reduction: antithetic variates =====
n_pairs = 100000

price_anti, error_anti = mc_call_price_antithetic(S0, K, r, sigma, T, n_pairs, 42)
price_plain, error_plain = mc_call_price(S0, K, r, sigma, T, n_pairs, 42)
price_fair, error_fair = mc_call_price(S0, K, r, sigma, T, 2 * n_pairs, 42)

X_plus, X_minus = antithetic_call_payoffs(S0, K, r, sigma, T, n_pairs, 42)
Y = (X_plus + X_minus) / 2
rho = np.corrcoef(X_plus, X_minus)[0, 1]
reduction = 1 - error_anti / error_fair

hits_anti = 0
for seed in range(1000):
    price, std_error = mc_call_price_antithetic(S0, K, r, sigma, T, 5000, seed)
    lower, upper = confidence_interval(price, std_error)
    if lower < bs_call < upper:
        hits_anti += 1

print()
print(f"Antithetic variates (Black-Scholes: {bs_call:.4f})")
print("-" * 55)
print(f"{'Method':<30} {'Price':>10} {'Std. error':>12}")
print(f"{'plain, 100,000':<30} {price_plain:>10.4f} {error_plain:>12.4f}")
print(f"{'plain, 200,000 (fair)':<30} {price_fair:>10.4f} {error_fair:>12.4f}")
print(f"{'antithetic, 100,000 pairs':<30} {price_anti:>10.4f} {error_anti:>12.4f}")
print()
print(f"{'Std. dev. of pair means Y':<30} {np.std(Y, ddof=1):>10.4f}")
print(f"{'Correlation rho':<30} {rho:>10.4f}")
print(f"{'Std. error reduction (fair)':<30} {reduction:>10.1%}")
print(f"{'Coverage (1000 runs)':<30} {hits_anti:>6} / 1000  ({hits_anti / 1000:.1%})")

# ===== Greeks: Black-Scholes formulas =====
delta_bs = bs_call_delta(S0, K, r, sigma, T)
vega_bs = bs_call_vega(S0, K, r, sigma, T)

delta_high = bs_call_delta(150, K, r, sigma, T)
delta_low = bs_call_delta(50, K, r, sigma, T)

h_S = 0.01
delta_fd = (black_scholes_call(S0 + h_S, K, r, sigma, T)
            - black_scholes_call(S0 - h_S, K, r, sigma, T)) / (2 * h_S)

h_sigma = 0.0001
vega_fd = (black_scholes_call(S0, K, r, sigma + h_sigma, T)
           - black_scholes_call(S0, K, r, sigma - h_sigma, T)) / (2 * h_sigma)

print()
print("Greeks: Black-Scholes formulas")
print("-" * 55)
print(f"{'Test':<32} {'Result':>10} {'Expected':>10}")
print(f"{'Black-Scholes call':<32} {bs_call:>10.4f} {10.4506:>10.4f}")
print(f"{'Black-Scholes put':<32} {bs_put:>10.4f} {5.5735:>10.4f}")
print(f"{'Delta':<32} {delta_bs:>10.4f} {0.6368:>10.4f}")
print(f"{'Vega':<32} {vega_bs:>10.4f} {37.5240:>10.4f}")
print(f"{'Delta at S0 = 150':<32} {delta_high:>10.4f} {'close to 1':>10}")
print(f"{'Delta at S0 = 50':<32} {delta_low:>10.4f} {'close to 0':>10}")
print(f"{'Delta by finite differences':<32} {delta_fd:>10.4f} {0.6368:>10.4f}")
print(f"{'Vega by finite differences':<32} {vega_fd:>10.4f} {37.5240:>10.4f}")

# ===== Greeks: Monte Carlo with finite differences =====
n_test = 100000
seed_test = 42
h_S = 1.0
h_sigma = 0.01

delta_crn = mc_call_delta(S0, K, r, sigma, T, n_test, h_S, seed_test)
vega_crn = mc_call_vega(S0, K, r, sigma, T, n_test, h_sigma, seed_test)

print()
print("Greeks: Monte Carlo with finite differences")
print("-" * 55)
print(f"{'Method':<32} {'Delta':>10} {'Vega':>10}")
print(f"{'Formula':<32} {delta_bs:>10.4f} {vega_bs:>10.4f}")
print(f"{'MC, same seed':<32} {delta_crn:>10.4f} {vega_crn:>10.4f}")

seed_pairs = [(1, 2), (3, 4), (5, 6)]
for seed_a, seed_b in seed_pairs:
    price_up, _ = mc_call_price(S0 + h_S, K, r, sigma, T, n_test, seed_a)
    price_down, _ = mc_call_price(S0 - h_S, K, r, sigma, T, n_test, seed_b)
    delta_other_seeds = (price_up - price_down) / (2 * h_S)

    price_up, _ = mc_call_price(S0, K, r, sigma + h_sigma, T, n_test, seed_a)
    price_down, _ = mc_call_price(S0, K, r, sigma - h_sigma, T, n_test, seed_b)
    vega_other_seeds = (price_up - price_down) / (2 * h_sigma)

    label = f"MC, seeds {seed_a} and {seed_b}"
    print(f"{label:<32} {delta_other_seeds:>10.4f} {vega_other_seeds:>10.4f}")

print()
print("Delta with the same seed for different h")
print("-" * 55)
print(f"{'h':>10} {'Delta':>10} {'Deviation':>12}")
for h in [10, 1, 0.1, 0.01]:
    delta_h = mc_call_delta(S0, K, r, sigma, T, n_test, h, seed_test)
    print(f"{h:>10} {delta_h:>10.4f} {delta_h - delta_bs:>12.4f}")

# ===== Greeks: Monte Carlo with the pathwise method =====
delta_pw, delta_pw_error, vega_pw, vega_pw_error = mc_call_greeks_pathwise(S0, K, r, sigma, T, n_test, seed_test)

delta_lower, delta_upper = confidence_interval(delta_pw, delta_pw_error)
vega_lower, vega_upper = confidence_interval(vega_pw, vega_pw_error)

delta_in_ci = "yes" if delta_lower < delta_bs < delta_upper else "no"
vega_in_ci = "yes" if vega_lower < vega_bs < vega_upper else "no"

print()
print("Greeks: comparison of all methods")
print("-" * 55)
print(f"{'Method':<32} {'Delta':>10} {'Vega':>10}")
print(f"{'Formula (Black-Scholes)':<32} {delta_bs:>10.4f} {vega_bs:>10.4f}")
print(f"{'Finite differences, same seed':<32} {delta_crn:>10.4f} {vega_crn:>10.4f}")
print(f"{'Pathwise':<32} {delta_pw:>10.4f} {vega_pw:>10.4f}")
print(f"{'Pathwise, std. error':<32} {delta_pw_error:>10.4f} {vega_pw_error:>10.4f}")
print()
print(f"{'95% CI Delta':<32} [{delta_lower:.4f}, {delta_upper:.4f}]  contains formula: {delta_in_ci}")
print(f"{'95% CI Vega':<32} [{vega_lower:.4f}, {vega_upper:.4f}]  contains formula: {vega_in_ci}")

# ===== Plot: convergence of the estimators =====
plt.figure()
plt.loglog(n_list, errors, "s-", label="Standard error")
plt.loglog(n_list, errors_anti, "o-", label="Standard error (antithetic)")
plt.loglog(n_list, abs_errors, "x-", label="Absolute error")
plt.loglog(n_list, 14.7 / np.sqrt(n_list), "--", label="Reference ~ 1/√N")
plt.title("Convergence of the Monte Carlo estimator")
plt.xlabel("Number of simulations N")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.savefig("convergence.png")
plt.show()
