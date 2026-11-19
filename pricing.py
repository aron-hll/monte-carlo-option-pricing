import numpy as np
from scipy.stats import norm


def d1_d2(S0, K, r, sigma, T):
    """Return d1 and d2 of the Black-Scholes formula."""
    d2 = (np.log(S0 / K) + (r - sigma**2 / 2)*T) / (sigma*np.sqrt(T))
    d1 = (np.log(S0 / K) + (r + sigma**2 / 2)*T) / (sigma*np.sqrt(T))

    return d1, d2


def black_scholes_call(S0, K, r, sigma, T):
    """Black-Scholes price of a European call."""
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    I1 = S0*norm.cdf(d1)
    I2 = K*np.exp(-r*T)*norm.cdf(d2)
    C = I1 - I2
    return C


def simulate_terminal_prices(S0, r, sigma, T, n_sims, seed=None):
    """Simulate n_sims terminal prices S_T under the risk-neutral measure."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    S_T = S0*np.exp((r - sigma**2 / 2)*T + sigma*np.sqrt(T)*Z)

    return S_T


def discounted_call_payoffs(S_T, K, r, T):
    """Discounted call payoffs for given terminal prices."""
    X = np.exp(-r*T)*np.maximum(S_T - K, 0)
    return X


def mc_call_price(S0, K, r, sigma, T, n_sims, seed=None):
    """Monte Carlo call price and its standard error."""
    S_T = simulate_terminal_prices(S0, r, sigma, T, n_sims, seed)
    X = discounted_call_payoffs(S_T, K, r, T)
    price = np.mean(X)
    std_error = np.std(X, ddof=1) / np.sqrt(n_sims)

    return price, std_error


def confidence_interval(price, std_error, z=1.96):
    """Confidence interval around an estimate (default: 95%)."""
    lower = price - z*std_error
    upper = price + z*std_error

    return lower, upper


def convergence_study(S0, K, r, sigma, T, n_list, seed=None):
    """Price, standard error and absolute error for each sample size in n_list."""
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    prices = []
    std_errors = []
    abs_errors = []
    for n in n_list:
        price, std_error = mc_call_price(S0, K, r, sigma, T, n, seed)
        prices.append(price)
        std_errors.append(std_error)
        abs_errors.append(np.abs(price - bs_price))

    return np.array(prices), np.array(std_errors), np.array(abs_errors)


def black_scholes_put(S0, K, r, sigma, T):
    """Black-Scholes price of a European put."""
    d1, d2 = d1_d2(S0, K, r, sigma, T)
    P = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
    return P


def discounted_put_payoffs(S_T, K, r, T):
    """Discounted put payoffs for given terminal prices."""
    X = np.exp(-r*T)*np.maximum(K - S_T, 0)

    return X


def mc_put_price(S0, K, r, sigma, T, n_sims, seed=None):
    """Monte Carlo put price and its standard error."""
    S_T = simulate_terminal_prices(S0, r, sigma, T, n_sims, seed)
    X = discounted_put_payoffs(S_T, K, r, T)
    price = np.mean(X)
    std_error = np.std(X, ddof=1) / np.sqrt(n_sims)

    return price, std_error


def antithetic_call_payoffs(S0, K, r, sigma, T, n_sims, seed=None):
    """Discounted call payoffs for Z and -Z (antithetic pairs)."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    drift = (r - sigma**2 / 2)*T
    S_T_plus = S0*np.exp(drift + sigma*np.sqrt(T)*Z)
    S_T_minus = S0*np.exp(drift - sigma*np.sqrt(T)*Z)
    X_plus = discounted_call_payoffs(S_T_plus, K, r, T)
    X_minus = discounted_call_payoffs(S_T_minus, K, r, T)
    return X_plus, X_minus


def mc_call_price_antithetic(S0, K, r, sigma, T, n_sims, seed=None):
    """Call price with antithetic variates; the error is based on the pair means."""
    X_plus, X_minus = antithetic_call_payoffs(S0, K, r, sigma, T, n_sims, seed)
    Y = (X_plus + X_minus) / 2
    price = np.mean(Y)
    std_error = np.std(Y, ddof=1) / np.sqrt(n_sims)
    return price, std_error


def convergence_study_antithetic(S0, K, r, sigma, T, n_list, seed=None):
    """Like convergence_study, with antithetic variates (n payoffs = n/2 pairs)."""
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    prices = []
    std_errors = []
    abs_errors = []
    for n in n_list:
        price, std_error = mc_call_price_antithetic(S0, K, r, sigma, T, n // 2, seed)
        prices.append(price)
        std_errors.append(std_error)
        abs_errors.append(np.abs(price - bs_price))
    return np.array(prices), np.array(std_errors), np.array(abs_errors)


def bs_call_delta(S0, K, r, sigma, T):
    """Delta of a European call (Black-Scholes)."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    delta = norm.cdf(d1)

    return delta


def bs_call_vega(S0, K, r, sigma, T):
    """Vega of a European call (Black-Scholes)."""
    d1, _ = d1_d2(S0, K, r, sigma, T)
    vega = S0*norm.pdf(d1)*np.sqrt(T)

    return vega


def mc_call_delta(S0, K, r, sigma, T, n_sims, h=1.0, seed=None):
    """Delta by central finite differences (same seed for both prices)."""
    price_up, _ = mc_call_price(S0 + h, K, r, sigma, T, n_sims, seed)
    price_down, _ = mc_call_price(S0 - h, K, r, sigma, T, n_sims, seed)

    delta = (price_up - price_down) / (2*h)

    return delta


def mc_call_vega(S0, K, r, sigma, T, n_sims, h=0.01, seed=None):
    """Vega by central finite differences (same seed for both prices)."""
    price_up, _ = mc_call_price(S0, K, r, sigma + h, T, n_sims, seed)
    price_down, _ = mc_call_price(S0, K, r, sigma - h, T, n_sims, seed)

    vega = (price_up - price_down) / (2*h)

    return vega


def mc_call_greeks_pathwise(S0, K, r, sigma, T, n_sims, seed=None):
    """Delta and Vega by the pathwise method, each with its standard error."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_sims)
    S_T = S0*np.exp((r - sigma**2 / 2)*T + sigma*np.sqrt(T)*Z)
    indicator = S_T > K
    discount = np.exp(-r*T)

    delta_samples = discount * indicator * S_T / S0
    vega_samples = discount * indicator * S_T * (np.sqrt(T)*Z - sigma*T)

    delta = np.mean(delta_samples)
    delta_error = np.std(delta_samples, ddof=1) / np.sqrt(n_sims)

    vega = np.mean(vega_samples)
    vega_error = np.std(vega_samples, ddof=1) / np.sqrt(n_sims)

    return delta, delta_error, vega, vega_error
