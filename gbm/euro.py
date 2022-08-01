from math import log, sqrt, exp
from scipy.stats import norm
import pytest

# Black-Scholes-Merton formula for a European put
def black_scholes_merton(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    sig_sqrt_t = sigma * sqrt(time_to_expiration);
    d1 = log(spot_price/strike) + (risk_free_rate-yield_rate+sigma*sigma/2)*time_to_expiration
    d1 /= sig_sqrt_t
    d2 = d1 - sig_sqrt_t
    
    if put_or_call.lower() == "put":
        final_option_price = strike * exp(-risk_free_rate*time_to_expiration) * norm.cdf(-d2)
        final_option_price -= spot_price * exp(-yield_rate*time_to_expiration) * norm.cdf(-d1)
    else:
        final_option_price = spot_price * exp(-yield_rate*time_to_expiration) * norm.cdf(d1)
        final_option_price -= strike * exp(-risk_free_rate*time_to_expiration) * norm.cdf(d2)
        
    return final_option_price

# Binomial tree algorithm for European put
# Not written optimally for Europeans, but instead as a stepping-stone to American pricing
def binomial(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, n_time_steps):
    dt = time_to_expiration / n_time_steps
    u = exp(sigma*sqrt(dt))
    u_2 = u * u
    d = 1/u
    disc = exp(-risk_free_rate*dt)
    p_up = (exp((risk_free_rate-yield_rate)*dt) - d)/(u-d)
    p_dn = 1-p_up
    is_put = (put_or_call.lower() == "put")

    # Generate vector of final underlying values and final option prices
    final_underlying = spot_price * (d ** n_time_steps)
    # underlying_values = [None] * (n_time_steps+1)
    option_prices = [None] * (n_time_steps+1)
    for state_idx in range(n_time_steps+1):
        # underlying_values[state_idx] = final_underlying
        if is_put:
            option_prices[state_idx] = max(strike-final_underlying, 0)
        else:
            option_prices[state_idx] = max(final_underlying-strike, 0)
        final_underlying *= u_2

    # Moving backwards in time, discount european options from the bottom up
    for time_idx in range(n_time_steps):
        for state_idx in range(n_time_steps-time_idx):
            option_prices[state_idx] = disc * (p_up * option_prices[state_idx+1] + p_dn * option_prices[state_idx])
            # underlying_values[state_idx] *= u
    final_option_price = option_prices[0]

    return final_option_price


# Monte Carlo pricing for a European put
# pseudoRandomDraws should be size [n_draws]
def monte_carlo(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, pseudoRandomDraws):
    n_draws = len(pseudoRandomDraws)
    b = risk_free_rate - yield_rate
    drift_term = (b-sigma * sigma / 2)*time_to_expiration
    sig_sqrt_t = sigma * sqrt(time_to_expiration)
    spot_adj = spot_price * exp(drift_term)
    is_put = (put_or_call.lower() == "put")

    # Calculate final underlying values and option payouts
    underlying_values = [None] * n_draws
    option_prices = [None] * n_draws
    for draw_idx in range(n_draws):
        underlying_values[draw_idx] = spot_adj * exp(sig_sqrt_t * pseudoRandomDraws[draw_idx])
        if is_put:
            option_prices[draw_idx] = max(strike-underlying_values[draw_idx], 0)
        else:
            option_prices[draw_idx] = max(underlying_values[draw_idx]-strike, 0)

    # Take mean of payouts and discount to time zero
    final_option_price = sum(option_prices) / n_draws
    final_option_price *= exp(-risk_free_rate*time_to_expiration)

    return final_option_price

