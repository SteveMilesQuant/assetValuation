from math import log, sqrt, exp
from scipy.stats import norm
from model import Model, ModelType
from option_enum import OptionType, PutOrCall
from option import Option


# Black-Scholes-Merton formula for a European option
def euro_black_scholes_merton(model: Model, option: Option):
    # Model inputs
    risk_free_rate = model.risk_free_rate
    yield_rate = model.yield_rate
    sigma = model.sigma

    # Contract inputs
    put_or_call = option.put_or_call
    spot_price = option.spot_value
    time_to_expiration = option.time_to_expiration
    strike = option.strike
    assert spot_price > 0, f'Error: in euro_black_scholes_merton, spot_price={spot_price} should be positive.'
    assert strike > 0, f'Error: in euro_black_scholes_merton, strike={strike} should be positive.'

    sig_sqrt_t = sigma * sqrt(time_to_expiration);
    d1 = log(spot_price/strike) + (risk_free_rate-yield_rate+sigma*sigma/2)*time_to_expiration
    d1 /= sig_sqrt_t
    d2 = d1 - sig_sqrt_t

    if put_or_call == PutOrCall.PUT:
        final_option_price = strike * exp(-risk_free_rate*time_to_expiration) * norm.cdf(-d2)
        final_option_price -= spot_price * exp(-yield_rate*time_to_expiration) * norm.cdf(-d1)
    else:
        final_option_price = spot_price * exp(-yield_rate*time_to_expiration) * norm.cdf(d1)
        final_option_price -= strike * exp(-risk_free_rate*time_to_expiration) * norm.cdf(d2)

    return final_option_price


# Binomial tree method for evaluating European and American options
def gbm_binomial_tree(model: Model, option: Option):
    # Model inputs
    risk_free_rate = model.risk_free_rate
    yield_rate = model.yield_rate
    sigma = model.sigma
    n_time_steps = model.n_time_steps
    model_type = model.model_type
    assert model_type == ModelType.GBM, f'Error: in gbm_binomial_tree, model_type={model_type} should be ModelType.GBM.'

    # Contract inputs
    option_type = option.option_type
    put_or_call = option.put_or_call
    spot_price = option.spot_value
    time_to_expiration = option.time_to_expiration
    strike = option.strike
    assert spot_price > 0, f'Error: in gbm_binomial_tree, spot_price={spot_price} should be positive.'
    assert strike > 0, f'Error: in gbm_binomial_tree, strike={strike} should be positive.'

    dt = time_to_expiration / n_time_steps
    u = exp(sigma*sqrt(dt))
    u_2 = u * u
    d = 1/u
    disc = exp(-risk_free_rate*dt)
    p_up = (exp((risk_free_rate-yield_rate)*dt) - d)/(u-d)
    p_dn = 1-p_up

    # Generate vector of final underlying values and final option prices
    final_underlying = spot_price * (d ** n_time_steps)
    underlying_values = [None] * (n_time_steps+1)
    option_prices = [None] * (n_time_steps+1)
    for state_idx in range(n_time_steps+1):
        underlying_values[state_idx] = final_underlying
        if put_or_call == PutOrCall.PUT:
            option_prices[state_idx] = max(strike-final_underlying, 0)
        else:
            option_prices[state_idx] = max(final_underlying-strike, 0)
        final_underlying *= u_2

    # Moving backwards in time, discount european options from the bottom up
    for time_idx in range(n_time_steps):
        for state_idx in range(n_time_steps-time_idx):
            underlying_values[state_idx] *= u
            option_prices[state_idx] = disc * (p_up * option_prices[state_idx+1] + p_dn * option_prices[state_idx])
            if option_type == OptionType.AMERICAN:
                if put_or_call == PutOrCall.PUT:
                    option_prices[state_idx] = max(option_prices[state_idx], strike-underlying_values[state_idx])
                else:
                    option_prices[state_idx] = max(option_prices[state_idx], underlying_values[state_idx]-strike)

    final_option_price = option_prices[0]

    return final_option_price


