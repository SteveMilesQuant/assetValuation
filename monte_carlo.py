import numpy
from math import sqrt, exp
from model import Model, ModelType
from option import Option
from option_enum import OptionType, PutOrCall


# Monte Carlo option pricing
def monte_carlo(model: Model, option: Option):
    # Model inputs
    model_type = model.model_type
    risk_free_rate = model.risk_free_rate
    yield_rate = model.yield_rate
    sigma = model.sigma
    random_draws = model.random_draws
    assert model_type == ModelType.GBM, f'Error: in monte_carlo, model_type={model_type} should be ModelType.GBM.'

    # Contract inputs
    option_type = option.option_type
    put_or_call = option.put_or_call
    spot_price = option.spot_value
    time_to_expiration = option.time_to_expiration
    strike = option.strike
    assert option_type == OptionType.EUROPEAN, f'Error: in monte_carlo, option_type={option_type} should be OptionType.EUROPEAN.'
    assert spot_price > 0, f'Error: in monte_carlo, spot_price={spot_price} should be positive.'
    assert strike > 0, f'Error: in monte_carlo, strike={strike} should be positive.'

    n_draws = random_draws.shape[0]
    b = risk_free_rate - yield_rate
    drift_term = (b-sigma * sigma / 2)*time_to_expiration
    sig_sqrt_t = sigma * sqrt(time_to_expiration)
    spot_adj = spot_price * exp(drift_term)

    # Calculate final underlying values and option payouts
    underlying_values = numpy.zeros(n_draws)
    option_prices = numpy.zeros(n_draws)
    time_idx = 0 # TODO: for non-European options, allow for a time step
    for draw_idx in range(n_draws):
        underlying_values[draw_idx] = spot_adj * exp(sig_sqrt_t * random_draws[draw_idx][time_idx])
        if put_or_call == PutOrCall.PUT:
            option_prices[draw_idx] = max(strike-underlying_values[draw_idx], 0)
        else:
            option_prices[draw_idx] = max(underlying_values[draw_idx]-strike, 0)

    # Take mean of payouts and discount to time zero
    final_option_price = sum(option_prices) / n_draws
    final_option_price *= exp(-risk_free_rate*time_to_expiration)

    return final_option_price

