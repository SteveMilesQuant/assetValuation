import numpy
from math import sqrt, exp
from model import Model, ModelType
from option import Option
from option_enum import OptionType, PutOrCall, BarrierTypeInOrOut, BarrierTypeUpOrDown


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
    barrier = option.barrier
    # TODO: support cash_rebate
    up_or_down, in_or_out = option.barrier_type
    assert spot_price > 0, f'Error: in monte_carlo, spot_price={spot_price} should be positive.'
    assert strike > 0, f'Error: in monte_carlo, strike={strike} should be positive.'

    if option_type == OptionType.EUROPEAN:
        n_time_steps = 1
    elif option_type == OptionType.BARRIER:
        n_time_steps = random_draws.shape[1]
        assert n_time_steps > 0, f'Error: in monte_carlo, n_time_steps={n_time_steps} should be a positive number.'
    else:
        assert 1 == 0, f'Error: in monte_carlo, option_type={option_type} should be OptionType.EUROPEAN or OptionType.BARRIER.'
    dt = time_to_expiration / n_time_steps

    n_draws = random_draws.shape[0]
    b = risk_free_rate - yield_rate
    drift_term = (b-sigma * sigma / 2)*dt
    drift_coeff = exp(drift_term)
    sig_sqrt_t = sigma * sqrt(dt)

    # Moving forward through time, calculate underlying value path and evaluate portfolio
    underlying_values = numpy.full(n_draws, float(spot_price))
    option_prices = numpy.zeros(n_draws)
    if option_type == OptionType.BARRIER:
        underlying_extrema = underlying_values.copy()
    for draw_idx in range(n_draws):
        for time_idx in range(n_time_steps):
            underlying_values[draw_idx] *= drift_coeff * exp(sig_sqrt_t * random_draws[draw_idx][time_idx])
            if option_type == OptionType.BARRIER:
                if up_or_down == BarrierTypeUpOrDown.UP:
                    underlying_extrema[draw_idx] = max(underlying_extrema[draw_idx], underlying_values[draw_idx])
                else:
                    underlying_extrema[draw_idx] = min(underlying_extrema[draw_idx], underlying_values[draw_idx])

            # At maturity, evaluate payout
            if time_idx == n_time_steps-1:
                if put_or_call == PutOrCall.PUT:
                    option_prices[draw_idx] = max(strike-underlying_values[draw_idx], 0)
                else:
                    option_prices[draw_idx] = max(underlying_values[draw_idx]-strike, 0)
                if option_type == OptionType.BARRIER:
                    if up_or_down == BarrierTypeUpOrDown.UP and underlying_extrema[draw_idx] >= barrier:
                        barrier_hit = True
                    elif up_or_down == BarrierTypeUpOrDown.DOWN and underlying_extrema[draw_idx] <= barrier:
                        barrier_hit = True
                    else:
                        barrier_hit = False
                    if in_or_out == BarrierTypeInOrOut.OUT and barrier_hit:
                        option_prices[draw_idx] = 0
                    elif in_or_out == BarrierTypeInOrOut.IN and not barrier_hit:
                        option_prices[draw_idx] = 0


    # Take mean of payouts and discount to time zero
    final_option_price = sum(option_prices) / n_draws
    final_option_price *= exp(-risk_free_rate*time_to_expiration)

    return final_option_price

