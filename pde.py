import numpy
from math import sqrt, exp
from model import Model, ModelType
from option import Option
from option_enum import OptionType, PutOrCall
from util import tridiag_solve


# Numerical solution to PDE pricing for various options
# Uses Crank-Nicolson method
def pde(model: Model, option: Option):
    # Model inputs
    model_type = model.model_type
    risk_free_rate = model.risk_free_rate
    yield_rate = model.yield_rate
    sigma = model.sigma
    n_time_steps = model.n_time_steps
    n_price_steps = model.n_price_steps
    assert model_type == ModelType.GBM, f'Error: in pde, model_type={model_type} should be ModelType.GBM.'

    # Contract inputs
    option_type = option.option_type
    put_or_call = option.put_or_call
    spot_price = option.spot_value
    time_to_expiration = option.time_to_expiration
    strike = option.strike
    assert spot_price > 0, f'Error: in pde, spot_price={spot_price} should be positive.'
    assert strike > 0, f'Error: in pde, strike={strike} should be positive.'

    if  n_price_steps % 2 == 0:
        n_price_steps += 1
    dt = time_to_expiration / n_time_steps
    dx = sigma * sqrt(3 * dt)
    b = risk_free_rate - yield_rate
    sigma_sq = sigma * sigma
    dx_sq = dx * dx
    matrix_1 = (b - sigma_sq / 2) / (4 * dx)
    matrix_2 = sigma_sq / (4 * dx_sq)
    matrix_3 = 1 / dt
    matrix_4 = (sigma_sq / dx_sq + risk_free_rate) / 2

    S = numpy.zeros(n_price_steps)
    f = numpy.zeros(n_price_steps)
    lhs_a = numpy.zeros(n_price_steps) # index 0 is ingnored
    lhs_b = numpy.zeros(n_price_steps)
    lhs_c = numpy.zeros(n_price_steps-1)
    rhs = numpy.zeros((n_price_steps, n_price_steps))

    # Set up boundary conditions
    half_n_price_steps = int(n_price_steps / 2)
    temp = exp(dx * half_n_price_steps) # from Haug p.342-343
    S[0] = spot_price / temp
    S[n_price_steps-1] = spot_price * temp
    lhs_b[0] = lhs_b[n_price_steps-1] = 1
    rhs[0][0] = rhs[n_price_steps-1][n_price_steps-1] = 1
    if put_or_call == PutOrCall.PUT:
        f[0] = max(strike - S[0], 0)
        f[n_price_steps-1] = max(strike - S[n_price_steps-1], 0)
    else:
        f[0] = max(S[0] - strike, 0)
        f[n_price_steps-1] = max(S[n_price_steps-1] - strike, 0)

    # Set up body of calculations (i.e. body of matrices) and maturity exercise value
    S_coeff = exp(dx)
    for i in range(1, n_price_steps-1):
        S[i] = S[i-1] * S_coeff
        lhs_a[i] = -matrix_1 + matrix_2
        lhs_b[i] = -matrix_3 - matrix_4
        lhs_c[i] = matrix_1 + matrix_2
        rhs[i][i-1] = matrix_1 - matrix_2
        rhs[i][i] = -matrix_3 + matrix_4
        rhs[i][i+1] = -matrix_1 - matrix_2
        if put_or_call == PutOrCall.PUT:
            f[i] = max(strike - S[i], 0)
        else:
            f[i] = max(S[i] - strike, 0)

    # Step through time, solving for f[t] using f[t+1]
    # Update f[t] for early execise
    for time_idx in range(n_time_steps):
        new_rhs = numpy.matmul(rhs, f)
        f = tridiag_solve(lhs_a, lhs_b, lhs_c, new_rhs)
        if option_type == OptionType.AMERICAN:
            for i in range(1, n_price_steps-1):
                if put_or_call == PutOrCall.PUT:
                    f[i] = max(strike - S[i], f[i])
                else:
                    f[i] = max(S[i] - strike, f[i])

    return f[half_n_price_steps]


