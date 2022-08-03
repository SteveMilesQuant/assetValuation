from math import sqrt, exp
import numpy
from util.num import tridiag_solve


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
    underlying_values = [None] * (n_time_steps+1)
    option_prices = [None] * (n_time_steps+1)
    for state_idx in range(n_time_steps+1):
        underlying_values[state_idx] = final_underlying
        if is_put:
            option_prices[state_idx] = max(strike-final_underlying, 0)
        else:
            option_prices[state_idx] = max(final_underlying-strike, 0)
        final_underlying *= u_2

    # Moving backwards in time, discount european options from the bottom up
    for time_idx in range(n_time_steps):
        for state_idx in range(n_time_steps-time_idx):
            underlying_values[state_idx] *= u
            option_prices[state_idx] = disc * (p_up * option_prices[state_idx+1] + p_dn * option_prices[state_idx])
            if is_put:
                option_prices[state_idx] = max(option_prices[state_idx], strike-underlying_values[state_idx])
            else:
                option_prices[state_idx] = max(option_prices[state_idx], underlying_values[state_idx]-strike)

    final_option_price = option_prices[0]

    return final_option_price

# Numerical solution to PDE pricing for a European option
# Uses Crank-Nicolson method
def pde(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, n_price_steps, n_time_steps):
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
    is_put = (put_or_call.lower() == "put")

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
    if is_put:
        f[0] = max(strike - S[0], 0)
        f[n_price_steps-1] = max(strike - S[n_price_steps-1], 0)
    else:
        f[0] = max(S[0] - strike, 0)
        f[n_price_steps-1] = max(S[n_price_steps-1] - strike, 0)

    # Set up body of calculations (i.e. body of matrices)
    S_coeff = exp(dx)
    for i in range(1, n_price_steps-1):
        S[i] = S[i-1] * S_coeff
        lhs_a[i] = -matrix_1 + matrix_2
        lhs_b[i] = -matrix_3 - matrix_4
        lhs_c[i] = matrix_1 + matrix_2
        rhs[i][i-1] = matrix_1 - matrix_2
        rhs[i][i] = -matrix_3 + matrix_4
        rhs[i][i+1] = -matrix_1 - matrix_2
        if is_put:
            f[i] = max(strike - S[i], 0)
        else:
            f[i] = max(S[i] - strike, 0)

    # Step through time, solving for f[t] using f[t+1]
    # Update f[t] for early execise
    for time_idx in range(n_time_steps):
        new_rhs = numpy.matmul(rhs, f)
        f = tridiag_solve(lhs_a, lhs_b, lhs_c, new_rhs)
        for i in range(1, n_price_steps-1):
            if is_put:
                f[i] = max(strike - S[i], f[i])
            else:
                f[i] = max(S[i] - strike, f[i])

    return f[half_n_price_steps]

