from math import sqrt, exp
import numpy

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
            option_prices[state_idx] = disc * (p_up * option_prices[state_idx+1] + p_dn * option_prices[state_idx])
            if is_put:
                option_prices[state_idx] = max(option_prices[state_idx], strike-underlying_values[state_idx])
            else:
                option_prices[state_idx] = max(option_prices[state_idx], underlying_values[state_idx]-strike)
            underlying_values[state_idx] *= u
    final_option_price = option_prices[0]

    return final_option_price


# Monte Carlo pricing for a European put
# random_draws should be an ndarray size [n_draws] by [n_time_steps]
def monte_carlo(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, random_draws):
    n_draws = random_draws.shape[0]
    n_time_steps = random_draws.shape[1]
    b = risk_free_rate - yield_rate
    dt = time_to_expiration / n_time_steps
    drift_term = (b-sigma * sigma / 2)*dt
    sig_sqrt_t = sigma * sqrt(dt)
    spot_adj = spot_price
    spot_coeff = exp(drift_term)
    is_put = (put_or_call.lower() == "put")
    disc = exp(-risk_free_rate * dt)

    # Calculate matrix of underlying values
    underlying_values = numpy.zeros((n_draws,n_time_steps))
    for time_idx in range(n_time_steps):
        spot_adj *= spot_coeff
        for draw_idx in range(n_draws):
            underlying_values[draw_idx][time_idx] = spot_adj * exp(sig_sqrt_t * random_draws[draw_idx][time_idx])

    # Calculate final option payouts
    option_prices = numpy.zeros(n_draws)
    do_exercise = [False] * n_draws
    time_idx = n_time_steps - 1
    for draw_idx in range(n_draws):
        if is_put:
            payout = strike - underlying_values[draw_idx][time_idx]
        else:
            payout = underlying_values[draw_idx][time_idx] - strike
        if payout > 0:
            do_exercise[draw_idx] = True
            option_prices[draw_idx] = payout
        else:
            do_exercise[draw_idx] = False
            option_prices[draw_idx] = 0


    # Step back from maturity, calculating expected future value and comparing to early exercise
    temp_future_underlyings = numpy.zeros(n_draws)
    for t in range(n_time_steps-1):
        time_idx -= 1

        # Sort the future market states as [underlying_value, option_value] pairs
        # We will use this to estimate the expected value of holding for each current state
        sort_future_states = []
        for i in range(n_draws):
            sort_future_states.append([underlying_values[i][time_idx+1],option_prices[i],random_draws[i][time_idx+1],do_exercise[i]])
        if is_put:
            sort_future_states.sort(reverse=True)
        else:
            sort_future_states.sort()
        exercise_at_idx = n_draws
        worthless_at_idx = -1
        for i in range(n_draws):
            if sort_future_states[i][1] == 0:
                worthless_at_idx = i
            if sort_future_states[i][3]:
                exercise_at_idx = i
                break


        # Loop over draws
        for draw_idx in range(n_draws):
            # Calculate early exercise payout
            curr_spot = underlying_values[draw_idx][time_idx]
            if is_put:
                payout = strike - curr_spot
            else:
                payout = curr_spot - strike

            # Generate a local simulation to get the expected value of holding
            for i in range(n_draws):
                temp_future_underlyings[i] = curr_spot * exp(sig_sqrt_t * sort_future_states[i][2])

            # Using sort_future_states as a key, estimate the value of holding for each temp_future_underlyings[i]
            j = 0
            future_option_prices = []
            for i in range(n_draws):
                if is_put:
                    while j < n_draws and -1e-8 < sort_future_states[j][0] - temp_future_underlyings[i]:
                        j += 1
                else:
                    while j < n_draws and -1e-8 < temp_future_underlyings[i] - sort_future_states[j][0]:
                        j += 1
                if j <= worthless_at_idx:
                    continue
                if j >= exercise_at_idx:
                    if is_put:
                        temp_payout = strike - temp_future_underlyings[i]
                    else:
                        temp_payout = temp_future_underlyings[i] - strike
                    future_option_prices.append(temp_payout)
                elif j == 0 or sort_future_states[j][0] - temp_future_underlyings[i] < 1e-8:
                    future_option_prices.append(sort_future_states[j][1])
                else:
                    # Linear interpolate, for now
                    x0 = sort_future_states[j-1][0]
                    x1 = sort_future_states[j][0]
                    y0 = sort_future_states[j-1][1]
                    y1 = sort_future_states[j][1]
                    x = temp_future_underlyings[i]
                    y = y0 + (x-x0) * (y1-y0) / (x1-x0)
                    future_option_prices.append(y)

            # Discount mean for expected value of holding
            ev_hold = disc * sum(future_option_prices) / n_draws

            # Determine whether holding or exercising is optimal for this draw
            if payout > ev_hold - 1e-8:
                do_exercise[draw_idx] = True
                option_prices[draw_idx] = payout
            else:
                do_exercise[draw_idx] = False
                option_prices[draw_idx] = ev_hold

    # Get final value
    if is_put:
        payout = strike - spot_price
    else:
        payout = spot_price - strike
    ev_hold = disc * sum(option_prices) / n_draws
    final_option_price = max(payout, ev_hold)

    return final_option_price

