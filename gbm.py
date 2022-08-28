from math import log, sqrt, exp
from scipy.stats import norm
from model import Model, ModelType
from option_enum import OptionType, PutOrCall, BarrierTypeInOrOut, BarrierTypeUpOrDown
from option import Option


# Black-Scholes-Merton formula for a European option
def euro_black_scholes_merton(model: Model, option: Option):
    # Model inputs
    risk_free_rate = model.risk_free_rate
    yield_rate = model.yield_rate
    sigma = model.sigma
    model_type = model.model_type
    assert model_type == ModelType.GBM, f'Error: in gbm_binomial_tree, model_type={model_type} should be ModelType.GBM.'

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


# Helper class, internal only
class _GBMStandardBarrierHelper:
    def __init__(self, spot_price: float, strike: float, barrier: float, cash_rebate: float, time_to_expiration: float, risk_free_rate: float, yield_rate: float, sigma: float):
        self._strike = strike
        self._cash_rebate = cash_rebate
        self._disc_factor = exp(-risk_free_rate * time_to_expiration)
        self.phi = 1.0
        self.eta = 1.0

        self._barr_over_spot = barrier / spot_price
        sigma_sq = sigma * sigma
        self._sig_sqrt_t = sigma * sqrt(time_to_expiration)
        b = risk_free_rate - yield_rate
        self._mu = b / sigma_sq - 0.5
        self._lambda = sqrt(self._mu * self._mu + 2.0 * risk_free_rate / sigma_sq)
        self._z = log(self._barr_over_spot) / self._sig_sqrt_t + self._lambda * self._sig_sqrt_t
        mu_addend = (1 + self._mu) * self._sig_sqrt_t

        self._spot_term = spot_price * exp(-yield_rate * time_to_expiration)
        self._strike_term = strike * self._disc_factor
        self._barrier_term_2 = (self._barr_over_spot) ** (2.0 * self._mu)
        self._barrier_term_1 = self._barrier_term_2 * self._barr_over_spot * self._barr_over_spot
        self._x1 = log(spot_price / strike) / self._sig_sqrt_t + mu_addend
        self._x2 = log(1.0/self._barr_over_spot) / self._sig_sqrt_t + mu_addend
        self._y1 = log(self._barr_over_spot * barrier / strike) / self._sig_sqrt_t + mu_addend
        self._y2 = log(self._barr_over_spot) / self._sig_sqrt_t + mu_addend

    def A(self) -> float:
        term_1 = self._spot_term * norm.cdf(self.phi * self._x1)
        term_2 = self._strike_term * norm.cdf(self.phi * (self._x1 - self._sig_sqrt_t))
        a_value = self.phi * (term_1 - term_2)
        return a_value

    def B(self) -> float:
        term_1 = self._spot_term * norm.cdf(self.phi * self._x2)
        term_2 = self._strike_term * norm.cdf(self.phi * (self._x2 - self._sig_sqrt_t))
        b_value = self.phi * (term_1 - term_2)
        return b_value

    def C(self) -> float:
        term_1 = self._spot_term * self._barrier_term_1 * norm.cdf(self.eta * self._y1)
        term_2 = self._strike_term * self._barrier_term_2 * norm.cdf(self.eta * ( self._y1 - self._sig_sqrt_t))
        c_value = self.phi * (term_1 - term_2)
        return c_value

    def D(self) -> float:
        term_1 = self._spot_term * self._barrier_term_1 * norm.cdf(self.eta * self._y2)
        term_2 = self._strike_term * self._barrier_term_2 * norm.cdf(self.eta * ( self._y2 - self._sig_sqrt_t))
        d_value = self.phi * (term_1 - term_2)
        return d_value

    def E(self) -> float:
        if self._cash_rebate == 0:
            return 0
        term_1 = norm.cdf(self.eta * (self._x2 - self._sig_sqrt_t))
        term_2 = self._barrier_term_2 * norm.cdf(self.eta * (self._y2 - self._sig_sqrt_t))
        e_value = self._cash_rebate * self._disc_factor * (term_1 - term_2)
        return e_value

    def F(self) -> float:
        if self._cash_rebate == 0:
            return 0
        term_1 = self._barr_over_spot ** (self._mu + self._lambda) * norm.cdf(self.eta * self._z)
        term_2 = self._barr_over_spot ** (self._mu - self._lambda) * norm.cdf(self.eta * (self._z - 2.0 * self._lambda * self._sig_sqrt_t))
        f_value = self._cash_rebate * (term_1 + term_2)
        return f_value


 # From Haug, Reiner and Rubinstein standard barrier option pricing
def barrier_reiner_rubinstein(model: Model, option: Option):
    # Model inputs
    risk_free_rate = model.risk_free_rate
    yield_rate = model.yield_rate
    sigma = model.sigma
    model_type = model.model_type
    assert model_type == ModelType.GBM, f'Error: in gbm_binomial_tree, model_type={model_type} should be ModelType.GBM.'

    # Contract inputs
    option_type = option.option_type
    put_or_call = option.put_or_call
    spot_price = option.spot_value
    time_to_expiration = option.time_to_expiration
    strike = option.strike
    barrier = option.barrier
    cash_rebate = option.cash_rebate
    assert spot_price > 0, f'Error: in barrier_reiner_rubinstein, spot_price={spot_price} should be positive.'
    assert strike > 0, f'Error: in barrier_reiner_rubinstein, strike={strike} should be positive.'
    assert barrier > 0, f'Error: in barrier_reiner_rubinstein, barrier={barrier} should be positive.'
    up_or_down, in_or_out = option.barrier_type

    # Check if we already hit the barrier (or more accurately, if we are currently across it)
    if up_or_down == BarrierTypeUpOrDown.DOWN and spot_price <= barrier:
        barrier_hit = True
    elif up_or_down == BarrierTypeUpOrDown.UP and spot_price >= barrier:
        barrier_hit = True
    else:
        barrier_hit = False
    if barrier_hit:
        if in_or_out == BarrierTypeInOrOut.OUT:
            return cash_rebate
        else:
            return euro_black_scholes_merton(model, option)

    # Barrier not hit - continue with Reiner and Rubinstein price
    barrier_helper = _GBMStandardBarrierHelper(spot_price, strike, barrier, cash_rebate, time_to_expiration, risk_free_rate, yield_rate, sigma)
    if in_or_out == BarrierTypeInOrOut.IN:
        if put_or_call == PutOrCall.CALL:
            if up_or_down == BarrierTypeUpOrDown.DOWN:
                barrier_helper.phi = 1.0
                barrier_helper.eta = 1.0
                if strike >= barrier:
                    option_price = barrier_helper.C() + barrier_helper.E()
                else:
                    option_price = barrier_helper.A() - barrier_helper.B() + barrier_helper.D() + barrier_helper.E()
            else:
                barrier_helper.phi = 1.0
                barrier_helper.eta = -1.0
                if strike >= barrier:
                    option_price = barrier_helper.A() + barrier_helper.E()
                else:
                    option_price = barrier_helper.B() - barrier_helper.C() + barrier_helper.D() + barrier_helper.E()
        else:
            if up_or_down == BarrierTypeUpOrDown.DOWN:
                barrier_helper.phi = -1.0
                barrier_helper.eta = 1.0
                if strike > barrier:
                    option_price = barrier_helper.B() - barrier_helper.C() + barrier_helper.D() + barrier_helper.E()
                else:
                    option_price = barrier_helper.A() + barrier_helper.E()
            else:
                barrier_helper.phi = -1.0
                barrier_helper.eta = -1.0
                if strike > barrier:
                    option_price = barrier_helper.A() - barrier_helper.B() + barrier_helper.D() + barrier_helper.E()
                else:
                    option_price = barrier_helper.C() + barrier_helper.E()
    else:
        if put_or_call == PutOrCall.CALL:
            if up_or_down == BarrierTypeUpOrDown.DOWN:
                barrier_helper.phi = 1.0
                barrier_helper.eta = 1.0
                if strike >= barrier:
                    option_price = barrier_helper.A() - barrier_helper.C() + barrier_helper.F()
                else:
                    option_price = barrier_helper.B() - barrier_helper.D() + barrier_helper.F()
            else:
                barrier_helper.phi = 1.0
                barrier_helper.eta = -1.0
                if strike >= barrier:
                    option_price = barrier_helper.F()
                else:
                    option_price = barrier_helper.A() - barrier_helper.B() + barrier_helper.C() - barrier_helper.D() + barrier_helper.F()
        else:
            if up_or_down == BarrierTypeUpOrDown.DOWN:
                barrier_helper.phi = -1.0
                barrier_helper.eta = 1.0
                if strike > barrier:
                    option_price = barrier_helper.A() - barrier_helper.B() + barrier_helper.C() - barrier_helper.D() + barrier_helper.F()
                else:
                    option_price = barrier_helper.F()
            else:
                barrier_helper.phi = -1.0
                barrier_helper.eta = -1.0
                if strike > barrier:
                    option_price = barrier_helper.B() - barrier_helper.D() + barrier_helper.F()
                else:
                    option_price = barrier_helper.A() - barrier_helper.C() + barrier_helper.F()

    return option_price

