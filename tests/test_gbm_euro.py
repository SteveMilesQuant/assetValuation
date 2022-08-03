import pytest
import gbm.euro
from random import seed, gauss
from math import exp
import numpy

# Verify closed form against book
# From Haug p3
@pytest.mark.parametrize(('spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration', 'call_price'), (
    (60, 65, 0.08, 0, 0.3, 0.25, 2.1334),
))
def test_gbm_euro_closed_form(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, call_price):
    put_price = call_price - spot_price * exp(-yield_rate*time_to_expiration) + strike * exp(-risk_free_rate*time_to_expiration)

    test_call_price = gbm.euro.black_scholes_merton('call', spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    assert abs(test_call_price - call_price) / call_price < 1e-4

    test_put_price = gbm.euro.black_scholes_merton('put', spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    assert abs(test_put_price - put_price) / put_price < 1e-4

# Use closed form to verify binomial
@pytest.mark.parametrize(('put_or_call', 'spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration'), (
    ('put', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('put', 100, 120, 0.08, 0.01, 0.3, 1),
    ('put', 100, 80, 0.08, 0.02, 0.33, 2),
    ('call', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('call', 100, 120, 0.08, 0.01, 0.3, 1),
    ('call', 100, 80, 0.08, 0.02, 0.33, 2),
))
def test_gbm_euro_binomial(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    n_time_steps = 500

    test_closed_form = gbm.euro.black_scholes_merton(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    test_binomial = gbm.euro.binomial(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, n_time_steps)
    assert abs(test_binomial-test_closed_form)/test_closed_form < 1e-3

# Use closed form to verify Monte Carlo
@pytest.mark.parametrize(('put_or_call', 'spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration'), (
    ('put', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('put', 100, 120, 0.08, 0.01, 0.3, 1),
    ('put', 100, 80, 0.08, 0.02, 0.33, 2),
    ('call', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('call', 100, 120, 0.08, 0.01, 0.3, 1),
    ('call', 100, 80, 0.08, 0.02, 0.33, 2),
))
def test_gbm_euro_monte_carlo(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    n_draws = 1000000

    test_closed_form = gbm.euro.black_scholes_merton(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    pseudoRandomDraws = numpy.zeros(n_draws)
    seed(12345)
    mirror_idx = n_draws
    for draw_idx in range(int(n_draws/2)):
        mirror_idx -= 1
        pseudoRandomDraws[draw_idx] = gauss(0, 1)
        pseudoRandomDraws[mirror_idx] = -pseudoRandomDraws[draw_idx]
    test_monte_carlo = gbm.euro.monte_carlo(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, pseudoRandomDraws)
    assert abs(test_monte_carlo-test_closed_form)/test_closed_form < 2e-3


# Use closed form to verify PDE
@pytest.mark.parametrize(('put_or_call', 'spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration'), (
    ('put', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('put', 100, 120, 0.08, 0.01, 0.3, 1),
    ('put', 100, 80, 0.08, 0.02, 0.33, 2),
    ('call', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('call', 100, 120, 0.08, 0.01, 0.3, 1),
    ('call', 100, 80, 0.08, 0.02, 0.33, 2),
))
def test_gbm_euro_pde(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    n_time_steps = 500
    n_price_steps = 501

    test_closed_form = gbm.euro.black_scholes_merton(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    test_pde = gbm.euro.pde(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, n_price_steps, n_time_steps)
    assert abs(test_pde-test_closed_form)/test_closed_form < 1e-3

