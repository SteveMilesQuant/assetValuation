import pytest
import gbm.euro
from random import seed, gauss
from math import exp

# Verify closed form against book
# From Haug p3
@pytest.mark.parametrize(('spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration', 'call_price'), (
    (60, 65, 0.08, 0, 0.3, 0.25, 2.1334),
))
def test_gbm_euro_closed_form(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, call_price):
    put_price = call_price - spot_price * exp(-yield_rate*time_to_expiration) + strike * exp(-risk_free_rate*time_to_expiration)

    test_call_price = gbm.euro.black_scholes_merton('call', spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    rel_diff = abs(test_call_price - call_price) / call_price
    assert rel_diff < 1e-4

    test_put_price = gbm.euro.black_scholes_merton('put', spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    rel_diff = abs(test_put_price - put_price) / put_price
    assert rel_diff < 1e-4

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
    test_closed_form = gbm.euro.black_scholes_merton(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    test_binomial = gbm.euro.binomial(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, 5000)
    rel_diff = abs(test_binomial-test_closed_form)/test_closed_form
    assert rel_diff < 1e-4

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
    test_closed_form = gbm.euro.black_scholes_merton(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    pseudoRandomDraws = [None] * 1000000
    seed(12345)
    for draw_idx in range(len(pseudoRandomDraws)):
        pseudoRandomDraws[draw_idx] = gauss(0, 1)
    test_monte_carlo = gbm.euro.monte_carlo(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, pseudoRandomDraws)
    rel_diff = abs(test_monte_carlo-test_closed_form)/test_closed_form
    assert rel_diff < 1e-3

