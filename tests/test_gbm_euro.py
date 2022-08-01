import pytest
from gbm.euro import euro_put_closed_form, euro_put_binomial, euro_put_monte_carlo
from random import seed, gauss
from math import exp

def test_gbm_euro_closed_form():
    # From Haug page 3
    spot_price = 60
    strike = 65
    risk_free_rate = 0.08
    yield_rate = 0
    sigma = 0.3
    time_to_expiration = 0.25
    call_price = 2.1334

    put_price = call_price - spot_price * exp(-yield_rate*time_to_expiration) + strike * exp(-risk_free_rate*time_to_expiration)
    test_closed_form = euro_put_closed_form(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    rel_diff = abs(test_closed_form - put_price) / put_price

    assert rel_diff < 1e-5

# Use closed form to verify binomial
@pytest.mark.parametrize(('spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration'), (
    (60, 65, 0.08, 0.01, 0.2, 0.25),
    (100, 120, 0.08, 0.01, 0.3, 1),
    (100, 80, 0.08, 0.02, 0.33, 2),
))
def test_gbm_euro_binomial(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    test_closed_form = euro_put_closed_form(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    test_binomial = euro_put_binomial(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, 5000)
    rel_diff = abs(test_binomial-test_closed_form)/test_closed_form
    assert rel_diff < 1e-4

# Use closed form to verify Monte Carlo
@pytest.mark.parametrize(('spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration'), (
    (60, 65, 0.08, 0.01, 0.2, 0.25),
    (100, 120, 0.08, 0.01, 0.3, 1),
    (100, 80, 0.08, 0.02, 0.33, 2),
))
def test_gbm_euro_monte_carlo(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    test_closed_form = euro_put_closed_form(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration)
    pseudoRandomDraws = [None] * 1000000
    seed(12345)
    for draw_idx in range(len(pseudoRandomDraws)):
        pseudoRandomDraws[draw_idx] = gauss(0, 1)
    test_monte_carlo = euro_put_monte_carlo(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, pseudoRandomDraws)
    rel_diff = abs(test_monte_carlo-test_closed_form)/test_closed_form
    assert rel_diff < 1e-3

