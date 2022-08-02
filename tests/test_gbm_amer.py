import pytest
import gbm.amer
from random import seed, gauss
from math import floor
import numpy

# Verify binomial and Monte Carlo against each other
@pytest.mark.parametrize(('put_or_call', 'spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration'), (
    ('put', 60, 65, 0.08, 0.01, 0.2, 0.25),
))
def test_gbm_amer_binomial_and_mc(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    n_time_steps_tree = 5000
    n_time_steps_mc = 20
    n_draws = 1000

    test_binomial = gbm.amer.binomial(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, n_time_steps_tree)

    random_draws = numpy.zeros((n_draws,n_time_steps_mc))
    seed(12345)
    mirror_idx = len(random_draws)
    for draw_idx in range(floor(n_draws/2)):
        mirror_idx -= 1
        for time_idx in range(n_time_steps_mc):
            random_draws[draw_idx][time_idx] = gauss(0, 1)
            random_draws[mirror_idx][time_idx] = -random_draws[draw_idx][time_idx]

    test_monte_carlo = gbm.amer.monte_carlo(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, random_draws)

    assert abs(test_monte_carlo-test_binomial)/test_binomial < 1e-3

