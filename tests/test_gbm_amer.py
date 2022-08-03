import pytest
import gbm.amer
import gbm.euro
import numpy
from random import seed, gauss

# Verify binomial and Monte Carlo against each other
@pytest.mark.parametrize(('put_or_call', 'spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration'), (
    ('put', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('put', 100, 120, 0.08, 0.01, 0.3, 1),
    ('put', 100, 80, 0.08, 0.02, 0.33, 2),
    ('call', 60, 65, 0.08, 0.01, 0.2, 0.25),
    ('call', 100, 120, 0.08, 0.01, 0.3, 1),
    ('call', 100, 80, 0.08, 0.02, 0.33, 2),
))
def test_gbm_amer_binomial_and_pde(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration):
    n_time_steps_binomial = 2500
    n_time_steps_pde = 1000
    n_price_steps_pde = 501

    test_binomial = gbm.amer.binomial(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, n_time_steps_binomial)
    test_pde = gbm.amer.pde(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, n_price_steps_pde, n_time_steps_pde)
    assert abs(test_pde-test_binomial)/test_binomial < 1e-3

