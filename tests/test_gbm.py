import pytest, numpy
from random import seed, gauss
from math import exp
from model import Model, ModelType, NumericalMethod
from option_enum import OptionType, PutOrCall
from option import Option
from option_util import add_all_evaluation_methods


# Verify closed form against book
# From Haug p3
@pytest.mark.parametrize(('spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration', 'call_price'), (
    (60, 65, 0.08, 0, 0.3, 0.25, 2.1334),
))
def test_gbm_euro_closed_form(spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, call_price):
    put_price = call_price - spot_price * exp(-yield_rate*time_to_expiration) + strike * exp(-risk_free_rate*time_to_expiration)

    model = Model(
        model_type = ModelType.GBM,
        numerical_method = NumericalMethod.CLOSED_FORM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.EUROPEAN,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    option.put_or_call = PutOrCall.CALL
    test_call_price = option.price()
    assert abs(test_call_price - call_price) / call_price < 1e-4

    option.put_or_call = PutOrCall.PUT
    test_put_price = option.price()
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
    model = Model(
        model_type = ModelType.GBM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.EUROPEAN,
        put_or_call = put_or_call,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    model.numerical_method = NumericalMethod.CLOSED_FORM
    test_closed_form = option.price()
    model.numerical_method = NumericalMethod.TREE
    model.n_time_steps = 500
    test_binomial = option.price()
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
    model = Model(
        model_type = ModelType.GBM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.EUROPEAN,
        put_or_call = put_or_call,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    model.numerical_method = NumericalMethod.CLOSED_FORM
    test_closed_form = option.price()

    n_draws = 1000000
    random_draws = numpy.zeros((n_draws, 1))
    seed(12345)
    mirror_idx = n_draws
    time_idx = 0
    for draw_idx in range(int(n_draws/2)):
        mirror_idx -= 1
        random_draws[draw_idx][time_idx] = gauss(0, 1)
        random_draws[mirror_idx][time_idx] = -random_draws[draw_idx][time_idx]
    model.numerical_method = NumericalMethod.MONTE_CARLO
    model.random_draws = random_draws
    test_monte_carlo = option.price()

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
    model = Model(
        model_type = ModelType.GBM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.EUROPEAN,
        put_or_call = put_or_call,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    model.numerical_method = NumericalMethod.CLOSED_FORM
    test_closed_form = option.price()

    model.numerical_method = NumericalMethod.PDE
    model.n_time_steps = 500
    model.n_price_steps = 501
    test_pde = option.price()

    assert abs(test_pde-test_closed_form)/test_closed_form < 1e-3


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
    model = Model(
        model_type = ModelType.GBM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.EUROPEAN,
        put_or_call = put_or_call,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    model.numerical_method = NumericalMethod.TREE
    model.n_time_steps = 2500
    test_binomial = option.price()

    model.numerical_method = NumericalMethod.PDE
    model.n_time_steps = 1000
    model.n_price_steps = 501
    test_pde = option.price()

    assert abs(test_pde-test_binomial)/test_binomial < 1e-3

