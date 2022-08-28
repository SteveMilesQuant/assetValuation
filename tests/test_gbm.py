import pytest, numpy
from random import seed, gauss
from math import exp
from model import Model, ModelType, NumericalMethod
from option_enum import OptionType, PutOrCall, BarrierTypeUpOrDown, BarrierTypeInOrOut
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

    n_draws = 100000
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


# Verify American binomial and pde against each other
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
        option_type = OptionType.AMERICAN,
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


# Verify closed form barrier against Haug book p. 154
@pytest.mark.parametrize(('put_or_call', 'barrier_type', 'strike', 'barrier', 'sigma', 'haug_price', 'spot_price', 'cash_rebate', 'time_to_expiration', 'risk_free_rate', 'yield_rate'), (
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 90, 95, 0.25, 9.0246, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 100, 95, 0.25, 6.7924, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 110, 95, 0.25, 4.8759, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 90, 100, 0.25, 3.0, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 100, 100, 0.25, 3.0, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 90, 100, 0.25, 3.0, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 90, 105, 0.25, 2.6789, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 100, 105, 0.25, 2.3580, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 110, 105, 0.25, 2.3453, 100, 3, 0.5, 0.08, 0.04),

    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 90, 95, 0.25, 7.7627, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 100, 95, 0.25, 4.0109, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 110, 95, 0.25, 2.0576, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 90, 100, 0.25, 13.8333, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 100, 100, 0.25, 7.8494, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 110, 100, 0.25, 3.9795, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 90, 105, 0.25, 14.1112, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 100, 105, 0.25, 8.4482, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.CALL, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 110, 105, 0.25, 4.5910, 100, 3, 0.5, 0.08, 0.04),

    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 90, 95, 0.25, 2.2798, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 100, 95, 0.25, 2.2947, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 110, 95, 0.25, 2.6252, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 90, 100, 0.25, 3.0, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 100, 100, 0.25, 3.0, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 110, 100, 0.25, 3.0, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 90, 105, 0.25, 3.7760, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 100, 105, 0.25, 5.4932, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 110, 105, 0.25, 7.5187, 100, 3, 0.5, 0.08, 0.04),

    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 90, 95, 0.25, 2.9586, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 100, 95, 0.25, 6.5677, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 110, 95, 0.25, 11.9752, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 90, 100, 0.25, 2.2845, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 100, 100, 0.25, 5.9085, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 110, 100, 0.25, 11.6465, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 90, 105, 0.25, 1.4653, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 100, 105, 0.25, 3.3721, 100, 3, 0.5, 0.08, 0.04),
    (PutOrCall.PUT, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 110, 105, 0.25, 7.0846, 100, 3, 0.5, 0.08, 0.04),
))
def test_gbm_barrier_closed_form(put_or_call, barrier_type, strike, barrier, sigma, haug_price, spot_price, cash_rebate, time_to_expiration, risk_free_rate, yield_rate):
    model = Model(
        model_type = ModelType.GBM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.BARRIER,
        put_or_call = put_or_call,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration,
        barrier = barrier,
        barrier_type = barrier_type,
        cash_rebate = cash_rebate )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    model.numerical_method = NumericalMethod.CLOSED_FORM
    test_closed_form = option.price()

    assert abs(test_closed_form-haug_price)/haug_price < 1e-4


# Verify barrier Monte Carlo against closed form
# TODO: figure out what's wrong with barrier Monte Carlo pricing and then re-enable
@pytest.mark.parametrize(('put_or_call', 'spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration', 'barrier_type', 'barrier'), (
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 65),
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 65),
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 55),
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 55),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 65),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 65),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 55),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 55),
))
def __test_gbm_barrier_monte_carlo(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, barrier_type, barrier):
    model = Model(
        model_type = ModelType.GBM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.BARRIER,
        put_or_call = put_or_call,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration,
        barrier = barrier,
        barrier_type = barrier_type )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    model.numerical_method = NumericalMethod.CLOSED_FORM
    test_closed_form = option.price()

    n_draws = 10000
    random_draws = numpy.zeros((n_draws, 50))
    seed(54321)
    mirror_idx = n_draws
    time_idx = 0
    for draw_idx in range(int(n_draws/2)):
        mirror_idx -= 1
        for time_idx in range(random_draws.shape[1]):
            random_draws[draw_idx][time_idx] = gauss(0, 1)
            random_draws[mirror_idx][time_idx] = -random_draws[draw_idx][time_idx]
    model.random_draws = random_draws
    model.numerical_method = NumericalMethod.MONTE_CARLO
    test_monte_carlo = option.price()

    assert abs(test_monte_carlo-test_closed_form)/test_closed_form < 1e-3


# Verify barrier PDE against closed form
# TODO: figure out what's wrong with barrier PDE pricing and then re-enable
@pytest.mark.parametrize(('put_or_call', 'spot_price', 'strike', 'risk_free_rate', 'yield_rate', 'sigma', 'time_to_expiration', 'barrier_type', 'barrier'), (
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 65),
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 65),
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 55),
    (PutOrCall.PUT, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 55),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.IN), 65),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.UP,BarrierTypeInOrOut.OUT), 65),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.IN), 55),
    (PutOrCall.CALL, 60, 60, 0.08, 0.01, 0.2, 0.25, (BarrierTypeUpOrDown.DOWN,BarrierTypeInOrOut.OUT), 55),
))
def __test_gbm_barrier_pde(put_or_call, spot_price, strike, risk_free_rate, yield_rate, sigma, time_to_expiration, barrier_type, barrier):
    model = Model(
        model_type = ModelType.GBM,
        risk_free_rate = risk_free_rate,
        yield_rate = yield_rate,
        sigma = sigma )

    option = Option(
        model=model,
        option_type = OptionType.BARRIER,
        put_or_call = put_or_call,
        spot_value = spot_price,
        strike = strike,
        time_to_expiration = time_to_expiration,
        barrier = barrier,
        barrier_type = barrier_type )
    add_all_evaluation_methods(option) # always do this (or create your own eval methods and add them)

    model.numerical_method = NumericalMethod.CLOSED_FORM
    test_closed_form = option.price()

    model.numerical_method = NumericalMethod.PDE
    model.n_time_steps = 1000
    model.n_price_steps = 501
    test_pde = option.price()

    assert abs(test_pde-test_closed_form)/test_closed_form < 1e-3

