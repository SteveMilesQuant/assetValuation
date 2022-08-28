import numpy
from option_enum import OptionType, PutOrCall, BarrierTypeInOrOut, BarrierTypeUpOrDown
from model import ModelType, NumericalMethod, Model

class Option:
    def __init__(self,
            model = Model(),
            option_type = OptionType.EUROPEAN,
            put_or_call = PutOrCall.PUT,
            spot_value = 0.0,
            time_to_expiration = 0.0,
            strike = 0.0,
            barrier = 0.0,
            barrier_type = (BarrierTypeUpOrDown.UP, BarrierTypeInOrOut.IN),
            cash_rebate = 0.0 ):
        self._model = model
        self._option_type = option_type
        self._put_or_call = put_or_call
        self._spot_value = spot_value
        self._time_to_expiration = time_to_expiration
        self._strike = strike
        self._barrier = barrier
        self._barrier_type = barrier_type
        self._cash_rebate = cash_rebate
        self.__evaluation_methods = {}

    @property
    def model(self):
        return self._model

    @property
    def option_type(self):
        return self._option_type

    @option_type.setter
    def option_type(self, option_type: OptionType):
        self._option_type = option_type

    @property
    def put_or_call(self):
        return self._put_or_call

    @put_or_call.setter
    def put_or_call(self, put_or_call: PutOrCall):
        self._put_or_call = put_or_call

    @property
    def spot_value(self):
        return self._spot_value

    @spot_value.setter
    def spot_value(self, spot_value: float):
        assert not numpy.isnan(spot_value), 'Error: in Option class, spot_value must be a number.'
        self._spot_value = spot_value

    @property
    def time_to_expiration(self):
        return self._time_to_expiration

    @time_to_expiration.setter
    def time_to_expiration(self, time_to_expiration: float):
        assert not numpy.isnan(time_to_expiration) and time_to_expiration >= 0, 'Error: in Option class, time_to_expiration must be a non-negative number.'
        self._time_to_expiration = time_to_expiration

    @property
    def strike(self):
        return self._strike

    @strike.setter
    def strike(self, strike: float):
        assert not numpy.isnan(strike), 'Error: in Option class, strike must be a number.'
        self._strike = strike

    @property
    def barrier(self):
        return self._barrier

    @barrier.setter
    def barrier(self, barrier: float):
        assert not numpy.isnan(barrier), 'Error: in Option class, barrier must be a number.'
        self._barrier = barrier
        
    @property
    def barrier_type(self):
        return self._barrier_type

    @barrier_type.setter
    def barrier_type(self, barrier_type: tuple):
        assert len(barrier_type) == 2, 'Error: in Option class, barrier_type must be a tuple length 2 (BarrierTypeUpOrDown, BarrierTypeInOrOut)'
        assert type(barrier_type[0]) ==  BarrierTypeUpOrDown
        assert type(barrier_type[1]) == BarrierTypeInOrOut
        self._barrier_type = barrier_type
        
    @property
    def cash_rebate(self):
        return self._cash_rebate

    @cash_rebate.setter
    def cash_rebate(self, cash_rebate: float):
        assert not numpy.isnan(cash_rebate) and cash_rebate >= 0, 'Error: in Option class, barrier must be a non-negative number.'
        self._cash_rebate = cash_rebate

    def add_evaluation_method(self, eval_method_key : tuple, eval_method):
        assert len(eval_method_key) == 3
        assert type(eval_method_key[0]) == ModelType
        assert type(eval_method_key[1]) == OptionType
        assert type(eval_method_key[2]) == NumericalMethod
        self.__evaluation_methods[eval_method_key] = eval_method

    def price(self):
        model = self._model
        eval_method = self.__evaluation_methods.get((model.model_type, self._option_type, model.numerical_method))
        assert eval_method is not None, f'Evaluation method not found for model_type={model.model_type}, option_type={self._option_type}, and numerical_method={model.numerical_method}.'
        return eval_method(model, self)

