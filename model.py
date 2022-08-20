import numpy
from enum import Enum


class ModelType(Enum):
    GBM = 0


class NumericalMethod(Enum):
    CLOSED_FORM = 0
    TREE = 1
    PDE = 2
    MONTE_CARLO = 3


class Model:
    def __init__( self,
            model_type=ModelType.GBM,
            numerical_method=NumericalMethod.CLOSED_FORM,
            risk_free_rate = 0.0,
            yield_rate = 0.0,
            sigma = 0.0,
            n_time_steps = 100.0,
            n_value_steps = 100.0,
            random_draws=None ):
        self._model_type = model_type
        self._numerical_method = numerical_method
        self._risk_free_rate = risk_free_rate
        self._yield_rate = yield_rate
        self._sigma = sigma
        self._n_time_steps = n_time_steps
        self._n_value_steps = n_value_steps
        self._random_draws = random_draws

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, model_type: ModelType):
        self._model_type = model_type

    @property
    def numerical_method(self):
        return self._numerical_method

    @numerical_method.setter
    def numerical_method(self, numerical_method: NumericalMethod):
        self._numerical_method = numerical_method

    @property
    def risk_free_rate(self):
        return self._risk_free_rate

    @risk_free_rate.setter
    def risk_free_rate(self, risk_free_rate: float):
        assert not numpy.isnan(risk_free_rate), 'Error: in Model class, risk_free_rate must be a number.'
        self._risk_free_rate = risk_free_rate

    @property
    def yield_rate(self):
        return self._yield_rate

    @yield_rate.setter
    def yield_rate(self, yield_rate: float):
        assert not numpy.isnan(yield_rate), 'Error: in Model class, yield_rate must be a number.'
        self._yield_rate = yield_rate

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float):
        assert not numpy.isnan(sigma) and sigma >= 0, 'Error: in Model class, sigma must be a non-negative number.'
        self._sigma = sigma

    @property
    def n_time_steps(self):
        return self._n_time_steps

    @n_time_steps.setter
    def n_time_steps(self, n_time_steps: float):
        assert not numpy.isnan(n_time_steps) and n_time_steps > 0, 'Error: in Model class, n_time_steps must be a positive number.'
        self._n_time_steps = n_time_steps

    @property
    def n_value_steps(self):
        return self._n_value_steps

    @n_value_steps.setter
    def n_value_steps(self, n_value_steps: float):
        assert not numpy.isnan(n_value_steps) and n_value_steps > 0, 'Error: in Model class, n_value_steps must be a positive number.'
        self._n_value_steps = n_value_steps
        
    @property
    def random_draws(self):
        return self._random_draws

    @random_draws.setter
    def random_draws(self, random_draws: numpy.ndarray):
        assert random_draws.ndim == 2, 'Error: in Model class, random_draws must be an ndarray with 2 dimensions (draws by time steps).'
        self._random_draws = random_draws


