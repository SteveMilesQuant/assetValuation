from enum import Enum


class OptionType(Enum):
    EUROPEAN = 0
    AMERICAN = 1
    BARRIER = 2


class PutOrCall(Enum):
    PUT = 0
    CALL = 1


class BarrierTypeInOrOut(Enum):
    IN = 0
    OUT = 1


class BarrierTypeUpOrDown(Enum):
    UP = 0
    DOWN = 1

