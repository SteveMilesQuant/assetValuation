from enum import Enum


class OptionType(Enum):
    EUROPEAN = 0
    AMERICAN = 1
    BARRIER = 2


class PutOrCall(Enum):
    PUT = 0
    CALL = 1

