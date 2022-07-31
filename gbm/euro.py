from math import log, sqrt, exp
from scipy.stats import norm
import pytest

# Black-Scholes-Merton formula for a European option
def euro_put(CurrentUnderlyingPrice, StrikePrice, RiskFreeRate, YieldRate, Sigma, TimeToExpiration):
    sigSqrtT = Sigma * sqrt(TimeToExpiration);
    d1 = log(CurrentUnderlyingPrice/StrikePrice) + (RiskFreeRate-YieldRate+Sigma*Sigma/2)*TimeToExpiration
    d1 /= sigSqrtT
    d2 = d1 - sigSqrtT
    euroOptionPrice = StrikePrice * exp(-RiskFreeRate*TimeToExpiration) * norm.cdf(-d2)
    euroOptionPrice -= CurrentUnderlyingPrice * exp(-YieldRate*TimeToExpiration) * norm.cdf(-d1)
    return euroOptionPrice

# Binomial tree algorithm for European option
# Not written optimally for Europeans, but instead as a stepping-stone to American pricing
def euro_put_binomial(CurrentUnderlyingPrice, StrikePrice, RiskFreeRate, YieldRate, Sigma, TimeToExpiration, nTimeSteps):
    dt = TimeToExpiration / nTimeSteps
    u = exp(Sigma*sqrt(dt))
    uSq = u * u
    d = 1/u
    disc = exp(-RiskFreeRate*dt)
    pUp = (exp((RiskFreeRate-YieldRate)*dt) - d)/(u-d)
    pDn = 1-pUp

    # Generate vector of final underlying values and final option prices
    finalUnderlying = CurrentUnderlyingPrice * (d ** nTimeSteps)
    # underlyingValues = [None] * (nTimeSteps+1)
    optionPrices = [0] * (nTimeSteps+1)
    for stateIdx in range(nTimeSteps+1):
        if StrikePrice <= finalUnderlying:
            break
        # underlyingValues[stateIdx] = finalUnderlying
        optionPrices[stateIdx] = StrikePrice-finalUnderlying
        finalUnderlying = finalUnderlying * uSq

    # Moving backwards in time, discount european options from the bottom up
    for timeIdx in range(nTimeSteps):
        for stateIdx in range(nTimeSteps-timeIdx):
            if optionPrices[stateIdx] == 0:
                break
            optionPrices[stateIdx] = disc * (pUp * optionPrices[stateIdx+1] + pDn * optionPrices[stateIdx])
            # underlyingValues[stateIdx] *= u

    return optionPrices[0]

