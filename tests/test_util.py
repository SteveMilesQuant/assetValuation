import pytest
import numpy
from util import tridiag_solve


# Test tridiag_solve
def test_util_tridiag_solve():
    n = 4
    a = numpy.full(n, 1) # index 0 is ignored
    b = numpy.full(n, -2.6)
    c = numpy.full(n-1, 1)
    d = numpy.zeros(n)
    d[0] = -240
    d[n-1] = -150

    x = tridiag_solve(a, b, c, d)

    for i in range(n):
        sum = 0
        if i > 0:
            sum += a[i] * x[i-1]
        sum += b[i] * x[i]
        if i < n-1:
            sum += c[i] * x[i+1]
        assert abs(sum - d[i]) < 1e-8

