import pytest
import numpy
from util.num import tridiag_solve, tridiag_inv

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


# Test tridiag_inv
def test_util_tridiag_inv():
    n = 4
    a = numpy.full(n, 1) # index 0 is ignored
    b = numpy.full(n, -2.6)
    c = numpy.full(n-1, 1)

    M = tridiag_inv(a, b, c)

    L = numpy.zeros((n, n))
    for i in range(n):
        if i > 0:
            L[i][i-1] = a[i]
        L[i][i] = b[i]
        if i < n-1:
            L[i][i+1] = c[i]

    I = numpy.matmul(L, M)

    for i in range(n):
        assert abs(I[i,i] - 1) < 1e-8
        for j in range(i):
            assert abs(I[i,j]) < 1e-8
            assert abs(I[j,i]) < 1e-8

