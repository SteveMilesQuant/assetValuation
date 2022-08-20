import numpy


# Thomas's algorithm for solving M * x = d, where M is a tridiagonal matrix
# a is indexed 1 to n-1 (ignoring index zero)
# b is indexed 0 to n-1
# c is indexed 1 to n-2 (ignoring index n-1)
def tridiag_solve(a, b, c, d):
    n = len(b)
    c_prime = numpy.zeros(n)
    d_prime = numpy.zeros(n)
    x = numpy.zeros(n)

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i-1]
        if i < n-1:
            c_prime[i] = c[i] / denom
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

    x[n-1] = d_prime[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x

