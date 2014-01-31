import numpy
import scipy.linalg

def toeplitz1(N):
    column = numpy.zeros(N)
    column[1] = 0.25
    row = numpy.zeros(N)
    row[1] = 1
    return scipy.linalg.toeplitz(column, row)
