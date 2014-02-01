# -*- coding: utf-8 -*-

import numpy
import scipy.linalg
from scipy.sparse import csr_matrix


def toeplitz1(N):
    column = numpy.zeros(N)
    column[1] = 0.25
    row = numpy.zeros(N)
    row[1] = 1
    return csr_matrix(scipy.linalg.toeplitz(column, row))
