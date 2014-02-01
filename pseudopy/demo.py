# -*- coding: utf-8 -*-

import numpy
from scipy.linalg import toeplitz, eigvals
from scipy.sparse import csr_matrix
from matplotlib import pyplot
from . import compute, visualize


def toeplitz1(N):
    column = numpy.zeros(N)
    column[1] = 0.25
    row = numpy.zeros(N)
    row[1] = 1
    return csr_matrix(toeplitz(column, row))


def grcar(N, k=3):
    column = numpy.zeros(N)
    column[0:2] = [1, -1]
    row = numpy.zeros(N)
    row[0:k+1] = 1
    return csr_matrix(toeplitz(column, row))


def grcar_demo():
    A = grcar(32).todense()

    # compute pseudospectrum
    X, Y, Z = compute.evaluate_meshgrid(A,
                                        real_min=-1, real_max=3, real_n=400,
                                        imag_min=-3.5, imag_max=3.5, imag_n=400)

    # plot pseudospectrum
    visualize.contour_meshgrid(X, Y, Z,
                               levels=[10**k for k in range(-4, 0)],
                               spectrum=eigvals(A))
    pyplot.show()
