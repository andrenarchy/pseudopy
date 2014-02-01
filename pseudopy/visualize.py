# -*- coding: utf-8 -*-

from matplotlib import pyplot
from matplotlib.ticker import LogFormatterMathtext

import numpy


def _contour(Z, levels, mode, triang=None, X=None, Y=None,
             spectrum=None,
             contour_labels=True,
             axes_labels=True
             ):
    # plot spectrum?
    if spectrum is not None:
        pyplot.plot(numpy.real(spectrum), numpy.imag(spectrum), 'o')

    # plot pseudospectrum contour
    if mode == 'meshgrid':
        contour = pyplot.contour(X, Y, Z, levels=levels,
                                 colors=pyplot.rcParams['axes.color_cycle'])
    elif mode == 'triang':
        contour = pyplot.tricontour(triang, Z, levels=levels)

    # plot contour labels?
    if contour_labels:
        pyplot.clabel(contour, inline=1,
                      fmt=LogFormatterMathtext())

    # plot axes labels?
    if axes_labels:
        pyplot.xlabel('Real part')
        pyplot.ylabel('Imaginary part')

    return contour


def contour_meshgrid(X, Y, Z, levels, **kwargs):
    return _contour(Z, levels, mode='meshgrid', X=X, Y=Y, **kwargs)


def contour_triang(triang, Z, levels, **kwargs):
    return _contour(Z, levels, mode='triang', triang=triang, **kwargs)
