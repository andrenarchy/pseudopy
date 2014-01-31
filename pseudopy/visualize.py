from matplotlib import pyplot
import numpy
from matplotlib.tri import Triangulation
from . import compute


def visualize(A, 
              real_min=-1, real_max=1, real_n=50,
              imag_min=-1, imag_max=1, imag_n=50,
              levels=None
              ):
    real = numpy.linspace(real_min, real_max, real_n)
    imag = numpy.linspace(imag_min, imag_max, real_n)

    x, y = numpy.meshgrid(real, imag)
    x = x.flatten()
    y = y.flatten()

    vals = compute.evaluate_points(A, x+1j*y)

    triang = Triangulation(x, y)
    pyplot.tricontour(triang, vals, levels=levels)

    pyplot.show()


