import numpy
from matplotlib import pyplot


class Path(object):
    def __init__(self, vertices):
        self.vertices = numpy.array(vertices)

    def __iter__(self):
        return iter(self.vertices)

    def length(self):
        return numpy.sum(numpy.abs(self.vertices[1:]-self.vertices[:-1]))


class Paths(object):
    def __init__(self, paths=None):
        if paths is None:
            paths = []
        self.paths = paths

    def __iter__(self):
        return iter(self.paths)

    def add(self, path):
        if isinstance(path, list):
            self.paths += path
        else:
            self.paths.append(path)

    def length(self):
        return numpy.sum([path.length() for path in self.paths])


def plot_finish(contours, spectrum=None, contour_labels=True):
    # plot spectrum?
    if spectrum is not None:
        pyplot.plot(numpy.real(spectrum), numpy.imag(spectrum), 'o')

    # plot contour labels?
    from matplotlib.ticker import LogFormatterMathtext
    if contour_labels:
        pyplot.clabel(contours, inline=1, fmt=LogFormatterMathtext())
