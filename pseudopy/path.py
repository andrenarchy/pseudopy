import numpy


class Path(object):
    def __init__(self, vertices):
        self.vertices = numpy.array(vertices)

    def length(self):
        return numpy.sum(numpy.abs(self.vertices[1:]-self.vertices[:-1]))


class Paths(object):
    def __init__(self, paths=None):
        if paths is None:
            paths = []
        self.paths = paths

    def add(self, path):
        self.paths.append(path)

    def length(self):
        return numpy.sum([path.length() for path in self.paths])
