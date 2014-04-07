import numpy
from matplotlib import pyplot


class Path(object):
    def __init__(self, vertices):
        self.vertices = numpy.array(vertices)

    def __iter__(self):
        return iter(self.vertices)

    def length(self):
        return numpy.sum(numpy.abs(self.vertices[1:]-self.vertices[:-1]))


class Paths(list):
    def length(self):
        return numpy.sum([path.length() for path in self])

    def vertices(self):
        verts = []
        for path in self:
            verts += list(path.vertices)
        return verts


def get_paths(obj):
    def _get_polygon_paths(polygon):
        def _get_points(c):
            vertices = numpy.array(c.coords)
            return vertices[:, 0] + 1j*vertices[:, 1]
        return [Path(_get_points(sub))
                for sub in [polygon.exterior]+list(polygon.interiors)]

    paths = Paths()
    import shapely.geometry as geom
    if isinstance(obj, geom.polygon.Polygon):
        paths += _get_polygon_paths(obj)
    elif isinstance(obj, geom.multipolygon.MultiPolygon):
        for polygon in obj:
            paths += _get_polygon_paths(polygon)
    return paths


def plot_finish(contours, spectrum=None, contour_labels=True, autofit=True):
    # plot spectrum?
    if spectrum is not None:
        pyplot.plot(numpy.real(spectrum), numpy.imag(spectrum), 'o')

    if autofit:
        vertices = []
        for collection in contours.collections:
            for path in collection.get_paths():
                vertices.append(path.vertices[:, 0] + 1j*path.vertices[:, 1])
        vertices = numpy.concatenate(vertices)
        pyplot.xlim(numpy.min(vertices.real), numpy.max(vertices.real))
        pyplot.ylim(numpy.min(vertices.imag), numpy.max(vertices.imag))

    # plot contour labels?
    from matplotlib.ticker import LogFormatterMathtext
    if contour_labels:
        pyplot.clabel(contours, inline=1, fmt=LogFormatterMathtext())
