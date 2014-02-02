import numpy
from matplotlib import pyplot
import shapely.geometry as geom
from shapely.ops import cascaded_union

from .utils import Path, Paths, plot_finish


class NormalEvals(object):
    def __init__(self, evals):
        self.evals = evals

    def plot(self, epsilons, **kwargs):
        epsilons = list(numpy.sort(epsilons))
        padepsilons = [epsilons[0]*0.9] + epsilons + [epsilons[-1]*1.1]
        X = []
        Y = []
        Z = []
        for epsilon in padepsilons:
            paths = self.contour_paths(epsilon)
            for path in paths:
                X += list(numpy.real(path.vertices[:-1]))
                Y += list(numpy.imag(path.vertices[:-1]))
                Z += [epsilon] * (len(path.vertices) - 1)
        contours = pyplot.tricontour(X, Y, Z, levels=epsilons,
                                     colors=pyplot.rcParams['axes.color_cycle']
                                     )
        plot_finish(contours, **kwargs)
        return contours

    def contour_paths(self, epsilon):
        '''Get boundary of union of circles around eigenvalues'''
        # create circles
        circles = [geom.Point(numpy.real(lamda), numpy.imag(lamda))
                   .buffer(epsilon) for lamda in self.evals]

        # pseudospectrum is union of circles
        pseudospec = cascaded_union(circles)

        def _get_polygon_paths(polygon):
            def _get_points(c):
                vertices = numpy.array(c.coords)
                return vertices[:, 0] + 1j*vertices[:, 1]
            return [Path(_get_points(sub))
                    for sub in [polygon.exterior]+list(polygon.interiors)]

        # grab paths
        paths = Paths()
        if isinstance(pseudospec, geom.polygon.Polygon):
            paths.add(_get_polygon_paths(pseudospec))
        elif isinstance(pseudospec, geom.multipolygon.MultiPolygon):
            for polygon in pseudospec:
                paths.add(_get_polygon_paths(polygon))

        return paths


class Normal(NormalEvals):
    def __init__(self, A):
        from scipy.linalg import eigvals
        super(Normal, self).__init__(eigvals(A))
