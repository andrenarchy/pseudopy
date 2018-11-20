import matplotlib
matplotlib.use('Agg')

import numpy
import pseudopy
from itertools import product

def dict_merge(*dicts):
    items = []
    for d in dicts:
        items += d.items()
    return dict(items)


def dict_slicevals(d, keys):
    return [d[k] for k in keys]


def test():
    n = 10
    A = numpy.diag(numpy.ones(n-1), -1)
    A[0, -1] = 1

    # compute evals
    from scipy.linalg import eigvals
    evals = eigvals(A)

    nonnormal_params = {'real_min': -2.2, 'real_max': 2.2, 'real_n': 200,
                        'imag_min': -2.2, 'imag_max': 2.2, 'imag_n': 200}

    # compute points
    real = numpy.linspace(*dict_slicevals(nonnormal_params,
                                          ['real_min', 'real_max', 'real_n']))
    imag = numpy.linspace(*dict_slicevals(nonnormal_params,
                                          ['imag_min', 'imag_max', 'imag_n']))
    Real, Imag = numpy.meshgrid(real, imag)
    points = Real.flatten() + 1j*Imag.flatten()

    # compute triang from points
    from matplotlib.tri import Triangulation
    triang = Triangulation(numpy.real(points), numpy.imag(points))

    # define classes to test
    classes = {
        pseudopy.NonnormalMeshgrid: [dict_merge(nonnormal_params, {'A': A})],
        pseudopy.NonnormalTriang: [{'A': A, 'triang': triang}],
        pseudopy.NonnormalPoints: [{'A': A, 'points': points}],
        pseudopy.Normal: [{'A': A}],
        pseudopy.NormalEvals: [{'evals': evals}]
        }

    # define epsilons
    epsilons = [0.2, 0.7, 1.1]

    for cls, params in classes.items():
        for param in params:
            pseudo = cls(**param)

            # test plot
            #yield run_plot, pseudo, epsilons

            # test contour_paths
            for epsilon in epsilons:
                yield run_contour_paths, pseudo, epsilon, evals


def run_plot(pseudo, epsilons):
    from matplotlib import pyplot
    pyplot.figure()
    pseudo.plot(epsilons)
    pyplot.close()


def run_contour_paths(pseudo, epsilon, evals):
    # get paths
    paths = pseudo.contour_paths(epsilon)

    # check if pseudospectrum is correct by matching the parts of it
    import shapely.geometry as geom
    from shapely.ops import cascaded_union
    # create circles
    circles = [geom.Point(numpy.real(lamda), numpy.imag(lamda))
               .buffer(epsilon) for lamda in evals]
    exact_pseudo = cascaded_union(circles)
    exact_paths = pseudopy.utils.get_paths(exact_pseudo)

    N = len(paths)
    assert(N == len(exact_paths))

    # create polygons
    polys = [geom.Polygon([(numpy.real(z), numpy.imag(z))
                           for z in path.vertices])
             for path in paths]
    exact_polys = [geom.Polygon([(numpy.real(z), numpy.imag(z))
                                 for z in path.vertices])
                   for path in exact_paths]

    # match elements by measuring intersections
    M = numpy.zeros((N, N))
    for (i, j) in product(range(N), range(N)):
        M[i, j] = exact_polys[i].symmetric_difference(polys[j]).area
    for i in range(N):
        assert(numpy.min(M[i, :]) < 0.1*epsilon)


if __name__ == '__main__':
    import nose
    nose.main()
