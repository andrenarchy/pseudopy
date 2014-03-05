import numpy
from scipy.linalg import svdvals, schur, solve_triangular
from scipy.sparse.linalg import eigsh, LinearOperator
from matplotlib.tri import Triangulation
from matplotlib import pyplot

from .utils import Path, Paths, plot_finish


def inv_resolvent_norm(A, z, method='svd'):
    r'''Compute the reciprocal norm of the resolvent

    :param A: the input matrix as a ``numpy.array``, sparse matrix or
      ``LinearOperator`` with ``A.shape==(m,n)``, where :math:`m\geq n`.
    :param z: a complex number
    :param method: (optional) one of

      * ``svd`` (default): computes the minimal singular value of :math:`A-zI`.
        This one should be used for dense matrices.
      * ``lanczos``: computes the minimal singular value with the Lanczos
        iteration on the matrix
        :math:`\begin{bmatrix}0&A\\A^*&0\end{bmatrix}`
    '''
    if method == 'svd':
        return numpy.min(svdvals(A - z*numpy.eye(*A.shape)))
    elif method == 'lanczos':
        m, n = A.shape
        if m > n:
            raise ValueError('m > n is not allowed')
        AH = A.T.conj()

        def matvec(x):
            r'''matrix-vector multiplication

            matrix-vector multiplication with matrix
            :math:`\begin{bmatrix}0&A\\A^*&0\end{bmatrix}`
            '''
            x1 = x[:m]
            x2 = x[m:]
            ret1 = AH.dot(x2) - numpy.conj(z)*x2
            ret2 = numpy.array(A.dot(x1), dtype=numpy.complex)
            ret2[:n] -= z*x1
            return numpy.c_[ret1, ret2]
        AH_A = LinearOperator(matvec=matvec, dtype=numpy.complex,
                              shape=(m+n, m+n))

        evals = eigsh(AH_A, k=2, tol=1e-6, which='SM', maxiter=m+n+1,
                      ncv=2*(m+n),
                      return_eigenvectors=False)

        return numpy.min(numpy.abs(evals))


class _Nonnormal(object):
    '''Base class for nonnormal pseudospectra'''
    def __init__(self, A, points, method='svd'):
        '''Evaluates the inverse resolvent norm on the given list of points

        Stores result in self.vals and points in self.points
        '''
        self.points = points
        if method == 'lanczosinv':
            self.vals = []

            # algorithm from page 375 of Trefethen/Embree 2005
            T, _ = schur(A, output='complex')
            m, n = A.shape
            if m != n:
                raise ValueError('m != n is not allowed in dense mode')
            for point in points:
                M = T - point*numpy.eye(*T.shape)

                def matvec(x):
                    r'''Matrix-vector multiplication

                    Matrix-vector multiplication with matrix
                    :math:`\begin{bmatrix}0&(A-\lambda I)^{-1}\\(A-\lambda I)^{-1}&0\end{bmatrix}`'''
                    return solve_triangular(
                        M,
                        solve_triangular(
                            M,
                            x,
                            check_finite=False
                            ),
                        trans=2,
                        check_finite=False
                        )
                MH_M = LinearOperator(matvec=matvec, dtype=numpy.complex,
                                      shape=(n, n))

                evals = eigsh(MH_M, k=1, tol=1e-3, which='LM',
                              maxiter=n,
                              ncv=n,
                              return_eigenvectors=False)

                self.vals.append(1/numpy.sqrt(numpy.max(numpy.abs(evals))))
        else:
            self.vals = [inv_resolvent_norm(A, point, method=method)
                         for point in points]



class NonnormalMeshgrid(_Nonnormal):
    def __init__(self, A,
                 real_min=-1, real_max=1, real_n=50,
                 imag_min=-1, imag_max=1, imag_n=50,
                 method='svd'):

        real = numpy.linspace(real_min, real_max, real_n)
        imag = numpy.linspace(imag_min, imag_max, imag_n)

        self.Real, self.Imag = numpy.meshgrid(real, imag)

        # call super constructor
        super(NonnormalMeshgrid, self).__init__(
            A, self.Real.flatten() + 1j*self.Imag.flatten())
        self.Vals = numpy.array(self.vals).reshape((imag_n, real_n))

    def plot(self, epsilons, **kwargs):
        contours = pyplot.contour(self.Real, self.Imag, self.Vals,
                                  levels=epsilons,
                                  colors=pyplot.rcParams['axes.color_cycle'])
        plot_finish(contours, **kwargs)
        return contours

    def contour_paths(self, epsilon):
        '''Extract the polygon patches for the provided epsilon'''
        figure = pyplot.figure()
        ax = figure.gca()
        contours = ax.contour(self.Real, self.Imag, self.Vals,
                              levels=[epsilon])
        paths = Paths()
        if len(contours.collections) == 0:
            return paths
        for path in contours.collections[0].get_paths():
            paths.append(Path(path.vertices[:, 0] + 1j*path.vertices[:, 1]))
        pyplot.close(figure)
        return paths


class NonnormalTriang(_Nonnormal):
    def __init__(self, A, triang, **kwargs):
        self.triang = triang
        super(NonnormalTriang, self).__init__(
            A, triang.x + 1j*triang.y, **kwargs)

    def plot(self, epsilons, **kwargs):
        contours = pyplot.tricontour(self.triang, self.vals, levels=epsilons)
        plot_finish(contours, **kwargs)
        return contours

    def contour_paths(self, epsilon):
        '''Extract the polygon patches for the provided epsilon'''
        figure = pyplot.figure()
        contours = pyplot.tricontour(self.triang, self.vals, levels=[epsilon])
        paths = Paths()
        if len(contours.collections) == 0:
            return paths
        for path in contours.collections[0].get_paths():
            paths.append(Path(path.vertices[:, 0] + 1j*path.vertices[:, 1]))
        pyplot.close(figure)
        return paths


class NonnormalPoints(NonnormalTriang):
    def __init__(self, A, points, **kwargs):
        triang = Triangulation(numpy.real(points), numpy.imag(points))
        super(NonnormalPoints, self).__init__(A, triang, **kwargs)


class NonnormalMeshgridAuto(NonnormalMeshgrid):
    '''Determines rough bounding box of pseudospectrum.

    The bounding box is determined for a diagonalizable matrix via the
    condition number of the eigenvector basis (see theorem 2.3 in the book
    of Trefethen and Embree). Note that this method produces a bounding box
    where the pseudospectrum with eps_max is guaranteed to be contained but
    that the bounding box may be overestimated severely.

    :param A: the matrix as numpy array with ``A.shape==(N,N)``.
    :param eps_max: maximal value of :math:`\varepsilon` that is of interest.
    '''
    def __init__(self, A, eps_max, **kwargs):
        from scipy.linalg import eig
        evals, evecs = eig(A)

        # compute condition number of eigenvector basis
        kappa = numpy.linalg.cond(evecs, 2)

        new_kwargs = {'real_min': numpy.min(evals.real) - eps_max*kappa,
                      'real_max': numpy.max(evals.real) + eps_max*kappa,
                      'imag_min': numpy.min(evals.imag) - eps_max*kappa,
                      'imag_max': numpy.max(evals.imag) + eps_max*kappa
                      }
        new_kwargs.update(kwargs)
        super(NonnormalMeshgridAuto, self).__init__(A, **new_kwargs)
