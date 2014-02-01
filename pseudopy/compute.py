# -*- coding: utf-8 -*-

import numpy
from scipy.linalg import svdvals, schur, solve_triangular
from scipy.sparse.linalg import eigsh, LinearOperator


def inv_resolvent_norm(A, z, method='svd'):
    '''Compute the reciprocal norm of the resolvent

    :param A: the input matrix as a ``numpy.array``, sparse matrix or
      ``LinearOperator`` with ``A.shape==(m,n)``, where :math:`m\geq n`.
    :param z: a complex number
    :param method: (optional) one of

      * ``svd`` (default): computes the minimal singular value of :math:`A-zI`.
        This one should be used for dense matrices.
      * ``lanczos``: computes the minimal singular value with the Lanczos
        iteration on the matrix
        :math:`\\begin{bmatrix}0&A\\\\A^*&0\\end{bmatrix}`
    '''
    if method == 'svd':
        return numpy.min(svdvals(A - z*numpy.eye(*A.shape)))
    elif method == 'lanczos':
        m, n = A.shape
        if m > n:
            raise ValueError('m > n is not allowed')
        AH = A.T.conj()

        def matvec(x):
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


def evaluate_meshgrid(A,
                      real_min=-1, real_max=1, real_n=50,
                      imag_min=-1, imag_max=1, imag_n=50,
                      **kwargs
                      ):
    '''Evaluates the inverse resolvent norm on a meshgrid'''
    real = numpy.linspace(real_min, real_max, real_n)
    imag = numpy.linspace(imag_min, imag_max, imag_n)

    Real, Imag = numpy.meshgrid(real, imag)

    points = Real.flatten() + 1j*Imag.flatten()
    vals = evaluate_points(A, points, **kwargs)

    return Real, Imag, numpy.array(vals).reshape((imag_n, real_n))


def evaluate_points(A, points, **kwargs):
    '''Evaluates the inverse resolvent norm on the given list of points'''
    if 'method' in kwargs and kwargs['method'] == 'lanczosinv':
        vals = []

        # algorithm from page 375 of Trefethen/Embree 2005
        T, _ = schur(A, output='complex')
        m, n = A.shape
        if m != n:
            raise ValueError('m != n is not allowed in dense mode')
        for point in points:
            M = T - point*numpy.eye(*T.shape)

            def matvec(x):
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

            vals.append(1/numpy.sqrt(numpy.max(numpy.abs(evals))))
        return vals
    else:
        return [inv_resolvent_norm(A, point, **kwargs) for point in points]


def level_paths_meshgrid(X, Y, Z, levels):
    '''Extract the polygon patches of the provided levels'''
    from matplotlib import pyplot
    contour = pyplot.contour(X, Y, Z, levels)

    level_paths = {}
    for level, i in zip(levels, range(0, len(levels))):
        line = contour.collections[i]
        level_paths[level] = []
        for path in line.get_paths():
            level_paths[level].append(
                path.vertices[:, 0]+1j*path.vertices[:, 1]
                )
    return level_paths


def path_length(path):
    '''Compute length of given polygon path'''
    return numpy.sum(numpy.abs(path[1:]-path[:-1]))
