import numpy
from scipy.linalg import svdvals
#from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigs, eigsh, LinearOperator


def inv_resolvent_norm(A, z, method='lanczos'):
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
                      ncv=m+n+1,
                      return_eigenvectors=False)

        return numpy.min(numpy.abs(evals))


def evaluate(A,
             real_min=-1, real_max=1, real_n=50,
             imag_min=-1, imag_max=1, imag_n=50,
             method='svd'
             ):
    real = numpy.linspace(real_min, real_max, real_n)
    imag = numpy.linspace(imag_min, imag_max, imag_n)

    Real, Imag = numpy.meshgrid(real, imag)

    Vals = numpy.zeros((imag_n, real_n))
    for imag_i in range(imag_n):
        for real_i in range(real_n):
            Vals[imag_i, real_i] = \
                inv_resolvent_norm(A, real[real_i]+1j*imag[imag_i])
    return Real, Imag, Vals


def evaluate_points(A, points):
    return [inv_resolvent_norm(A, point) for point in points]
