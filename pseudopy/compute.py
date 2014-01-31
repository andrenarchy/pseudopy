import numpy
from scipy.linalg import svdvals, schur, eigvalsh, solve_triangular
#from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigs, eigsh, LinearOperator


def inv_resolvent_norm(A, z, method='svd'):
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


def evaluate(A,
             real_min=-1, real_max=1, real_n=50,
             imag_min=-1, imag_max=1, imag_n=50,
             method='svd'
             ):
    real = numpy.linspace(real_min, real_max, real_n)
    imag = numpy.linspace(imag_min, imag_max, imag_n)

    Real, Imag = numpy.meshgrid(real, imag)

    Vals = numpy.zeros((imag_n, real_n))
    if method == 'dense':
        # algorithm from page 375 of Trefethen/Embree 2005
        maxiter = 100
        T, _ = schur(A, output='complex')
        for imag_i in range(imag_n):
            for real_i in range(real_n):
                z = real[real_i]+1j*imag[imag_i]
                T1 = z*numpy.eye(*A.shape) - T
                T2 = T1.T.conj()
                sig_old = 0
                q_old = numpy.random.rand(A.shape[0])
                beta = 0
                H = numpy.zeros((maxiter+1, maxiter+1))
                q = numpy.ones(A.shape[0])
                q /= numpy.linalg.norm(q)
                for p in range(maxiter):
                    v = solve_triangular(T1,
                                         solve_triangular(T2, q)
                                         ) \
                        - beta*q_old
                    alpha = numpy.real(numpy.vdot(q, v))
                    v = v - alpha*q
                    beta = numpy.linalg.norm(v)
                    q_old = q.copy()
                    q = v/beta
                    H[p+1, p] = beta
                    H[p, p+1] = beta
                    H[p, p] = alpha
                    sig = numpy.max(eigvalsh(H[:p+1, :p+1]))
                    if abs(sig_old/sig - 1) < 1e-3:
                        break
                    sig_old = sig
                Vals[imag_i, real_i] = numpy.sqrt(sig)

    else:
        for imag_i in range(imag_n):
            for real_i in range(real_n):
                Vals[imag_i, real_i] = \
                    inv_resolvent_norm(A,
                                       real[real_i]+1j*imag[imag_i],
                                       method=method)
    return Real, Imag, Vals


def evaluate_points(A, points):
    return [inv_resolvent_norm(A, point) for point in points]
