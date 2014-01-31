import numpy
from scipy.linalg import svdvals
#from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh, LinearOperator


def svds(A, k=6, ncv=None, tol=0, which='LM', v0=None,
         maxiter=None, return_singular_vectors=True):

    n, m = A.shape

    def matvec_AH_A(x):
        return A.T.conj().dot(A.dot(x))

    AH_A = LinearOperator(matvec=matvec_AH_A, dtype=A.dtype,
                          shape=(A.shape[1], A.shape[1]))

    evals = eigsh(AH_A, k=k, tol=tol ** 2, maxiter=maxiter,
                              ncv=ncv, which=which, v0=v0,
                              return_eigenvectors=False)
    print(evals)
    return numpy.sqrt(evals)


def inv_resolvent_norm(A, z, method='svd'):
    if method == 'svd':
        return numpy.min(svdvals(A - z*numpy.eye(*A.shape)))
    elif method == 'lanczos':
        B = A - z*numpy.eye(*A.shape)
        return svds(B, k=1, which='SM', tol=0, return_singular_vectors=True)


def evaluate_points(A, points):
    return [inv_resolvent_norm(A, point) for point in points]
