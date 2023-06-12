from fcntl import F_GETFL
from typing import Callable, Union
import numpy as np

from numpy import linalg as la
from numpy.linalg.linalg import LinAlgError

def isHermitian(M: np.ndarray) -> bool:
    '''Check if a matrix is Hermitian or not'''
    # getH gives the Hermitian conjugate- only works for matrix data type
    return np.allclose(M, np.asmatrix(M).getH())

def isPositive(M: Union[int, float, np.ndarray], checkHerm=False, tol: float = 1e-9) -> bool:
    '''
    Check if a Hermitian matrix is positive semidefinite or not. Returns isPositive = True or False. 
    If checkHerm= True, then it will also check if the matrix is Hermitian or not
    '''
    # Eigenvalue calculation doesn't support int and float
    if type(M)==float or type(M)==int or M.ndim==0:
        return M>=0
    if checkHerm:
        assert isHermitian(M), "Matrix input to isPositive function should be Hermitian"
    eig_val = la.eigvalsh(M)
    eig_min = np.min(eig_val)
    # Matrix is not positive only if some eigenvalue is lesser than -tolerance
    # [-tolerance, 0] is assumed to be positive
    if eig_min < -tol:
        return False
    else:
        return True

def recreateMatrixFromEigenDecomp(eig_M: np.ndarray, eig_vec_M: np.ndarray) -> np.ndarray:
    '''
    Recreate the matrix given its eigenvalue and eigenvectors as M= PDP^-1.
    The columns of P in this representation are the eigenvectors of M and D= Diag(Eigenvectors of M).
    Input eig_vec_M should be the eigenvectors stored as columns (as eig(h) returns eigenvectors)
    '''
    D = np.diag(eig_M)
    P = eig_vec_M
    invP = la.inv(P)
    # A@B represents matrix multiplication between A and B 
    M = P @ D @ invP
    return M

def computeMatrixFunction (
    func: Callable[[float], float],
    M: np.ndarray
    ) -> np.ndarray:
    '''Computes the function func on M; returns func(M)'''
    assert isHermitian(M), "Matrix input to computeMatrixFunction function should be Hermitian"
    eig_val, eig_vec = la.eigh(M)
    func_M_eig = np.array([func(x) for x in eig_val])
    func_M = recreateMatrixFromEigenDecomp(func_M_eig, eig_vec)
    return func_M

def randomMatrix(rng: np.random._generator.Generator, dim: int) -> np.ndarray:
    '''Create random square matrices of size dim with entries uniformly distributed in [-1,1] X [-1j, 1j]'''
    # -1 subtracts 1 from every element of the matrix
    return (2*rng.random((dim,dim))-1) + 1j*(
        2*rng.random((dim,dim))-1)

def randomPositiveMatrix(rng: np.random._generator.Generator, dim: int) -> np.ndarray:
    '''Create a random positive matrix of size dim by using X= randomMatrix() and returning X*adj(X)'''
    # @ is matrix multiplication
    X = randomMatrix(rng, dim)
    return (X @ np.asmatrix(X).getH())

def isDiagonallyDominant(M: np.ndarray, tol: float = 1e-9) -> bool:
    '''Returns whether |M[i,i]| >= sum_{not j==i} |M[i,j]| for every i or not'''
    dim = M.shape[0]
    if not M.shape== (dim, dim):
        raise ValueError("Input must be a square matrix")

    for i in range(dim):
        row_sum =0
        for j in range(dim):
            if not j==i:
                row_sum+=abs(M[i,j])
        if row_sum > abs(M[i,i]) + tol:
            return False
    # If not diagonally dominant then it has already returned
    return True



'''
# Testing
from numpy.random import default_rng
rng = default_rng()

# Tested successfully with N=10000 and dim =10 
N=10000
dim = 100
negFlag=0

# This function needs the global variable rng and negFlag
def _flipSign(x: float):
    global negFlag
    if rng.random() >0.5:
        # whenever function makes something negative negFlag is turned on
        negFlag = 1 
        return -x
    else:
        return x


for i in range(N):
    # Create a random matrix with entries uniformly distributed in [-1,1] X [-1j, 1j]
    randMat = randomMatrix(rng, dim)
    # randPosMat = randMat. adjoint(randMat) is psd
    randPosMat = randMat @ np.asmatrix(randMat).getH()
    if not isPositive(randPosMat, checkHerm=True):
        print("Error in isHermitian or isPositive",randPosMat)
        break
    try:
        eig_randMat, eig_vec_randMat = la.eig(randMat)
        randMat_reconstructed = recreateMatrixFromEigenDecomp(eig_randMat, eig_vec_randMat)
        if not np.allclose(randMat, randMat_reconstructed):
            print("Error in recreateMatrixFromEigenDecomp",randMat)
            break
    except LinAlgError: 
        continue
    if i==N-1:
        print("Completed isHermitian, isPositive and recreateMatrixFromEigenDecomp tests")


# Testing for computeMatrixFunction

for i in range(N):
    randMat_P = randomMatrix(rng, dim)
    randMat_Q = randomMatrix(rng, dim)
    # P (Q) = randMat_P (_Q). adjoint(randMat_P (_Q)) is psd
    # @ is matrix multiplication
    P = randMat_P @ np.asmatrix(randMat_P).getH()
    Q = randMat_Q @ np.asmatrix(randMat_Q).getH()
    S= (P+Q)/2

    sqrtP = computeMatrixFunction(np.sqrt, P)
    sqrtQ = computeMatrixFunction(np.sqrt, Q)
    sqrtS = computeMatrixFunction(np.sqrt, S)
    Diff = sqrtS - (sqrtP + sqrtQ)/2

    # sqrt function is operator concave
    if not isPositive(Diff):
        print("Error in computeMatrixFunction", Diff)
        break
    if i==N-1:
        print("Completed computeMatrixFunction test")


for i in range(N):
    # Create a random matrix with entries uniformly distributed in [-1,1] X [-1j, 1j]
    randMat = randomMatrix(rng, dim)
    # randPosMat = randMat. adjoint(randMat) is psd
    randPosMat = randMat @ np.asmatrix(randMat).getH()
    randPosMat = computeMatrixFunction(_flipSign, randPosMat)
    if isPositive(randPosMat) == negFlag:
        print("Error in isPositive")
        break
    elif negFlag==1:
        negFlag=0
    if i==N-1:
        print("Completed isPositive test")
'''