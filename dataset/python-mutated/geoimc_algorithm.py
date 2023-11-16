"""
Module maintaining the IMC problem.
"""
import numpy as np
from scipy.sparse import csr_matrix
from numba import njit, prange
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product, SymmetricPositiveDefinite
from pymanopt.solvers import ConjugateGradient
from pymanopt.solvers.linesearch import LineSearchBackTracking

class IMCProblem(object):
    """
    Implements the IMC problem.
    """

    def __init__(self, dataPtr, lambda1=0.01, rank=10):
        if False:
            i = 10
            return i + 15
        'Initialize parameters\n\n        Args:\n            dataPtr (DataPtr): An object of which contains X, Z side features and target matrix Y.\n            lambda1 (uint): Regularizer.\n            rank (uint): rank of the U, B, V parametrization.\n        '
        self.dataset = dataPtr
        self.X = self.dataset.get_entity('row')
        self.Z = self.dataset.get_entity('col')
        self.rank = rank
        self._loadTarget()
        self.shape = (self.X.shape[0], self.Z.shape[0])
        self.lambda1 = lambda1
        self.nSamples = self.Y.data.shape[0]
        self.W = None
        self.optima_reached = False
        self.manifold = Product([Stiefel(self.X.shape[1], self.rank), SymmetricPositiveDefinite(self.rank), Stiefel(self.Z.shape[1], self.rank)])

    def _loadTarget(self):
        if False:
            for i in range(10):
                print('nop')
        'Loads target matrix from the dataset pointer.'
        self.Y = self.dataset.get_data()

    @staticmethod
    @njit(nogil=True, parallel=True)
    def _computeLoss_csrmatrix(a, b, cd, indices, indptr, residual_global):
        if False:
            for i in range(10):
                print('nop')
        'computes residual_global = a*b - cd at given indices in csr_matrix format'
        N = a.shape[0]
        M = a.shape[1]
        for i in prange(N):
            for j in prange(indptr[i], indptr[i + 1]):
                num = 0.0
                for k in range(M):
                    num += a[i, k] * b[k, indices[j]]
                residual_global[j] = num - cd[j]
        return residual_global

    def _cost(self, params, residual_global):
        if False:
            i = 10
            return i + 15
        'Compute the cost of GeoIMC optimization problem\n\n        Args:\n            params (Iterator): An iterator containing the manifold point at which\n            the cost needs to be evaluated.\n            residual_global (csr_matrix): Residual matrix.\n        '
        U = params[0]
        B = params[1]
        V = params[2]
        regularizer = 0.5 * self.lambda1 * np.sum(B ** 2)
        IMCProblem._computeLoss_csrmatrix(self.X.dot(U.dot(B)), V.T.dot(self.Z.T), self.Y.data, self.Y.indices, self.Y.indptr, residual_global)
        cost = 0.5 * np.sum(residual_global ** 2) / self.nSamples + regularizer
        return cost

    def _egrad(self, params, residual_global):
        if False:
            for i in range(10):
                print('nop')
        'Computes the euclidean gradient\n\n        Args:\n            params (Iterator): An iterator containing the manifold point at which\n            the cost needs to be evaluated.\n            residual_global (csr_matrix): Residual matrix.\n        '
        U = params[0]
        B = params[1]
        V = params[2]
        residual_global_csr = csr_matrix((residual_global, self.Y.indices, self.Y.indptr), shape=self.shape)
        gradU = np.dot(self.X.T, residual_global_csr.dot(self.Z.dot(V.dot(B.T)))) / self.nSamples
        gradB = np.dot(self.X.dot(U).T, residual_global_csr.dot(self.Z.dot(V))) / self.nSamples + self.lambda1 * B
        gradB_sym = (gradB + gradB.T) / 2
        gradV = np.dot(self.X.dot(U.dot(B)).T, residual_global_csr.dot(self.Z)).T / self.nSamples
        return [gradU, gradB_sym, gradV]

    def solve(self, *args):
        if False:
            while True:
                i = 10
        'Main solver of the IMC model\n\n        Args:\n            max_opt_time (uint): Maximum time (in secs) for optimization\n            max_opt_iter (uint): Maximum iterations for optimization\n            verbosity (uint): The level of verbosity for Pymanopt logs\n        '
        if self.optima_reached:
            return
        self._optimize(*args)
        self.optima_reached = True
        return

    def _optimize(self, max_opt_time, max_opt_iter, verbosity):
        if False:
            return 10
        'Optimize the GeoIMC optimization problem\n\n        Args: The args of `solve`\n        '
        residual_global = np.zeros(self.Y.data.shape)
        solver = ConjugateGradient(maxtime=max_opt_time, maxiter=max_opt_iter, linesearch=LineSearchBackTracking())
        prb = Problem(manifold=self.manifold, cost=lambda x: self._cost(x, residual_global), egrad=lambda z: self._egrad(z, residual_global), verbosity=verbosity)
        solution = solver.solve(prb, x=self.W)
        self.W = [solution[0], solution[1], solution[2]]
        return self._cost(self.W, residual_global)

    def reset(self):
        if False:
            while True:
                i = 10
        'Reset the model.'
        self.optima_reached = False
        self.W = None
        return