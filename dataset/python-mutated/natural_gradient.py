"""Natural Gradient."""
from collections.abc import Iterable
from typing import List, Tuple, Callable, Optional, Union
import numpy as np
from qiskit.circuit import ParameterVector, ParameterExpression
from qiskit.circuit._utils import sort_parameters
from qiskit.utils import optionals as _optionals
from qiskit.utils.deprecation import deprecate_func
from ..operator_base import OperatorBase
from ..list_ops.list_op import ListOp
from ..list_ops.composed_op import ComposedOp
from ..state_fns.circuit_state_fn import CircuitStateFn
from .circuit_gradients import CircuitGradient
from .circuit_qfis import CircuitQFI
from .gradient import Gradient
from .gradient_base import GradientBase
from .qfi import QFI
ETOL = 1e-08
RCOND = 0.01

class NaturalGradient(GradientBase):
    """Deprecated: Convert an operator expression to the first-order gradient.

    Given an ill-posed inverse problem

        x = arg min{||Ax-C||^2} (1)

    one can use regularization schemes can be used to stabilize the system and find a numerical
    solution

        x_lambda = arg min{||Ax-C||^2 + lambda*R(x)} (2)

    where R(x) represents the penalization term.
    """

    @deprecate_func(since='0.24.0', package_name='qiskit-terra', additional_msg='For code migration guidelines, visit https://qisk.it/opflow_migration.')
    def __init__(self, grad_method: Union[str, CircuitGradient]='lin_comb', qfi_method: Union[str, CircuitQFI]='lin_comb_full', regularization: Optional[str]=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Args:\n            grad_method: The method used to compute the state gradient. Can be either\n                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.\n            qfi_method: The method used to compute the QFI. Can be either\n                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.\n            regularization: Use the following regularization with a least square method to solve the\n                underlying system of linear equations\n                Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``\n                ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search\n                If regularization is None but the metric is ill-conditioned or singular then\n                a least square solver is used without regularization\n            kwargs (dict): Optional parameters for a CircuitGradient\n        "
        super().__init__(grad_method)
        self._qfi_method = QFI(qfi_method)
        self._regularization = regularization
        self._epsilon = kwargs.get('epsilon', 1e-06)

    def convert(self, operator: OperatorBase, params: Optional[Union[ParameterVector, ParameterExpression, List[ParameterExpression]]]=None) -> OperatorBase:
        if False:
            return 10
        '\n        Args:\n            operator: The operator we are taking the gradient of.\n            params: The parameters we are taking the gradient with respect to. If not explicitly\n                passed, they are inferred from the operator and sorted by name.\n\n        Returns:\n            An operator whose evaluation yields the NaturalGradient.\n\n        Raises:\n            TypeError: If ``operator`` does not represent an expectation value or the quantum\n                state is not ``CircuitStateFn``.\n            ValueError: If ``params`` contains a parameter not present in ``operator``.\n            ValueError: If ``operator`` is not parameterized.\n        '
        if not isinstance(operator, ComposedOp):
            if not (isinstance(operator, ListOp) and len(operator.oplist) == 1):
                raise TypeError('Please provide the operator either as ComposedOp or as ListOp of a CircuitStateFn potentially with a combo function.')
        if not isinstance(operator[-1], CircuitStateFn):
            raise TypeError('Please make sure that the operator for which you want to compute Quantum Fisher Information represents an expectation value or a loss function and that the quantum state is given as CircuitStateFn.')
        if len(operator.parameters) == 0:
            raise ValueError('The operator we are taking the gradient of is not parameterized!')
        if params is None:
            params = sort_parameters(operator.parameters)
        if not isinstance(params, Iterable):
            params = [params]
        grad = Gradient(self._grad_method, epsilon=self._epsilon).convert(operator, params)
        metric = self._qfi_method.convert(operator[-1], params) * 0.25

        def combo_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return self.nat_grad_combo_fn(x, self.regularization)
        return ListOp([grad, metric], combo_fn=combo_fn)

    @staticmethod
    def nat_grad_combo_fn(x: tuple, regularization: Optional[str]=None) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Natural Gradient Function Implementation.\n\n        Args:\n            x: Iterable consisting of Gradient, Quantum Fisher Information.\n            regularization: Regularization method.\n\n        Returns:\n            Natural Gradient.\n\n        Raises:\n            ValueError: If the gradient has imaginary components that are non-negligible.\n\n        '
        gradient = x[0]
        metric = x[1]
        if np.amax(np.abs(np.imag(gradient))) > ETOL:
            raise ValueError(f'The imaginary part of the gradient are non-negligible. The largest absolute imaginary value in the gradient is {np.amax(np.abs(np.imag(gradient)))}. Please increase the number of shots.')
        gradient = np.real(gradient)
        if np.amax(np.abs(np.imag(metric))) > ETOL:
            raise ValueError(f'The imaginary part of the metric are non-negligible. The largest absolute imaginary value in the gradient is {np.amax(np.abs(np.imag(metric)))}. Please increase the number of shots.')
        metric = np.real(metric)
        if regularization is not None:
            nat_grad = NaturalGradient._regularized_sle_solver(metric, gradient, regularization=regularization)
        else:
            (w, v) = np.linalg.eigh(metric)
            if not all((ew >= -1 * ETOL for ew in w)):
                raise ValueError(f'The underlying metric has at least one Eigenvalue < -{ETOL}. The smallest Eigenvalue is {np.amin(w)} Please use a regularized least-square solver for this problem or increase the number of backend shots.')
            if not all((ew >= 0 for ew in w)):
                w = [max(ETOL, ew) for ew in w]
                metric = np.real(v @ np.diag(w) @ np.linalg.inv(v))
            nat_grad = np.linalg.lstsq(metric, gradient, rcond=RCOND)[0]
        return nat_grad

    @property
    def qfi_method(self) -> CircuitQFI:
        if False:
            for i in range(10):
                print('nop')
        'Returns ``CircuitQFI``.\n\n        Returns: ``CircuitQFI``.\n\n        '
        return self._qfi_method.qfi_method

    @property
    def regularization(self) -> Optional[str]:
        if False:
            return 10
        'Returns the regularization option.\n\n        Returns: the regularization option.\n\n        '
        return self._regularization

    @staticmethod
    def _reg_term_search(metric: np.ndarray, gradient: np.ndarray, reg_method: Callable[[np.ndarray, np.ndarray, float], float], lambda1: float=0.001, lambda4: float=1.0, tol: float=1e-08) -> Tuple[float, np.ndarray]:
        if False:
            print('Hello World!')
        '\n        This method implements a search for a regularization parameter lambda by finding for the\n        corner of the L-curve.\n        More explicitly, one has to evaluate a suitable lambda by finding a compromise between\n        the error in the solution and the norm of the regularization.\n        This function implements a method presented in\n        `A simple algorithm to find the L-curve corner in the regularization of inverse problems\n         <https://arxiv.org/pdf/1608.04571.pdf>`\n\n        Args:\n            metric: See (1) and (2).\n            gradient: See (1) and (2).\n            reg_method: Given the metric, gradient and lambda the regularization method must return\n                ``x_lambda`` - see (2).\n            lambda1: Left starting point for L-curve corner search.\n            lambda4: Right starting point for L-curve corner search.\n            tol: Termination threshold.\n\n        Returns:\n            Regularization coefficient which is the solution to the regularization inverse problem.\n        '

        def _get_curvature(x_lambda: List) -> float:
            if False:
                return 10
            'Calculate Menger curvature\n\n            Menger, K. (1930).  Untersuchungen  ̈uber Allgemeine Metrik. Math. Ann.,103(1), 466–501\n\n            Args:\n                ``x_lambda: [[x_lambdaj], [x_lambdak], [x_lambdal]]``\n                    ``lambdaj < lambdak < lambdal``\n\n            Returns:\n                Menger Curvature\n\n            '
            eps = []
            eta = []
            for x in x_lambda:
                try:
                    eps.append(np.log(np.linalg.norm(np.matmul(metric, x) - gradient) ** 2))
                except ValueError:
                    eps.append(np.log(np.linalg.norm(np.matmul(metric, np.transpose(x)) - gradient) ** 2))
                eta.append(np.log(max(np.linalg.norm(x) ** 2, ETOL)))
            p_temp = 1
            c_k = 0
            for i in range(3):
                p_temp *= (eps[np.mod(i + 1, 3)] - eps[i]) ** 2 + (eta[np.mod(i + 1, 3)] - eta[i]) ** 2
                c_k += eps[i] * eta[np.mod(i + 1, 3)] - eps[np.mod(i + 1, 3)] * eta[i]
            c_k = 2 * c_k / max(0.0001, np.sqrt(p_temp))
            return c_k

        def get_lambda2_lambda3(lambda1, lambda4):
            if False:
                for i in range(10):
                    print('nop')
            gold_sec = (1 + np.sqrt(5)) / 2.0
            lambda2 = 10 ** ((np.log10(lambda4) + np.log10(lambda1) * gold_sec) / (1 + gold_sec))
            lambda3 = 10 ** (np.log10(lambda1) + np.log10(lambda4) - np.log10(lambda2))
            return (lambda2, lambda3)
        (lambda2, lambda3) = get_lambda2_lambda3(lambda1, lambda4)
        lambda_ = [lambda1, lambda2, lambda3, lambda4]
        x_lambda = []
        for lam in lambda_:
            x_lambda.append(reg_method(metric, gradient, lam))
        counter = 0
        while (lambda_[3] - lambda_[0]) / lambda_[3] >= tol:
            counter += 1
            c_2 = _get_curvature(x_lambda[:-1])
            c_3 = _get_curvature(x_lambda[1:])
            while c_3 < 0:
                lambda_[3] = lambda_[2]
                x_lambda[3] = x_lambda[2]
                lambda_[2] = lambda_[1]
                x_lambda[2] = x_lambda[1]
                (lambda2, _) = get_lambda2_lambda3(lambda_[0], lambda_[3])
                lambda_[1] = lambda2
                x_lambda[1] = reg_method(metric, gradient, lambda_[1])
                c_3 = _get_curvature(x_lambda[1:])
            if c_2 > c_3:
                lambda_mc = lambda_[1]
                x_mc = x_lambda[1]
                lambda_[3] = lambda_[2]
                x_lambda[3] = x_lambda[2]
                lambda_[2] = lambda_[1]
                x_lambda[2] = x_lambda[1]
                (lambda2, _) = get_lambda2_lambda3(lambda_[0], lambda_[3])
                lambda_[1] = lambda2
                x_lambda[1] = reg_method(metric, gradient, lambda_[1])
            else:
                lambda_mc = lambda_[2]
                x_mc = x_lambda[2]
                lambda_[0] = lambda_[1]
                x_lambda[0] = x_lambda[1]
                lambda_[1] = lambda_[2]
                x_lambda[1] = x_lambda[2]
                (_, lambda3) = get_lambda2_lambda3(lambda_[0], lambda_[3])
                lambda_[2] = lambda3
                x_lambda[2] = reg_method(metric, gradient, lambda_[2])
        return (lambda_mc, x_mc)

    @staticmethod
    @_optionals.HAS_SKLEARN.require_in_call
    def _ridge(metric: np.ndarray, gradient: np.ndarray, lambda_: float=1.0, lambda1: float=0.0001, lambda4: float=0.1, tol_search: float=1e-08, fit_intercept: bool=True, normalize: bool=False, copy_a: bool=True, max_iter: int=1000, tol: float=0.0001, solver: str='auto', random_state: Optional[int]=None) -> Tuple[float, np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Ridge Regression with automatic search for a good regularization term lambda\n        x_lambda = arg min{||Ax-C||^2 + lambda*||x||_2^2} (3)\n        `Scikit Learn Ridge Regression\n        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`\n\n        Args:\n            metric: See (1) and (2).\n            gradient: See (1) and (2).\n            lambda_ : regularization parameter used if auto_search = False\n            lambda1: left starting point for L-curve corner search\n            lambda4: right starting point for L-curve corner search\n            tol_search: termination threshold for regularization parameter search\n            fit_intercept: if True calculate intercept\n            normalize: ignored if fit_intercept=False, if True normalize A for regression\n            copy_a: if True A is copied, else overwritten\n            max_iter: max. number of iterations if solver is CG\n            tol: precision of the regression solution\n            solver: solver {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}\n            random_state: seed for the pseudo random number generator used when data is shuffled\n\n        Returns:\n           regularization coefficient, solution to the regularization inverse problem\n\n        Raises:\n            MissingOptionalLibraryError: scikit-learn not installed\n\n        '
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        reg = Ridge(alpha=lambda_, fit_intercept=fit_intercept, copy_X=copy_a, max_iter=max_iter, tol=tol, solver=solver, random_state=random_state)

        def reg_method(a, c, alpha):
            if False:
                while True:
                    i = 10
            reg.set_params(alpha=alpha)
            if normalize:
                reg.fit(StandardScaler().fit_transform(a), c)
            else:
                reg.fit(a, c)
            return reg.coef_
        (lambda_mc, x_mc) = NaturalGradient._reg_term_search(metric, gradient, reg_method, lambda1=lambda1, lambda4=lambda4, tol=tol_search)
        return (lambda_mc, np.transpose(x_mc))

    @staticmethod
    @_optionals.HAS_SKLEARN.require_in_call
    def _lasso(metric: np.ndarray, gradient: np.ndarray, lambda_: float=1.0, lambda1: float=0.0001, lambda4: float=0.1, tol_search: float=1e-08, fit_intercept: bool=True, normalize: bool=False, precompute: Union[bool, Iterable]=False, copy_a: bool=True, max_iter: int=1000, tol: float=0.0001, warm_start: bool=False, positive: bool=False, random_state: Optional[int]=None, selection: str='random') -> Tuple[float, np.ndarray]:
        if False:
            while True:
                i = 10
        "\n        Lasso Regression with automatic search for a good regularization term lambda\n        x_lambda = arg min{||Ax-C||^2/(2*n_samples) + lambda*||x||_1} (4)\n        `Scikit Learn Lasso Regression\n        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`\n\n        Args:\n            metric: Matrix of size mxn.\n            gradient: Vector of size m.\n            lambda_ : regularization parameter used if auto_search = False\n            lambda1: left starting point for L-curve corner search\n            lambda4: right starting point for L-curve corner search\n            tol_search: termination threshold for regularization parameter search\n            fit_intercept: if True calculate intercept\n            normalize: ignored if fit_intercept=False, if True normalize A for regression\n            precompute: If True compute and use Gram matrix to speed up calculations.\n                                             Gram matrix can also be given explicitly\n            copy_a: if True A is copied, else overwritten\n            max_iter: max. number of iterations if solver is CG\n            tol: precision of the regression solution\n            warm_start: if True reuse solution from previous fit as initialization\n            positive: if True force positive coefficients\n            random_state: seed for the pseudo random number generator used when data is shuffled\n            selection: {'cyclic', 'random'}\n\n        Returns:\n            regularization coefficient, solution to the regularization inverse problem\n\n        Raises:\n            MissingOptionalLibraryError: scikit-learn not installed\n\n        "
        from sklearn.linear_model import Lasso
        from sklearn.preprocessing import StandardScaler
        reg = Lasso(alpha=lambda_, fit_intercept=fit_intercept, precompute=precompute, copy_X=copy_a, max_iter=max_iter, tol=tol, warm_start=warm_start, positive=positive, random_state=random_state, selection=selection)

        def reg_method(a, c, alpha):
            if False:
                return 10
            reg.set_params(alpha=alpha)
            if normalize:
                reg.fit(StandardScaler().fit_transform(a), c)
            else:
                reg.fit(a, c)
            return reg.coef_
        (lambda_mc, x_mc) = NaturalGradient._reg_term_search(metric, gradient, reg_method, lambda1=lambda1, lambda4=lambda4, tol=tol_search)
        return (lambda_mc, x_mc)

    @staticmethod
    def _regularized_sle_solver(metric: np.ndarray, gradient: np.ndarray, regularization: str='perturb_diag', lambda1: float=0.001, lambda4: float=1.0, alpha: float=0.0, tol_norm_x: Tuple[float, float]=(1e-08, 5.0), tol_cond_a: float=1000.0) -> np.ndarray:
        if False:
            print('Hello World!')
        "\n        Solve a linear system of equations with a regularization method and automatic lambda fitting\n\n        Args:\n            metric: Matrix of size mxn.\n            gradient: Vector of size m.\n            regularization: Regularization scheme to be used: 'ridge', 'lasso',\n                'perturb_diag_elements' or 'perturb_diag'\n            lambda1: left starting point for L-curve corner search (for 'ridge' and 'lasso')\n            lambda4: right starting point for L-curve corner search (for 'ridge' and 'lasso')\n            alpha: perturbation coefficient for 'perturb_diag_elements' and 'perturb_diag'\n            tol_norm_x: tolerance for the norm of x\n            tol_cond_a: tolerance for the condition number of A\n\n        Returns:\n            solution to the regularized system of linear equations\n\n        "
        if regularization == 'ridge':
            (_, x) = NaturalGradient._ridge(metric, gradient, lambda1=lambda1)
        elif regularization == 'lasso':
            (_, x) = NaturalGradient._lasso(metric, gradient, lambda1=lambda1)
        elif regularization == 'perturb_diag_elements':
            alpha = 1e-07
            while np.linalg.cond(metric + alpha * np.diag(metric)) > tol_cond_a:
                alpha *= 10
            (x, _, _, _) = np.linalg.lstsq(metric + alpha * np.diag(metric), gradient, rcond=None)
        elif regularization == 'perturb_diag':
            alpha = 1e-07
            while np.linalg.cond(metric + alpha * np.eye(len(gradient))) > tol_cond_a:
                alpha *= 10
            (x, _, _, _) = np.linalg.lstsq(metric + alpha * np.eye(len(gradient)), gradient, rcond=None)
        else:
            (x, _, _, _) = np.linalg.lstsq(metric, gradient, rcond=None)
        if np.linalg.norm(x) > tol_norm_x[1] or np.linalg.norm(x) < tol_norm_x[0]:
            if regularization == 'ridge':
                lambda1 = lambda1 / 10.0
                (_, x) = NaturalGradient._ridge(metric, gradient, lambda1=lambda1, lambda4=lambda4)
            elif regularization == 'lasso':
                lambda1 = lambda1 / 10.0
                (_, x) = NaturalGradient._lasso(metric, gradient, lambda1=lambda1)
            elif regularization == 'perturb_diag_elements':
                while np.linalg.cond(metric + alpha * np.diag(metric)) > tol_cond_a:
                    if alpha == 0:
                        alpha = 1e-07
                    else:
                        alpha *= 10
                (x, _, _, _) = np.linalg.lstsq(metric + alpha * np.diag(metric), gradient, rcond=None)
            else:
                if alpha == 0:
                    alpha = 1e-07
                else:
                    alpha *= 10
                while np.linalg.cond(metric + alpha * np.eye(len(gradient))) > tol_cond_a:
                    (x, _, _, _) = np.linalg.lstsq(metric + alpha * np.eye(len(gradient)), gradient, rcond=None)
                    alpha *= 10
        return x