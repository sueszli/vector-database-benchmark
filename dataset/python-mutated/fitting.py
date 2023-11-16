"""
This module implements classes (called Fitters) which combine optimization
algorithms (typically from `scipy.optimize`) with statistic functions to perform
fitting. Fitters are implemented as callable classes. In addition to the data
to fit, the ``__call__`` method takes an instance of
`~astropy.modeling.core.FittableModel` as input, and returns a copy of the
model with its parameters determined by the optimizer.

Optimization algorithms, called "optimizers" are implemented in
`~astropy.modeling.optimizers` and statistic functions are in
`~astropy.modeling.statistic`. The goal is to provide an easy to extend
framework and allow users to easily create new fitters by combining statistics
with optimizers.

There are two exceptions to the above scheme.
`~astropy.modeling.fitting.LinearLSQFitter` uses Numpy's `~numpy.linalg.lstsq`
function.  `~astropy.modeling.fitting.LevMarLSQFitter` uses
`~scipy.optimize.leastsq` which combines optimization and statistic in one
implementation.
"""
import abc
import inspect
import operator
import warnings
from functools import reduce, wraps
from importlib.metadata import entry_points
import numpy as np
from astropy.units import Quantity
from astropy.utils.exceptions import AstropyUserWarning
from .optimizers import DEFAULT_ACC, DEFAULT_EPS, DEFAULT_MAXITER, SLSQP, Simplex
from .spline import SplineExactKnotsFitter, SplineInterpolateFitter, SplineSmoothingFitter, SplineSplrepFitter
from .statistic import leastsquare
from .utils import _combine_equivalency_dict, poly_map_domain
__all__ = ['LinearLSQFitter', 'LevMarLSQFitter', 'TRFLSQFitter', 'DogBoxLSQFitter', 'LMLSQFitter', 'FittingWithOutlierRemoval', 'SLSQPLSQFitter', 'SimplexLSQFitter', 'JointFitter', 'Fitter', 'ModelLinearityError', 'ModelsError', 'SplineExactKnotsFitter', 'SplineInterpolateFitter', 'SplineSmoothingFitter', 'SplineSplrepFitter']
STATISTICS = [leastsquare]
OPTIMIZERS = [Simplex, SLSQP]

class NonFiniteValueError(RuntimeError):
    """
    Error raised when attempting to a non-finite value.
    """

class Covariance:
    """Class for covariance matrix calculated by fitter."""

    def __init__(self, cov_matrix, param_names):
        if False:
            print('Hello World!')
        self.cov_matrix = cov_matrix
        self.param_names = param_names

    def pprint(self, max_lines, round_val):
        if False:
            for i in range(10):
                print('nop')
        longest_name = max((len(x) for x in self.param_names))
        ret_str = 'parameter variances / covariances \n'
        fstring = f"{'': <{longest_name}}| {{0}}\n"
        for (i, row) in enumerate(self.cov_matrix):
            if i <= max_lines - 1:
                param = self.param_names[i]
                ret_str += fstring.replace(' ' * len(param), param, 1).format(repr(np.round(row[:i + 1], round_val))[7:-2])
            else:
                ret_str += '...'
        return ret_str.rstrip()

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.pprint(max_lines=10, round_val=3)

    def __getitem__(self, params):
        if False:
            return 10
        if len(params) != 2:
            raise ValueError('Covariance must be indexed by two values.')
        if all((isinstance(item, str) for item in params)):
            (i1, i2) = (self.param_names.index(params[0]), self.param_names.index(params[1]))
        elif all((isinstance(item, int) for item in params)):
            (i1, i2) = params
        else:
            raise TypeError('Covariance can be indexed by two parameter names or integer indices.')
        return self.cov_matrix[i1][i2]

class StandardDeviations:
    """Class for fitting uncertainties."""

    def __init__(self, cov_matrix, param_names):
        if False:
            print('Hello World!')
        self.param_names = param_names
        self.stds = self._calc_stds(cov_matrix)

    def _calc_stds(self, cov_matrix):
        if False:
            i = 10
            return i + 15
        stds = [np.sqrt(x) if x > 0 else None for x in np.diag(cov_matrix)]
        return stds

    def pprint(self, max_lines, round_val):
        if False:
            while True:
                i = 10
        longest_name = max((len(x) for x in self.param_names))
        ret_str = 'standard deviations\n'
        for (i, std) in enumerate(self.stds):
            if i <= max_lines - 1:
                param = self.param_names[i]
                ret_str += f"{param}{' ' * (longest_name - len(param))}| {np.round(std, round_val)}\n"
            else:
                ret_str += '...'
        return ret_str.rstrip()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self.pprint(max_lines=10, round_val=3)

    def __getitem__(self, param):
        if False:
            print('Hello World!')
        if isinstance(param, str):
            i = self.param_names.index(param)
        elif isinstance(param, int):
            i = param
        else:
            raise TypeError('Standard deviation can be indexed by parameter name or integer.')
        return self.stds[i]

class ModelsError(Exception):
    """Base class for model exceptions."""

class ModelLinearityError(ModelsError):
    """Raised when a non-linear model is passed to a linear fitter."""

class UnsupportedConstraintError(ModelsError, ValueError):
    """
    Raised when a fitter does not support a type of constraint.
    """

class _FitterMeta(abc.ABCMeta):
    """
    Currently just provides a registry for all Fitter classes.
    """
    registry = set()

    def __new__(mcls, name, bases, members):
        if False:
            while True:
                i = 10
        cls = super().__new__(mcls, name, bases, members)
        if not inspect.isabstract(cls) and (not name.startswith('_')):
            mcls.registry.add(cls)
        return cls

def fitter_unit_support(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    This is a decorator that can be used to add support for dealing with\n    quantities to any __call__ method on a fitter which may not support\n    quantities itself. This is done by temporarily removing units from all\n    parameters then adding them back once the fitting has completed.\n    '

    @wraps(func)
    def wrapper(self, model, x, y, z=None, **kwargs):
        if False:
            i = 10
            return i + 15
        equivalencies = kwargs.pop('equivalencies', None)
        data_has_units = isinstance(x, Quantity) or isinstance(y, Quantity) or isinstance(z, Quantity)
        model_has_units = model._has_units
        if data_has_units or model_has_units:
            if model._supports_unit_fitting:
                input_units_equivalencies = _combine_equivalency_dict(model.inputs, equivalencies, model.input_units_equivalencies)
                if model.input_units is not None:
                    if isinstance(x, Quantity):
                        x = x.to(model.input_units[model.inputs[0]], equivalencies=input_units_equivalencies[model.inputs[0]])
                    if isinstance(y, Quantity) and z is not None:
                        y = y.to(model.input_units[model.inputs[1]], equivalencies=input_units_equivalencies[model.inputs[1]])
                rename_data = {model.inputs[0]: x}
                if z is not None:
                    rename_data[model.outputs[0]] = z
                    rename_data[model.inputs[1]] = y
                else:
                    rename_data[model.outputs[0]] = y
                    rename_data['z'] = None
                model = model.without_units_for_data(**rename_data)
                if isinstance(model, tuple):
                    rename_data['_left_kwargs'] = model[1]
                    rename_data['_right_kwargs'] = model[2]
                    model = model[0]
                add_back_units = False
                if isinstance(x, Quantity):
                    add_back_units = True
                    xdata = x.value
                else:
                    xdata = np.asarray(x)
                if isinstance(y, Quantity):
                    add_back_units = True
                    ydata = y.value
                else:
                    ydata = np.asarray(y)
                if z is not None:
                    if isinstance(z, Quantity):
                        add_back_units = True
                        zdata = z.value
                    else:
                        zdata = np.asarray(z)
                if z is None:
                    model_new = func(self, model, xdata, ydata, **kwargs)
                else:
                    model_new = func(self, model, xdata, ydata, zdata, **kwargs)
                if add_back_units:
                    model_new = model_new.with_units_from_data(**rename_data)
                return model_new
            else:
                raise NotImplementedError('This model does not support being fit to data with units.')
        else:
            return func(self, model, x, y, z=z, **kwargs)
    return wrapper

class Fitter(metaclass=_FitterMeta):
    """
    Base class for all fitters.

    Parameters
    ----------
    optimizer : callable
        A callable implementing an optimization algorithm
    statistic : callable
        Statistic function

    """
    supported_constraints = []

    def __init__(self, optimizer, statistic):
        if False:
            i = 10
            return i + 15
        if optimizer is None:
            raise ValueError('Expected an optimizer.')
        if statistic is None:
            raise ValueError('Expected a statistic function.')
        if isinstance(optimizer, type):
            self._opt_method = optimizer()
        elif inspect.isfunction(optimizer):
            self._opt_method = optimizer
        else:
            raise ValueError('Expected optimizer to be a callable class or a function.')
        if isinstance(statistic, type):
            self._stat_method = statistic()
        else:
            self._stat_method = statistic

    def objective_function(self, fps, *args):
        if False:
            print('Hello World!')
        '\n        Function to minimize.\n\n        Parameters\n        ----------\n        fps : list\n            parameters returned by the fitter\n        args : list\n            [model, [other_args], [input coordinates]]\n            other_args may include weights or any other quantities specific for\n            a statistic\n\n        Notes\n        -----\n        The list of arguments (args) is set in the `__call__` method.\n        Fitters may overwrite this method, e.g. when statistic functions\n        require other arguments.\n\n        '
        model = args[0]
        meas = args[-1]
        fitter_to_model_params(model, fps)
        res = self._stat_method(meas, model, *args[1:-1])
        return res

    @staticmethod
    def _add_fitting_uncertainties(*args):
        if False:
            for i in range(10):
                print('nop')
        '\n        When available, calculate and sets the parameter covariance matrix\n        (model.cov_matrix) and standard deviations (model.stds).\n        '
        return None

    @abc.abstractmethod
    def __call__(self):
        if False:
            return 10
        '\n        This method performs the actual fitting and modifies the parameter list\n        of a model.\n        Fitter subclasses should implement this method.\n        '
        raise NotImplementedError('Subclasses should implement this method.')

class LinearLSQFitter(metaclass=_FitterMeta):
    """
    A class performing a linear least square fitting.
    Uses `numpy.linalg.lstsq` to do the fitting.
    Given a model and data, fits the model to the data and changes the
    model's parameters. Keeps a dictionary of auxiliary fitting information.

    Notes
    -----
    Note that currently LinearLSQFitter does not support compound models.
    """
    supported_constraints = ['fixed']
    supports_masked_input = True

    def __init__(self, calc_uncertainties=False):
        if False:
            return 10
        self.fit_info = {'residuals': None, 'rank': None, 'singular_values': None, 'params': None}
        self._calc_uncertainties = calc_uncertainties

    @staticmethod
    def _is_invertible(m):
        if False:
            return 10
        'Check if inverse of matrix can be obtained.'
        if m.shape[0] != m.shape[1]:
            return False
        if np.linalg.matrix_rank(m) < m.shape[0]:
            return False
        return True

    def _add_fitting_uncertainties(self, model, a, n_coeff, x, y, z=None, resids=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate and parameter covariance matrix and standard deviations\n        and set `cov_matrix` and `stds` attributes.\n        '
        x_dot_x_prime = np.dot(a.T, a)
        masked = False or hasattr(y, 'mask')
        if not self._is_invertible(x_dot_x_prime):
            return model
        inv_x_dot_x_prime = np.linalg.inv(x_dot_x_prime)
        if z is None:
            if len(model) == 1:
                mask = None
                if masked:
                    mask = y.mask
                xx = np.ma.array(x, mask=mask)
                RSS = [1 / (xx.count() - n_coeff) * resids]
            if len(model) > 1:
                RSS = []
                for j in range(len(model)):
                    mask = None
                    if masked:
                        mask = y.mask[..., j].flatten()
                    xx = np.ma.array(x, mask=mask)
                    eval_y = model(xx, model_set_axis=False)
                    eval_y = np.rollaxis(eval_y, model.model_set_axis)[j]
                    RSS.append(1 / (xx.count() - n_coeff) * np.sum((y[..., j] - eval_y) ** 2))
        elif len(model) == 1:
            mask = None
            if masked:
                warnings.warn('Calculation of fitting uncertainties for 2D models with masked values not currently supported.\n', AstropyUserWarning)
                return
            (xx, _) = (np.ma.array(x, mask=mask), np.ma.array(y, mask=mask))
            RSS = [1 / (len(xx) - n_coeff) * resids]
        else:
            RSS = []
            for j in range(len(model)):
                eval_z = model(x, y, model_set_axis=False)
                mask = None
                if model.model_set_axis == 1:
                    eval_z = np.rollaxis(eval_z, 1)
                eval_z = eval_z[j]
                RSS.append([1 / (len(x) - n_coeff) * np.sum((z[j] - eval_z) ** 2)])
        covs = [inv_x_dot_x_prime * r for r in RSS]
        free_param_names = [x for x in model.fixed if model.fixed[x] is False and model.tied[x] is False]
        if len(covs) == 1:
            model.cov_matrix = Covariance(covs[0], model.param_names)
            model.stds = StandardDeviations(covs[0], free_param_names)
        else:
            model.cov_matrix = [Covariance(cov, model.param_names) for cov in covs]
            model.stds = [StandardDeviations(cov, free_param_names) for cov in covs]

    @staticmethod
    def _deriv_with_constraints(model, param_indices, x=None, y=None):
        if False:
            while True:
                i = 10
        if y is None:
            d = np.array(model.fit_deriv(x, *model.parameters))
        else:
            d = np.array(model.fit_deriv(x, y, *model.parameters))
        if model.col_fit_deriv:
            return d[param_indices]
        else:
            return d[..., param_indices]

    def _map_domain_window(self, model, x, y=None):
        if False:
            i = 10
            return i + 15
        '\n        Maps domain into window for a polynomial model which has these\n        attributes.\n        '
        if y is None:
            if hasattr(model, 'domain') and model.domain is None:
                model.domain = [x.min(), x.max()]
            if hasattr(model, 'window') and model.window is None:
                model.window = [-1, 1]
            return poly_map_domain(x, model.domain, model.window)
        else:
            if hasattr(model, 'x_domain') and model.x_domain is None:
                model.x_domain = [x.min(), x.max()]
            if hasattr(model, 'y_domain') and model.y_domain is None:
                model.y_domain = [y.min(), y.max()]
            if hasattr(model, 'x_window') and model.x_window is None:
                model.x_window = [-1.0, 1.0]
            if hasattr(model, 'y_window') and model.y_window is None:
                model.y_window = [-1.0, 1.0]
            xnew = poly_map_domain(x, model.x_domain, model.x_window)
            ynew = poly_map_domain(y, model.y_domain, model.y_window)
            return (xnew, ynew)

    @fitter_unit_support
    def __call__(self, model, x, y, z=None, weights=None, rcond=None):
        if False:
            while True:
                i = 10
        '\n        Fit data to this model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.FittableModel`\n            model to fit to x, y, z\n        x : array\n            Input coordinates\n        y : array-like\n            Input coordinates\n        z : array-like, optional\n            Input coordinates.\n            If the dependent (``y`` or ``z``) coordinate values are provided\n            as a `numpy.ma.MaskedArray`, any masked points are ignored when\n            fitting. Note that model set fitting is significantly slower when\n            there are masked points (not just an empty mask), as the matrix\n            equation has to be solved for each model separately when their\n            coordinate grids differ.\n        weights : array, optional\n            Weights for fitting.\n            For data with Gaussian uncertainties, the weights should be\n            1/sigma.\n        rcond :  float, optional\n            Cut-off ratio for small singular values of ``a``.\n            Singular values are set to zero if they are smaller than ``rcond``\n            times the largest singular value of ``a``.\n        equivalencies : list or None, optional, keyword-only\n            List of *additional* equivalencies that are should be applied in\n            case x, y and/or z have units. Default is None.\n\n        Returns\n        -------\n        model_copy : `~astropy.modeling.FittableModel`\n            a copy of the input model with parameters set by the fitter\n\n        '
        if not model.fittable:
            raise ValueError('Model must be a subclass of FittableModel')
        if not model.linear:
            raise ModelLinearityError('Model is not linear in parameters, linear fit methods should not be used.')
        if hasattr(model, 'submodel_names'):
            raise ValueError('Model must be simple, not compound')
        _validate_constraints(self.supported_constraints, model)
        model_copy = model.copy()
        model_copy.sync_constraints = False
        (_, fitparam_indices, _) = model_to_fit_params(model_copy)
        if model_copy.n_inputs == 2 and z is None:
            raise ValueError('Expected x, y and z for a 2 dimensional model.')
        farg = _convert_input(x, y, z, n_models=len(model_copy), model_set_axis=model_copy.model_set_axis)
        n_fixed = sum(model_copy.fixed.values())
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
        if n_fixed:
            fixparam_indices = [idx for idx in range(len(model_copy.param_names)) if idx not in fitparam_indices]
            fixparams = np.asarray([getattr(model_copy, model_copy.param_names[idx]).value for idx in fixparam_indices])
        if len(farg) == 2:
            (x, y) = farg
            if weights is not None:
                (_, weights) = _convert_input(x, weights, n_models=len(model_copy) if weights.ndim == y.ndim else 1, model_set_axis=model_copy.model_set_axis)
            if hasattr(model_copy, 'domain'):
                x = self._map_domain_window(model_copy, x)
            if n_fixed:
                lhs = np.asarray(self._deriv_with_constraints(model_copy, fitparam_indices, x=x))
                fixderivs = self._deriv_with_constraints(model_copy, fixparam_indices, x=x)
            else:
                lhs = np.asarray(model_copy.fit_deriv(x, *model_copy.parameters))
            sum_of_implicit_terms = model_copy.sum_of_implicit_terms(x)
            rhs = y
        else:
            (x, y, z) = farg
            if weights is not None:
                (_, _, weights) = _convert_input(x, y, weights, n_models=len(model_copy) if weights.ndim == z.ndim else 1, model_set_axis=model_copy.model_set_axis)
            if hasattr(model_copy, 'x_domain'):
                (x, y) = self._map_domain_window(model_copy, x, y)
            if n_fixed:
                lhs = np.asarray(self._deriv_with_constraints(model_copy, fitparam_indices, x=x, y=y))
                fixderivs = self._deriv_with_constraints(model_copy, fixparam_indices, x=x, y=y)
            else:
                lhs = np.asanyarray(model_copy.fit_deriv(x, y, *model_copy.parameters))
            sum_of_implicit_terms = model_copy.sum_of_implicit_terms(x, y)
            if len(model_copy) > 1:
                model_axis = model_copy.model_set_axis or 0
                if z.ndim > 2:
                    rhs = np.rollaxis(z, model_axis, z.ndim)
                    rhs = rhs.reshape(-1, rhs.shape[-1])
                else:
                    rhs = z.T if model_axis == 0 else z
                if weights is not None:
                    if weights.ndim > 2:
                        weights = np.rollaxis(weights, model_axis, weights.ndim)
                        weights = weights.reshape(-1, weights.shape[-1])
                    elif weights.ndim == z.ndim:
                        weights = weights.T if model_axis == 0 else weights
                    else:
                        weights = weights.flatten()
            else:
                rhs = z.flatten()
                if weights is not None:
                    weights = weights.flatten()
        if model_copy.col_fit_deriv:
            lhs = np.asarray(lhs).T
        if np.asanyarray(lhs).ndim > 2:
            raise ValueError(f'{type(model_copy).__name__} gives unsupported >2D derivative matrix for this x/y')
        if n_fixed:
            if model_copy.col_fit_deriv:
                fixderivs = np.asarray(fixderivs).T
            rhs = rhs - fixderivs.dot(fixparams)
        if sum_of_implicit_terms is not None:
            if len(model_copy) > 1:
                sum_of_implicit_terms = sum_of_implicit_terms[..., np.newaxis]
            rhs = rhs - sum_of_implicit_terms
        if weights is not None:
            if rhs.ndim == 2:
                if weights.shape == rhs.shape:
                    lhs = lhs[..., np.newaxis] * weights[:, np.newaxis]
                    rhs = rhs * weights
                else:
                    lhs *= weights[:, np.newaxis]
                    rhs = rhs * weights[:, np.newaxis]
            else:
                lhs *= weights[:, np.newaxis]
                rhs = rhs * weights
        scl = (lhs * lhs).sum(0)
        lhs /= scl
        masked = np.any(np.ma.getmask(rhs))
        if weights is not None and (not masked) and np.any(np.isnan(lhs)):
            raise ValueError('Found NaNs in the coefficient matrix, which should not happen and would crash the lapack routine. Maybe check that weights are not null.')
        a = None
        if masked and len(model_copy) > 1 or (weights is not None and weights.ndim > 1):
            lacoef = np.zeros(lhs.shape[1:2] + rhs.shape[-1:], dtype=rhs.dtype)
            if lhs.ndim > 2:
                lhs_stack = np.rollaxis(lhs, -1, 0)
            else:
                lhs_stack = np.broadcast_to(lhs, rhs.shape[-1:] + lhs.shape)
            for (model_lhs, model_rhs, model_lacoef) in zip(lhs_stack, rhs.T, lacoef.T):
                good = ~model_rhs.mask if masked else slice(None)
                model_lhs = model_lhs[good]
                model_rhs = model_rhs[good][..., np.newaxis]
                a = model_lhs
                (t_coef, resids, rank, sval) = np.linalg.lstsq(model_lhs, model_rhs, rcond)
                model_lacoef[:] = t_coef.T
        else:
            good = ~rhs.mask if masked else slice(None)
            a = lhs[good]
            (lacoef, resids, rank, sval) = np.linalg.lstsq(lhs[good], rhs[good], rcond)
        self.fit_info['residuals'] = resids
        self.fit_info['rank'] = rank
        self.fit_info['singular_values'] = sval
        lacoef /= scl[:, np.newaxis] if scl.ndim < rhs.ndim else scl
        self.fit_info['params'] = lacoef
        fitter_to_model_params(model_copy, lacoef.flatten())
        if hasattr(model_copy, '_order') and len(model_copy) == 1 and (rank < model_copy._order - n_fixed):
            warnings.warn('The fit may be poorly conditioned\n', AstropyUserWarning)
        if self._calc_uncertainties:
            if len(y) > len(lacoef):
                self._add_fitting_uncertainties(model_copy, a * scl, len(lacoef), x, y, z, resids)
        model_copy.sync_constraints = True
        return model_copy

class FittingWithOutlierRemoval:
    """
    This class combines an outlier removal technique with a fitting procedure.
    Basically, given a maximum number of iterations ``niter``, outliers are
    removed and fitting is performed for each iteration, until no new outliers
    are found or ``niter`` is reached.

    Parameters
    ----------
    fitter : `Fitter`
        An instance of any Astropy fitter, i.e., LinearLSQFitter,
        LevMarLSQFitter, SLSQPLSQFitter, SimplexLSQFitter, JointFitter. For
        model set fitting, this must understand masked input data (as
        indicated by the fitter class attribute ``supports_masked_input``).
    outlier_func : callable
        A function for outlier removal.
        If this accepts an ``axis`` parameter like the `numpy` functions, the
        appropriate value will be supplied automatically when fitting model
        sets (unless overridden in ``outlier_kwargs``), to find outliers for
        each model separately; otherwise, the same filtering must be performed
        in a loop over models, which is almost an order of magnitude slower.
    niter : int, optional
        Maximum number of iterations.
    outlier_kwargs : dict, optional
        Keyword arguments for outlier_func.

    Attributes
    ----------
    fit_info : dict
        The ``fit_info`` (if any) from the last iteration of the wrapped
        ``fitter`` during the most recent fit. An entry is also added with the
        keyword ``niter`` that records the actual number of fitting iterations
        performed (as opposed to the user-specified maximum).
    """

    def __init__(self, fitter, outlier_func, niter=3, **outlier_kwargs):
        if False:
            while True:
                i = 10
        self.fitter = fitter
        self.outlier_func = outlier_func
        self.niter = niter
        self.outlier_kwargs = outlier_kwargs
        self.fit_info = {'niter': None}

    def __str__(self):
        if False:
            while True:
                i = 10
        return f'Fitter: {self.fitter.__class__.__name__}\nOutlier function: {self.outlier_func.__name__}\nNum. of iterations: {self.niter}\nOutlier func. args.: {self.outlier_kwargs}'

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}(fitter: {self.fitter.__class__.__name__}, outlier_func: {self.outlier_func.__name__}, niter: {self.niter}, outlier_kwargs: {self.outlier_kwargs})'

    def __call__(self, model, x, y, z=None, weights=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Parameters\n        ----------\n        model : `~astropy.modeling.FittableModel`\n            An analytic model which will be fit to the provided data.\n            This also contains the initial guess for an optimization\n            algorithm.\n        x : array-like\n            Input coordinates.\n        y : array-like\n            Data measurements (1D case) or input coordinates (2D case).\n        z : array-like, optional\n            Data measurements (2D case).\n        weights : array-like, optional\n            Weights to be passed to the fitter.\n        kwargs : dict, optional\n            Keyword arguments to be passed to the fitter.\n\n        Returns\n        -------\n        fitted_model : `~astropy.modeling.FittableModel`\n            Fitted model after outlier removal.\n        mask : `numpy.ndarray`\n            Boolean mask array, identifying which points were used in the final\n            fitting iteration (False) and which were found to be outliers or\n            were masked in the input (True).\n        '
        if len(model) == 1:
            model_set_axis = None
        else:
            if not hasattr(self.fitter, 'supports_masked_input') or self.fitter.supports_masked_input is not True:
                raise ValueError(f'{type(self.fitter).__name__} cannot fit model sets with masked values')
            model_set_axis = model.model_set_axis
        if z is None:
            coords = (x,)
            data = y
        else:
            coords = (x, y)
            data = z
        if model_set_axis is not None:
            if model_set_axis < 0:
                model_set_axis += data.ndim
            if 'axis' not in self.outlier_kwargs:
                self.outlier_kwargs['axis'] = tuple((n for n in range(data.ndim) if n != model_set_axis))
        loop = False
        fitted_model = self.fitter(model, x, y, z, weights=weights, **kwargs)
        filtered_data = np.ma.masked_array(data)
        if filtered_data.mask is np.ma.nomask:
            filtered_data.mask = False
        filtered_weights = weights
        last_n_masked = filtered_data.mask.sum()
        n = 0
        for n in range(1, self.niter + 1):
            model_vals = fitted_model(*coords, model_set_axis=False)
            if not loop:
                try:
                    filtered_data = self.outlier_func(filtered_data - model_vals, **self.outlier_kwargs)
                except TypeError:
                    if model_set_axis is None:
                        raise
                    else:
                        self.outlier_kwargs.pop('axis', None)
                        loop = True
                        filtered_data = np.ma.masked_array(filtered_data, dtype=np.result_type(filtered_data, model_vals), copy=True)
                        if filtered_data.mask is np.ma.nomask:
                            filtered_data.mask = False
                        data_T = np.rollaxis(filtered_data, model_set_axis, 0)
                        mask_T = np.rollaxis(filtered_data.mask, model_set_axis, 0)
            if loop:
                model_vals_T = np.rollaxis(model_vals, model_set_axis, 0)
                for (row_data, row_mask, row_mod_vals) in zip(data_T, mask_T, model_vals_T):
                    masked_residuals = self.outlier_func(row_data - row_mod_vals, **self.outlier_kwargs)
                    row_data.data[:] = masked_residuals.data
                    row_mask[:] = masked_residuals.mask
                warnings.warn('outlier_func did not accept axis argument; reverted to slow loop over models.', AstropyUserWarning)
            filtered_data += model_vals
            if model_set_axis is None:
                good = ~filtered_data.mask
                if weights is not None:
                    filtered_weights = weights[good]
                fitted_model = self.fitter(fitted_model, *(c[good] for c in coords), filtered_data.data[good], weights=filtered_weights, **kwargs)
            else:
                fitted_model = self.fitter(fitted_model, *coords, filtered_data, weights=filtered_weights, **kwargs)
            this_n_masked = filtered_data.mask.sum()
            if this_n_masked == last_n_masked:
                break
            last_n_masked = this_n_masked
        self.fit_info = {'niter': n}
        self.fit_info.update(getattr(self.fitter, 'fit_info', {}))
        return (fitted_model, filtered_data.mask)

class _NonLinearLSQFitter(metaclass=_FitterMeta):
    """
    Base class for Non-Linear least-squares fitters.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covarience matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds : bool
        If the set parameter bounds for a model will be enforced each given
        parameter while fitting via a simple min/max condition.
        Default: True
    """
    supported_constraints = ['fixed', 'tied', 'bounds']
    '\n    The constraint types supported by this fitter type.\n    '

    def __init__(self, calc_uncertainties=False, use_min_max_bounds=True):
        if False:
            while True:
                i = 10
        self.fit_info = None
        self._calc_uncertainties = calc_uncertainties
        self._use_min_max_bounds = use_min_max_bounds
        super().__init__()

    def objective_function(self, fps, *args):
        if False:
            return 10
        '\n        Function to minimize.\n\n        Parameters\n        ----------\n        fps : list\n            parameters returned by the fitter\n        args : list\n            [model, [weights], [input coordinates]]\n\n        '
        model = args[0]
        weights = args[1]
        fitter_to_model_params(model, fps, self._use_min_max_bounds)
        meas = args[-1]
        if weights is None:
            value = np.ravel(model(*args[2:-1]) - meas)
        else:
            value = np.ravel(weights * (model(*args[2:-1]) - meas))
        if not np.all(np.isfinite(value)):
            raise NonFiniteValueError('Objective function has encountered a non-finite value, this will cause the fit to fail!\nPlease remove non-finite values from your input data before fitting to avoid this error.')
        return value

    @staticmethod
    def _add_fitting_uncertainties(model, cov_matrix):
        if False:
            print('Hello World!')
        '\n        Set ``cov_matrix`` and ``stds`` attributes on model with parameter\n        covariance matrix returned by ``optimize.leastsq``.\n        '
        free_param_names = [x for x in model.fixed if model.fixed[x] is False and model.tied[x] is False]
        model.cov_matrix = Covariance(cov_matrix, free_param_names)
        model.stds = StandardDeviations(cov_matrix, free_param_names)

    @staticmethod
    def _wrap_deriv(params, model, weights, x, y, z=None):
        if False:
            return 10
        '\n        Wraps the method calculating the Jacobian of the function to account\n        for model constraints.\n        `scipy.optimize.leastsq` expects the function derivative to have the\n        above signature (parlist, (argtuple)). In order to accommodate model\n        constraints, instead of using p directly, we set the parameter list in\n        this function.\n        '
        if weights is None:
            weights = 1.0
        if any(model.fixed.values()) or any(model.tied.values()):
            fitter_to_model_params(model, params)
            if z is None:
                full = np.array(model.fit_deriv(x, *model.parameters))
                if not model.col_fit_deriv:
                    full_deriv = np.ravel(weights) * full.T
                else:
                    full_deriv = np.ravel(weights) * full
            else:
                full = np.array([np.ravel(_) for _ in model.fit_deriv(x, y, *model.parameters)])
                if not model.col_fit_deriv:
                    full_deriv = np.ravel(weights) * full.T
                else:
                    full_deriv = np.ravel(weights) * full
            pars = [getattr(model, name) for name in model.param_names]
            fixed = [par.fixed for par in pars]
            tied = [par.tied for par in pars]
            tied = list(np.where([par.tied is not False for par in pars], True, tied))
            fix_and_tie = np.logical_or(fixed, tied)
            ind = np.logical_not(fix_and_tie)
            if not model.col_fit_deriv:
                residues = np.asarray(full_deriv[np.nonzero(ind)]).T
            else:
                residues = full_deriv[np.nonzero(ind)]
            return [np.ravel(_) for _ in residues]
        elif z is None:
            fit_deriv = np.array(model.fit_deriv(x, *params))
            try:
                output = np.array([np.ravel(_) for _ in np.array(weights) * fit_deriv])
                if output.shape != fit_deriv.shape:
                    output = np.array([np.ravel(_) for _ in np.atleast_2d(weights).T * fit_deriv])
                return output
            except ValueError:
                return np.array([np.ravel(_) for _ in np.array(weights) * np.moveaxis(fit_deriv, -1, 0)]).transpose()
        else:
            if not model.col_fit_deriv:
                return [np.ravel(_) for _ in (np.ravel(weights) * np.array(model.fit_deriv(x, y, *params)).T).T]
            return [np.ravel(_) for _ in weights * np.array(model.fit_deriv(x, y, *params))]

    def _compute_param_cov(self, model, y, init_values, cov_x, fitparams, farg, weights=None):
        if False:
            while True:
                i = 10
        if len(y) > len(init_values) and cov_x is not None:
            self.fit_info['param_cov'] = cov_x
            if weights is None:
                sum_sqrs = np.sum(self.objective_function(fitparams, *farg) ** 2)
                dof = len(y) - len(init_values)
                self.fit_info['param_cov'] *= sum_sqrs / dof
        else:
            self.fit_info['param_cov'] = None
        if self._calc_uncertainties is True:
            if self.fit_info['param_cov'] is not None:
                self._add_fitting_uncertainties(model, self.fit_info['param_cov'])

    def _run_fitter(self, model, farg, maxiter, acc, epsilon, estimate_jacobian):
        if False:
            while True:
                i = 10
        return (None, None, None)

    def _filter_non_finite(self, x, y, z=None, weights=None):
        if False:
            print('Hello World!')
        '\n        Filter out non-finite values in x, y, z.\n\n        Returns\n        -------\n        x, y, z : ndarrays\n            x, y, and z with non-finite values filtered out.\n        '
        MESSAGE = 'Non-Finite input data has been removed by the fitter.'
        mask = np.ones_like(x, dtype=bool) if weights is None else np.isfinite(weights)
        mask &= np.isfinite(y) if z is None else np.isfinite(z)
        if not np.all(mask):
            warnings.warn(MESSAGE, AstropyUserWarning)
        return (x[mask], y[mask], None if z is None else z[mask], None if weights is None else weights[mask])

    @fitter_unit_support
    def __call__(self, model, x, y, z=None, weights=None, maxiter=DEFAULT_MAXITER, acc=DEFAULT_ACC, epsilon=DEFAULT_EPS, estimate_jacobian=False, filter_non_finite=False):
        if False:
            print('Hello World!')
        '\n        Fit data to this model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.FittableModel`\n            model to fit to x, y, z\n        x : array\n           input coordinates\n        y : array\n           input coordinates\n        z : array, optional\n           input coordinates\n        weights : array, optional\n            Weights for fitting. For data with Gaussian uncertainties, the weights\n            should be 1/sigma.\n\n            .. versionchanged:: 5.3\n                Calculate parameter covariances while accounting for ``weights``\n                as "absolute" inverse uncertainties. To recover the old behavior,\n                choose ``weights=None``.\n\n        maxiter : int\n            maximum number of iterations\n        acc : float\n            Relative error desired in the approximate solution\n        epsilon : float\n            A suitable step length for the forward-difference\n            approximation of the Jacobian (if model.fjac=None). If\n            epsfcn is less than the machine precision, it is\n            assumed that the relative errors in the functions are\n            of the order of the machine precision.\n        estimate_jacobian : bool\n            If False (default) and if the model has a fit_deriv method,\n            it will be used. Otherwise the Jacobian will be estimated.\n            If True, the Jacobian will be estimated in any case.\n        equivalencies : list or None, optional, keyword-only\n            List of *additional* equivalencies that are should be applied in\n            case x, y and/or z have units. Default is None.\n        filter_non_finite : bool, optional\n            Whether or not to filter data with non-finite values. Default is False\n\n        Returns\n        -------\n        model_copy : `~astropy.modeling.FittableModel`\n            a copy of the input model with parameters set by the fitter\n\n        '
        model_copy = _validate_model(model, self.supported_constraints)
        model_copy.sync_constraints = False
        if filter_non_finite:
            (x, y, z, weights) = self._filter_non_finite(x, y, z, weights)
        farg = (model_copy, weights) + _convert_input(x, y, z)
        (init_values, fitparams, cov_x) = self._run_fitter(model_copy, farg, maxiter, acc, epsilon, estimate_jacobian)
        self._compute_param_cov(model_copy, y, init_values, cov_x, fitparams, farg, weights)
        model.sync_constraints = True
        return model_copy

class LevMarLSQFitter(_NonLinearLSQFitter):
    """
    Levenberg-Marquardt algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covarience matrix should be computed and set in the fit_info.
        Default: False

    Attributes
    ----------
    fit_info : dict
        The `scipy.optimize.leastsq` result for the most recent fit (see
        notes).

    Notes
    -----
    The ``fit_info`` dictionary contains the values returned by
    `scipy.optimize.leastsq` for the most recent fit, including the values from
    the ``infodict`` dictionary it returns. See the `scipy.optimize.leastsq`
    documentation for details on the meaning of these values. Note that the
    ``x`` return value is *not* included (as it is instead the parameter values
    of the returned model).
    Additionally, one additional element of ``fit_info`` is computed whenever a
    model is fit, with the key 'param_cov'. The corresponding value is the
    covariance matrix of the parameters as a 2D numpy array.  The order of the
    matrix elements matches the order of the parameters in the fitted model
    (i.e., the same order as ``model.param_names``).

    """

    def __init__(self, calc_uncertainties=False):
        if False:
            print('Hello World!')
        super().__init__(calc_uncertainties)
        self.fit_info = {'nfev': None, 'fvec': None, 'fjac': None, 'ipvt': None, 'qtf': None, 'message': None, 'ierr': None, 'param_jac': None, 'param_cov': None}

    def _run_fitter(self, model, farg, maxiter, acc, epsilon, estimate_jacobian):
        if False:
            print('Hello World!')
        from scipy import optimize
        if model.fit_deriv is None or estimate_jacobian:
            dfunc = None
        else:
            dfunc = self._wrap_deriv
        (init_values, _, _) = model_to_fit_params(model)
        (fitparams, cov_x, dinfo, mess, ierr) = optimize.leastsq(self.objective_function, init_values, args=farg, Dfun=dfunc, col_deriv=model.col_fit_deriv, maxfev=maxiter, epsfcn=epsilon, xtol=acc, full_output=True)
        fitter_to_model_params(model, fitparams)
        self.fit_info.update(dinfo)
        self.fit_info['cov_x'] = cov_x
        self.fit_info['message'] = mess
        self.fit_info['ierr'] = ierr
        if ierr not in [1, 2, 3, 4]:
            warnings.warn("The fit may be unsuccessful; check fit_info['message'] for more information.", AstropyUserWarning)
        return (init_values, fitparams, cov_x)

class _NLLSQFitter(_NonLinearLSQFitter):
    """
    Wrapper class for `scipy.optimize.least_squares` method, which provides:
        - Trust Region Reflective
        - dogbox
        - Levenberg-Marqueardt
    algorithms using the least squares statistic.

    Parameters
    ----------
    method : str
        ‘trf’ :  Trust Region Reflective algorithm, particularly suitable
            for large sparse problems with bounds. Generally robust method.
        ‘dogbox’ : dogleg algorithm with rectangular trust regions, typical
            use case is small problems with bounds. Not recommended for
            problems with rank-deficient Jacobian.
        ‘lm’ : Levenberg-Marquardt algorithm as implemented in MINPACK.
            Doesn’t handle bounds and sparse Jacobians. Usually the most
            efficient method for small unconstrained problems.
    calc_uncertainties : bool
        If the covarience matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds: bool
        If the set parameter bounds for a model will be enforced each given
        parameter while fitting via a simple min/max condition. A True setting
        will replicate how LevMarLSQFitter enforces bounds.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """

    def __init__(self, method, calc_uncertainties=False, use_min_max_bounds=False):
        if False:
            while True:
                i = 10
        super().__init__(calc_uncertainties, use_min_max_bounds)
        self._method = method

    def _run_fitter(self, model, farg, maxiter, acc, epsilon, estimate_jacobian):
        if False:
            i = 10
            return i + 15
        from scipy import optimize
        from scipy.linalg import svd
        if model.fit_deriv is None or estimate_jacobian:
            dfunc = '2-point'
        else:

            def _dfunc(params, model, weights, x, y, z=None):
                if False:
                    i = 10
                    return i + 15
                if model.col_fit_deriv:
                    return np.transpose(self._wrap_deriv(params, model, weights, x, y, z))
                else:
                    return self._wrap_deriv(params, model, weights, x, y, z)
            dfunc = _dfunc
        (init_values, _, bounds) = model_to_fit_params(model)
        if self._use_min_max_bounds:
            bounds = (-np.inf, np.inf)
        self.fit_info = optimize.least_squares(self.objective_function, init_values, args=farg, jac=dfunc, max_nfev=maxiter, diff_step=np.sqrt(epsilon), xtol=acc, method=self._method, bounds=bounds)
        (_, s, VT) = svd(self.fit_info.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(self.fit_info.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        cov_x = np.dot(VT.T / s ** 2, VT)
        fitter_to_model_params(model, self.fit_info.x, False)
        if not self.fit_info.success:
            warnings.warn(f'The fit may be unsuccessful; check: \n    {self.fit_info.message}', AstropyUserWarning)
        return (init_values, self.fit_info.x, cov_x)

class TRFLSQFitter(_NLLSQFitter):
    """
    Trust Region Reflective algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covarience matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds: bool
        If the set parameter bounds for a model will be enforced each given
        parameter while fitting via a simple min/max condition. A True setting
        will replicate how LevMarLSQFitter enforces bounds.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """

    def __init__(self, calc_uncertainties=False, use_min_max_bounds=False):
        if False:
            return 10
        super().__init__('trf', calc_uncertainties, use_min_max_bounds)

class DogBoxLSQFitter(_NLLSQFitter):
    """
    DogBox algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covarience matrix should be computed and set in the fit_info.
        Default: False
    use_min_max_bounds: bool
        If the set parameter bounds for a model will be enforced each given
        parameter while fitting via a simple min/max condition. A True setting
        will replicate how LevMarLSQFitter enforces bounds.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """

    def __init__(self, calc_uncertainties=False, use_min_max_bounds=False):
        if False:
            i = 10
            return i + 15
        super().__init__('dogbox', calc_uncertainties, use_min_max_bounds)

class LMLSQFitter(_NLLSQFitter):
    """
    `scipy.optimize.least_squares` Levenberg-Marquardt algorithm and least squares statistic.

    Parameters
    ----------
    calc_uncertainties : bool
        If the covarience matrix should be computed and set in the fit_info.
        Default: False

    Attributes
    ----------
    fit_info :
        A `scipy.optimize.OptimizeResult` class which contains all of
        the most recent fit information
    """

    def __init__(self, calc_uncertainties=False):
        if False:
            for i in range(10):
                print('nop')
        super().__init__('lm', calc_uncertainties, True)

class SLSQPLSQFitter(Fitter):
    """
    Sequential Least Squares Programming (SLSQP) optimization algorithm and
    least squares statistic.

    Raises
    ------
    ModelLinearityError
        A linear model is passed to a nonlinear fitter

    Notes
    -----
    See also the `~astropy.modeling.optimizers.SLSQP` optimizer.

    """
    supported_constraints = SLSQP.supported_constraints

    def __init__(self):
        if False:
            return 10
        super().__init__(optimizer=SLSQP, statistic=leastsquare)
        self.fit_info = {}

    @fitter_unit_support
    def __call__(self, model, x, y, z=None, weights=None, **kwargs):
        if False:
            return 10
        '\n        Fit data to this model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.FittableModel`\n            model to fit to x, y, z\n        x : array\n            input coordinates\n        y : array\n            input coordinates\n        z : array, optional\n            input coordinates\n        weights : array, optional\n            Weights for fitting.\n            For data with Gaussian uncertainties, the weights should be\n            1/sigma.\n        kwargs : dict\n            optional keyword arguments to be passed to the optimizer or the statistic\n        verblevel : int\n            0-silent\n            1-print summary upon completion,\n            2-print summary after each iteration\n        maxiter : int\n            maximum number of iterations\n        epsilon : float\n            the step size for finite-difference derivative estimates\n        acc : float\n            Requested accuracy\n        equivalencies : list or None, optional, keyword-only\n            List of *additional* equivalencies that are should be applied in\n            case x, y and/or z have units. Default is None.\n\n        Returns\n        -------\n        model_copy : `~astropy.modeling.FittableModel`\n            a copy of the input model with parameters set by the fitter\n\n        '
        model_copy = _validate_model(model, self._opt_method.supported_constraints)
        model_copy.sync_constraints = False
        farg = _convert_input(x, y, z)
        farg = (model_copy, weights) + farg
        (init_values, _, _) = model_to_fit_params(model_copy)
        (fitparams, self.fit_info) = self._opt_method(self.objective_function, init_values, farg, **kwargs)
        fitter_to_model_params(model_copy, fitparams)
        model_copy.sync_constraints = True
        return model_copy

class SimplexLSQFitter(Fitter):
    """
    Simplex algorithm and least squares statistic.

    Raises
    ------
    `ModelLinearityError`
        A linear model is passed to a nonlinear fitter

    """
    supported_constraints = Simplex.supported_constraints

    def __init__(self):
        if False:
            return 10
        super().__init__(optimizer=Simplex, statistic=leastsquare)
        self.fit_info = {}

    @fitter_unit_support
    def __call__(self, model, x, y, z=None, weights=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit data to this model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.FittableModel`\n            model to fit to x, y, z\n        x : array\n            input coordinates\n        y : array\n            input coordinates\n        z : array, optional\n            input coordinates\n        weights : array, optional\n            Weights for fitting.\n            For data with Gaussian uncertainties, the weights should be\n            1/sigma.\n        kwargs : dict\n            optional keyword arguments to be passed to the optimizer or the statistic\n        maxiter : int\n            maximum number of iterations\n        acc : float\n            Relative error in approximate solution\n        equivalencies : list or None, optional, keyword-only\n            List of *additional* equivalencies that are should be applied in\n            case x, y and/or z have units. Default is None.\n\n        Returns\n        -------\n        model_copy : `~astropy.modeling.FittableModel`\n            a copy of the input model with parameters set by the fitter\n\n        '
        model_copy = _validate_model(model, self._opt_method.supported_constraints)
        model_copy.sync_constraints = False
        farg = _convert_input(x, y, z)
        farg = (model_copy, weights) + farg
        (init_values, _, _) = model_to_fit_params(model_copy)
        (fitparams, self.fit_info) = self._opt_method(self.objective_function, init_values, farg, **kwargs)
        fitter_to_model_params(model_copy, fitparams)
        model_copy.sync_constraints = True
        return model_copy

class JointFitter(metaclass=_FitterMeta):
    """
    Fit models which share a parameter.
    For example, fit two gaussians to two data sets but keep
    the FWHM the same.

    Parameters
    ----------
    models : list
        a list of model instances
    jointparameters : list
        a list of joint parameters
    initvals : list
        a list of initial values

    """

    def __init__(self, models, jointparameters, initvals):
        if False:
            i = 10
            return i + 15
        self.models = list(models)
        self.initvals = list(initvals)
        self.jointparams = jointparameters
        self._verify_input()
        self.fitparams = self.model_to_fit_params()
        self.modeldims = [m.n_inputs for m in self.models]
        self.ndim = np.sum(self.modeldims)

    def model_to_fit_params(self):
        if False:
            return 10
        fparams = []
        fparams.extend(self.initvals)
        for model in self.models:
            params = model.parameters.tolist()
            joint_params = self.jointparams[model]
            param_metrics = model._param_metrics
            for param_name in joint_params:
                slice_ = param_metrics[param_name]['slice']
                del params[slice_]
            fparams.extend(params)
        return fparams

    def objective_function(self, fps, *args):
        if False:
            i = 10
            return i + 15
        '\n        Function to minimize.\n\n        Parameters\n        ----------\n        fps : list\n            the fitted parameters - result of an one iteration of the\n            fitting algorithm\n        args : dict\n            tuple of measured and input coordinates\n            args is always passed as a tuple from optimize.leastsq\n\n        '
        lstsqargs = list(args)
        fitted = []
        fitparams = list(fps)
        numjp = len(self.initvals)
        jointfitparams = fitparams[:numjp]
        del fitparams[:numjp]
        for model in self.models:
            joint_params = self.jointparams[model]
            margs = lstsqargs[:model.n_inputs + 1]
            del lstsqargs[:model.n_inputs + 1]
            numfp = len(model._parameters) - len(joint_params)
            mfparams = fitparams[:numfp]
            del fitparams[:numfp]
            mparams = []
            param_metrics = model._param_metrics
            for param_name in model.param_names:
                if param_name in joint_params:
                    index = joint_params.index(param_name)
                    mparams.extend([jointfitparams[index]])
                else:
                    slice_ = param_metrics[param_name]['slice']
                    plen = slice_.stop - slice_.start
                    mparams.extend(mfparams[:plen])
                    del mfparams[:plen]
            modelfit = model.evaluate(margs[:-1], *mparams)
            fitted.extend(modelfit - margs[-1])
        return np.ravel(fitted)

    def _verify_input(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self.models) <= 1:
            raise TypeError(f'Expected >1 models, {len(self.models)} is given')
        if len(self.jointparams.keys()) < 2:
            raise TypeError(f'At least two parameters are expected, {len(self.jointparams.keys())} is given')
        for j in self.jointparams.keys():
            if len(self.jointparams[j]) != len(self.initvals):
                raise TypeError(f'{len(self.jointparams[j])} parameter(s) provided but {len(self.initvals)} expected')

    def __call__(self, *args):
        if False:
            return 10
        '\n        Fit data to these models keeping some of the parameters common to the\n        two models.\n        '
        from scipy import optimize
        if len(args) != reduce(lambda x, y: x + 1 + y + 1, self.modeldims):
            raise ValueError(f'Expected {reduce(lambda x, y: x + 1 + y + 1, self.modeldims)} coordinates in args but {len(args)} provided')
        (self.fitparams[:], _) = optimize.leastsq(self.objective_function, self.fitparams, args=args)
        fparams = self.fitparams[:]
        numjp = len(self.initvals)
        jointfitparams = fparams[:numjp]
        del fparams[:numjp]
        for model in self.models:
            joint_params = self.jointparams[model]
            numfp = len(model._parameters) - len(joint_params)
            mfparams = fparams[:numfp]
            del fparams[:numfp]
            mparams = []
            param_metrics = model._param_metrics
            for param_name in model.param_names:
                if param_name in joint_params:
                    index = joint_params.index(param_name)
                    mparams.extend([jointfitparams[index]])
                else:
                    slice_ = param_metrics[param_name]['slice']
                    plen = slice_.stop - slice_.start
                    mparams.extend(mfparams[:plen])
                    del mfparams[:plen]
            model.parameters = np.array(mparams)

def _convert_input(x, y, z=None, n_models=1, model_set_axis=0):
    if False:
        while True:
            i = 10
    'Convert inputs to float arrays.'
    x = np.asanyarray(x, dtype=float)
    y = np.asanyarray(y, dtype=float)
    if z is not None:
        z = np.asanyarray(z, dtype=float)
        (data_ndim, data_shape) = (z.ndim, z.shape)
    else:
        (data_ndim, data_shape) = (y.ndim, y.shape)
    if n_models > 1 or data_ndim > x.ndim:
        if (model_set_axis or 0) >= data_ndim:
            raise ValueError('model_set_axis out of range')
        if data_shape[model_set_axis] != n_models:
            raise ValueError('Number of data sets (y or z array) is expected to equal the number of parameter sets')
        if z is None:
            y = np.rollaxis(y, model_set_axis, y.ndim)
            data_shape = y.shape[:-1]
        else:
            data_shape = z.shape[:model_set_axis] + z.shape[model_set_axis + 1:]
    if z is None:
        if data_shape != x.shape:
            raise ValueError('x and y should have the same shape')
        farg = (x, y)
    else:
        if not x.shape == y.shape == data_shape:
            raise ValueError('x, y and z should have the same shape')
        farg = (x, y, z)
    return farg

def fitter_to_model_params(model, fps, use_min_max_bounds=True):
    if False:
        print('Hello World!')
    '\n    Constructs the full list of model parameters from the fitted and\n    constrained parameters.\n\n    Parameters\n    ----------\n    model :\n        The model being fit\n    fps :\n        The fit parameter values to be assigned\n    use_min_max_bounds: bool\n        If the set parameter bounds for model will be enforced on each\n        parameter with bounds.\n        Default: True\n    '
    (_, fit_param_indices, _) = model_to_fit_params(model)
    has_tied = any(model.tied.values())
    has_fixed = any(model.fixed.values())
    has_bound = any((b != (None, None) for b in model.bounds.values()))
    parameters = model.parameters
    if not (has_tied or has_fixed or has_bound):
        model.parameters = fps
        return
    fit_param_indices = set(fit_param_indices)
    offset = 0
    param_metrics = model._param_metrics
    for (idx, name) in enumerate(model.param_names):
        if idx not in fit_param_indices:
            continue
        slice_ = param_metrics[name]['slice']
        shape = param_metrics[name]['shape']
        size = reduce(operator.mul, shape, 1)
        values = fps[offset:offset + size]
        if model.bounds[name] != (None, None) and use_min_max_bounds:
            (_min, _max) = model.bounds[name]
            if _min is not None:
                values = np.fmax(values, _min)
            if _max is not None:
                values = np.fmin(values, _max)
        parameters[slice_] = values
        offset += size
    model._array_to_parameters()
    if has_tied:
        for (idx, name) in enumerate(model.param_names):
            if model.tied[name]:
                value = model.tied[name](model)
                slice_ = param_metrics[name]['slice']
                parameters[slice_] = value
                model._array_to_parameters()

def model_to_fit_params(model):
    if False:
        i = 10
        return i + 15
    "\n    Convert a model instance's parameter array to an array that can be used\n    with a fitter that doesn't natively support fixed or tied parameters.\n    In particular, it removes fixed/tied parameters from the parameter\n    array.\n    These may be a subset of the model parameters, if some of them are held\n    constant or tied.\n    "
    fitparam_indices = list(range(len(model.param_names)))
    model_params = model.parameters
    model_bounds = list(model.bounds.values())
    if any(model.fixed.values()) or any(model.tied.values()):
        params = list(model_params)
        param_metrics = model._param_metrics
        for (idx, name) in list(enumerate(model.param_names))[::-1]:
            if model.fixed[name] or model.tied[name]:
                slice_ = param_metrics[name]['slice']
                del params[slice_]
                del model_bounds[slice_]
                del fitparam_indices[idx]
        model_params = np.array(params)
    for (idx, bound) in enumerate(model_bounds):
        if bound[0] is None:
            lower = -np.inf
        else:
            lower = bound[0]
        if bound[1] is None:
            upper = np.inf
        else:
            upper = bound[1]
        model_bounds[idx] = (lower, upper)
    model_bounds = tuple(zip(*model_bounds))
    return (model_params, fitparam_indices, model_bounds)

def _validate_constraints(supported_constraints, model):
    if False:
        return 10
    'Make sure model constraints are supported by the current fitter.'
    message = 'Optimizer cannot handle {0} constraints.'
    if any(model.fixed.values()) and 'fixed' not in supported_constraints:
        raise UnsupportedConstraintError(message.format('fixed parameter'))
    if any(model.tied.values()) and 'tied' not in supported_constraints:
        raise UnsupportedConstraintError(message.format('tied parameter'))
    if any((tuple(b) != (None, None) for b in model.bounds.values())) and 'bounds' not in supported_constraints:
        raise UnsupportedConstraintError(message.format('bound parameter'))
    if model.eqcons and 'eqcons' not in supported_constraints:
        raise UnsupportedConstraintError(message.format('equality'))
    if model.ineqcons and 'ineqcons' not in supported_constraints:
        raise UnsupportedConstraintError(message.format('inequality'))

def _validate_model(model, supported_constraints):
    if False:
        i = 10
        return i + 15
    '\n    Check that model and fitter are compatible and return a copy of the model.\n    '
    if not model.fittable:
        raise ValueError('Model does not appear to be fittable.')
    if model.linear:
        warnings.warn('Model is linear in parameters; consider using linear fitting methods.', AstropyUserWarning)
    elif len(model) != 1:
        raise ValueError('Non-linear fitters can only fit one data set at a time.')
    _validate_constraints(supported_constraints, model)
    model_copy = model.copy()
    return model_copy

def populate_entry_points(entry_points):
    if False:
        print('Hello World!')
    "\n    This injects entry points into the `astropy.modeling.fitting` namespace.\n    This provides a means of inserting a fitting routine without requirement\n    of it being merged into astropy's core.\n\n    Parameters\n    ----------\n    entry_points : list of `~importlib.metadata.EntryPoint`\n        entry_points are objects which encapsulate importable objects and\n        are defined on the installation of a package.\n\n    Notes\n    -----\n    An explanation of entry points can be found `here\n    <http://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`_\n    "
    for entry_point in entry_points:
        name = entry_point.name
        try:
            entry_point = entry_point.load()
        except Exception as e:
            warnings.warn(AstropyUserWarning(f'{type(e).__name__} error occurred in entry point {name}.'))
        else:
            if not isinstance(entry_point, type):
                warnings.warn(AstropyUserWarning(f'Modeling entry point {name} expected to be a Class.'))
            elif issubclass(entry_point, Fitter):
                name = entry_point.__name__
                globals()[name] = entry_point
                __all__.append(name)
            else:
                warnings.warn(AstropyUserWarning(f'Modeling entry point {name} expected to extend astropy.modeling.Fitter'))

def _populate_ep():
    if False:
        for i in range(10):
            print('nop')
    ep = entry_points()
    if hasattr(ep, 'select'):
        populate_entry_points(ep.select(group='astropy.modeling'))
    else:
        populate_entry_points(ep.get('astropy.modeling', []))
_populate_ep()