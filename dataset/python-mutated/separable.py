"""
Functions to determine if a model is separable, i.e.
if the model outputs are independent.

It analyzes ``n_inputs``, ``n_outputs`` and the operators
in a compound model by stepping through the transforms
and creating a ``coord_matrix`` of shape (``n_outputs``, ``n_inputs``).


Each modeling operator is represented by a function which
takes two simple models (or two ``coord_matrix`` arrays) and
returns an array of shape (``n_outputs``, ``n_inputs``).

"""
import numpy as np
from .core import CompoundModel, Model, ModelDefinitionError
from .mappings import Mapping
__all__ = ['is_separable', 'separability_matrix']

def is_separable(transform):
    if False:
        return 10
    '\n    A separability test for the outputs of a transform.\n\n    Parameters\n    ----------\n    transform : `~astropy.modeling.core.Model`\n        A (compound) model.\n\n    Returns\n    -------\n    is_separable : ndarray\n        A boolean array with size ``transform.n_outputs`` where\n        each element indicates whether the output is independent\n        and the result of a separable transform.\n\n    Examples\n    --------\n    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D\n    >>> is_separable(Shift(1) & Shift(2) | Scale(1) & Scale(2))\n        array([ True,  True]...)\n    >>> is_separable(Shift(1) & Shift(2) | Rotation2D(2))\n        array([False, False]...)\n    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) |         Polynomial2D(1) & Polynomial2D(2))\n        array([False, False]...)\n    >>> is_separable(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))\n        array([ True,  True,  True,  True]...)\n\n    '
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        is_separable = np.array([False] * transform.n_outputs).T
        return is_separable
    separable_matrix = _separable(transform)
    is_separable = separable_matrix.sum(1)
    is_separable = np.where(is_separable != 1, False, True)
    return is_separable

def separability_matrix(transform):
    if False:
        print('Hello World!')
    '\n    Compute the correlation between outputs and inputs.\n\n    Parameters\n    ----------\n    transform : `~astropy.modeling.core.Model`\n        A (compound) model.\n\n    Returns\n    -------\n    separable_matrix : ndarray\n        A boolean correlation matrix of shape (n_outputs, n_inputs).\n        Indicates the dependence of outputs on inputs. For completely\n        independent outputs, the diagonal elements are True and\n        off-diagonal elements are False.\n\n    Examples\n    --------\n    >>> from astropy.modeling.models import Shift, Scale, Rotation2D, Polynomial2D\n    >>> separability_matrix(Shift(1) & Shift(2) | Scale(1) & Scale(2))\n        array([[ True, False], [False,  True]]...)\n    >>> separability_matrix(Shift(1) & Shift(2) | Rotation2D(2))\n        array([[ True,  True], [ True,  True]]...)\n    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]) |         Polynomial2D(1) & Polynomial2D(2))\n        array([[ True,  True], [ True,  True]]...)\n    >>> separability_matrix(Shift(1) & Shift(2) | Mapping([0, 1, 0, 1]))\n        array([[ True, False], [False,  True], [ True, False], [False,  True]]...)\n\n    '
    if transform.n_inputs == 1 and transform.n_outputs > 1:
        return np.ones((transform.n_outputs, transform.n_inputs), dtype=np.bool_)
    separable_matrix = _separable(transform)
    separable_matrix = np.where(separable_matrix != 0, True, False)
    return separable_matrix

def _compute_n_outputs(left, right):
    if False:
        for i in range(10):
            print('nop')
    '\n    Compute the number of outputs of two models.\n\n    The two models are the left and right model to an operation in\n    the expression tree of a compound model.\n\n    Parameters\n    ----------\n    left, right : `astropy.modeling.Model` or ndarray\n        If input is of an array, it is the output of `coord_matrix`.\n\n    '
    if isinstance(left, Model):
        lnout = left.n_outputs
    else:
        lnout = left.shape[0]
    if isinstance(right, Model):
        rnout = right.n_outputs
    else:
        rnout = right.shape[0]
    noutp = lnout + rnout
    return noutp

def _arith_oper(left, right):
    if False:
        i = 10
        return i + 15
    "\n    Function corresponding to one of the arithmetic operators\n    ['+', '-'. '*', '/', '**'].\n\n    This always returns a nonseparable output.\n\n    Parameters\n    ----------\n    left, right : `astropy.modeling.Model` or ndarray\n        If input is of an array, it is the output of `coord_matrix`.\n\n    Returns\n    -------\n    result : ndarray\n        Result from this operation.\n    "

    def _n_inputs_outputs(input):
        if False:
            while True:
                i = 10
        if isinstance(input, Model):
            (n_outputs, n_inputs) = (input.n_outputs, input.n_inputs)
        else:
            (n_outputs, n_inputs) = input.shape
        return (n_inputs, n_outputs)
    (left_inputs, left_outputs) = _n_inputs_outputs(left)
    (right_inputs, right_outputs) = _n_inputs_outputs(right)
    if left_inputs != right_inputs or left_outputs != right_outputs:
        raise ModelDefinitionError(f'Unsupported operands for arithmetic operator: left (n_inputs={left_inputs}, n_outputs={left_outputs}) and right (n_inputs={right_inputs}, n_outputs={right_outputs}); models must have the same n_inputs and the same n_outputs for this operator.')
    result = np.ones((left_outputs, left_inputs))
    return result

def _coord_matrix(model, pos, noutp):
    if False:
        print('Hello World!')
    "\n    Create an array representing inputs and outputs of a simple model.\n\n    The array has a shape (noutp, model.n_inputs).\n\n    Parameters\n    ----------\n    model : `astropy.modeling.Model`\n        model\n    pos : str\n        Position of this model in the expression tree.\n        One of ['left', 'right'].\n    noutp : int\n        Number of outputs of the compound model of which the input model\n        is a left or right child.\n\n    "
    if isinstance(model, Mapping):
        axes = []
        for i in model.mapping:
            axis = np.zeros((model.n_inputs,))
            axis[i] = 1
            axes.append(axis)
        m = np.vstack(axes)
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[:model.n_outputs, :model.n_inputs] = m
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = m
        return mat
    if not model.separable:
        mat = np.zeros((noutp, model.n_inputs))
        if pos == 'left':
            mat[:model.n_outputs, :model.n_inputs] = 1
        else:
            mat[-model.n_outputs:, -model.n_inputs:] = 1
    else:
        mat = np.zeros((noutp, model.n_inputs))
        for i in range(model.n_inputs):
            mat[i, i] = 1
        if pos == 'right':
            mat = np.roll(mat, noutp - model.n_outputs)
    return mat

def _cstack(left, right):
    if False:
        while True:
            i = 10
    "\n    Function corresponding to '&' operation.\n\n    Parameters\n    ----------\n    left, right : `astropy.modeling.Model` or ndarray\n        If input is of an array, it is the output of `coord_matrix`.\n\n    Returns\n    -------\n    result : ndarray\n        Result from this operation.\n\n    "
    noutp = _compute_n_outputs(left, right)
    if isinstance(left, Model):
        cleft = _coord_matrix(left, 'left', noutp)
    else:
        cleft = np.zeros((noutp, left.shape[1]))
        cleft[:left.shape[0], :left.shape[1]] = left
    if isinstance(right, Model):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
        cright[-right.shape[0]:, -right.shape[1]:] = right
    return np.hstack([cleft, cright])

def _cdot(left, right):
    if False:
        for i in range(10):
            print('nop')
    '\n    Function corresponding to "|" operation.\n\n    Parameters\n    ----------\n    left, right : `astropy.modeling.Model` or ndarray\n        If input is of an array, it is the output of `coord_matrix`.\n\n    Returns\n    -------\n    result : ndarray\n        Result from this operation.\n    '
    (left, right) = (right, left)

    def _n_inputs_outputs(input, position):
        if False:
            while True:
                i = 10
        '\n        Return ``n_inputs``, ``n_outputs`` for a model or coord_matrix.\n        '
        if isinstance(input, Model):
            coords = _coord_matrix(input, position, input.n_outputs)
        else:
            coords = input
        return coords
    cleft = _n_inputs_outputs(left, 'left')
    cright = _n_inputs_outputs(right, 'right')
    try:
        result = np.dot(cleft, cright)
    except ValueError:
        raise ModelDefinitionError(f'Models cannot be combined with the "|" operator; left coord_matrix is {cright}, right coord_matrix is {cleft}')
    return result

def _separable(transform):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the separability of outputs.\n\n    Parameters\n    ----------\n    transform : `astropy.modeling.Model`\n        A transform (usually a compound model).\n\n    Returns :\n    is_separable : ndarray of dtype np.bool\n        An array of shape (transform.n_outputs,) of boolean type\n        Each element represents the separablity of the corresponding output.\n    '
    if (transform_matrix := transform._calculate_separability_matrix()) is not NotImplemented:
        return transform_matrix
    elif isinstance(transform, CompoundModel):
        sepleft = _separable(transform.left)
        sepright = _separable(transform.right)
        return _operators[transform.op](sepleft, sepright)
    elif isinstance(transform, Model):
        return _coord_matrix(transform, 'left', transform.n_outputs)
_operators = {'&': _cstack, '|': _cdot, '+': _arith_oper, '-': _arith_oper, '*': _arith_oper, '/': _arith_oper, '**': _arith_oper}