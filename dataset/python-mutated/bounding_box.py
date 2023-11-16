"""
This module is to contain an improved bounding box.
"""
from __future__ import annotations
import abc
import copy
import warnings
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from astropy.units import Quantity
from astropy.utils import isiterable
if TYPE_CHECKING:
    from typing import Any, Callable
    from typing_extensions import Self
    from astropy.units import UnitBase
__all__ = ['ModelBoundingBox', 'CompoundBoundingBox']

class _BaseInterval(NamedTuple):
    lower: float
    upper: float

class _Interval(_BaseInterval):
    """
    A single input's bounding box interval.

    Parameters
    ----------
    lower : float
        The lower bound of the interval

    upper : float
        The upper bound of the interval

    Methods
    -------
    validate :
        Constructs a valid interval

    outside :
        Determine which parts of an input array are outside the interval.

    domain :
        Constructs a discretization of the points inside the interval.
    """

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Interval(lower={self.lower}, upper={self.upper})'

    def copy(self):
        if False:
            return 10
        return copy.deepcopy(self)

    @staticmethod
    def _validate_shape(interval):
        if False:
            return 10
        'Validate the shape of an interval representation.'
        MESSAGE = 'An interval must be some sort of sequence of length 2'
        try:
            shape = np.shape(interval)
        except TypeError:
            try:
                if len(interval) == 1:
                    interval = interval[0]
                shape = np.shape([b.to_value() for b in interval])
            except (ValueError, TypeError, AttributeError):
                raise ValueError(MESSAGE)
        valid_shape = shape in ((2,), (1, 2), (2, 0))
        if not valid_shape:
            valid_shape = len(shape) > 0 and shape[0] == 2 and all((isinstance(b, np.ndarray) for b in interval))
        if not isiterable(interval) or not valid_shape:
            raise ValueError(MESSAGE)

    @classmethod
    def _validate_bounds(cls, lower, upper):
        if False:
            print('Hello World!')
        'Validate the bounds are reasonable and construct an interval from them.'
        if (np.asanyarray(lower) > np.asanyarray(upper)).all():
            warnings.warn(f'Invalid interval: upper bound {upper} is strictly less than lower bound {lower}.', RuntimeWarning)
        return cls(lower, upper)

    @classmethod
    def validate(cls, interval):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct and validate an interval.\n\n        Parameters\n        ----------\n        interval : iterable\n            A representation of the interval.\n\n        Returns\n        -------\n        A validated interval.\n        '
        cls._validate_shape(interval)
        if len(interval) == 1:
            interval = tuple(interval[0])
        else:
            interval = tuple(interval)
        return cls._validate_bounds(interval[0], interval[1])

    def outside(self, _input: np.ndarray):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        _input : np.ndarray\n            The evaluation input in the form of an array.\n\n        Returns\n        -------\n        Boolean array indicating which parts of _input are outside the interval:\n            True  -> position outside interval\n            False -> position inside  interval\n        '
        return np.logical_or(_input < self.lower, _input > self.upper)

    def domain(self, resolution):
        if False:
            return 10
        return np.arange(self.lower, self.upper + resolution, resolution)
_ignored_interval = _Interval.validate((-np.inf, np.inf))

def get_index(model, key) -> int:
    if False:
        while True:
            i = 10
    '\n    Get the input index corresponding to the given key.\n        Can pass in either:\n            the string name of the input or\n            the input index itself.\n    '
    if isinstance(key, str):
        if key in model.inputs:
            index = model.inputs.index(key)
        else:
            raise ValueError(f"'{key}' is not one of the inputs: {model.inputs}.")
    elif np.issubdtype(type(key), np.integer):
        if 0 <= key < len(model.inputs):
            index = key
        else:
            raise IndexError(f'Integer key: {key} must be non-negative and < {len(model.inputs)}.')
    else:
        raise ValueError(f'Key value: {key} must be string or integer.')
    return index

def get_name(model, index: int):
    if False:
        print('Hello World!')
    'Get the input name corresponding to the input index.'
    return model.inputs[index]

class _BoundingDomain(abc.ABC):
    """
    Base class for ModelBoundingBox and CompoundBoundingBox.
        This is where all the `~astropy.modeling.core.Model` evaluation
        code for evaluating with a bounding box is because it is common
        to both types of bounding box.

    Parameters
    ----------
    model : `~astropy.modeling.Model`
        The Model this bounding domain is for.

    prepare_inputs :
        Generates the necessary input information so that model can
        be evaluated only for input points entirely inside bounding_box.
        This needs to be implemented by a subclass. Note that most of
        the implementation is in ModelBoundingBox.

    prepare_outputs :
        Fills the output values in for any input points outside the
        bounding_box.

    evaluate :
        Performs a complete model evaluation while enforcing the bounds
        on the inputs and returns a complete output.
    """

    def __init__(self, model, ignored: list[int] | None=None, order: str='C'):
        if False:
            i = 10
            return i + 15
        self._model = model
        self._ignored = self._validate_ignored(ignored)
        self._order = self._get_order(order)

    @property
    def model(self):
        if False:
            print('Hello World!')
        return self._model

    @property
    def order(self) -> str:
        if False:
            return 10
        return self._order

    @property
    def ignored(self) -> list[int]:
        if False:
            for i in range(10):
                print('nop')
        return self._ignored

    def _get_order(self, order: str | None=None) -> str:
        if False:
            return 10
        '\n        Get if bounding_box is C/python ordered or Fortran/mathematically\n        ordered.\n        '
        if order is None:
            order = self._order
        if order not in ('C', 'F'):
            raise ValueError(f"order must be either 'C' (C/python order) or 'F' (Fortran/mathematical order), got: {order}.")
        return order

    def _get_index(self, key) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the input index corresponding to the given key.\n            Can pass in either:\n                the string name of the input or\n                the input index itself.\n        '
        return get_index(self._model, key)

    def _get_name(self, index: int):
        if False:
            return 10
        'Get the input name corresponding to the input index.'
        return get_name(self._model, index)

    @property
    def ignored_inputs(self) -> list[str]:
        if False:
            while True:
                i = 10
        return [self._get_name(index) for index in self._ignored]

    def _validate_ignored(self, ignored: list) -> list[int]:
        if False:
            while True:
                i = 10
        if ignored is None:
            return []
        else:
            return [self._get_index(key) for key in ignored]

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('This bounding box is fixed by the model and does not have adjustable parameters.')

    @abc.abstractmethod
    def fix_inputs(self, model, fixed_inputs: dict):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fix the bounding_box for a `fix_inputs` compound model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The new model for which this will be a bounding_box\n        fixed_inputs : dict\n            Dictionary of inputs which have been fixed by this bounding box.\n        '
        raise NotImplementedError('This should be implemented by a child class.')

    @abc.abstractmethod
    def prepare_inputs(self, input_shape, inputs) -> tuple[Any, Any, Any]:
        if False:
            i = 10
            return i + 15
        '\n        Get prepare the inputs with respect to the bounding box.\n\n        Parameters\n        ----------\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        inputs : list\n            List of all the model inputs\n\n        Returns\n        -------\n        valid_inputs : list\n            The inputs reduced to just those inputs which are all inside\n            their respective bounding box intervals\n        valid_index : array_like\n            array of all indices inside the bounding box\n        all_out: bool\n            if all of the inputs are outside the bounding_box\n        '
        raise NotImplementedError('This has not been implemented for BoundingDomain.')

    @staticmethod
    def _base_output(input_shape, fill_value):
        if False:
            print('Hello World!')
        '\n        Create a baseline output, assuming that the entire input is outside\n        the bounding box.\n\n        Parameters\n        ----------\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n\n        Returns\n        -------\n        An array of the correct shape containing all fill_value\n        '
        return np.zeros(input_shape) + fill_value

    def _all_out_output(self, input_shape, fill_value):
        if False:
            while True:
                i = 10
        '\n        Create output if all inputs are outside the domain.\n\n        Parameters\n        ----------\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n\n        Returns\n        -------\n        A full set of outputs for case that all inputs are outside domain.\n        '
        return ([self._base_output(input_shape, fill_value) for _ in range(self._model.n_outputs)], None)

    def _modify_output(self, valid_output, valid_index, input_shape, fill_value):
        if False:
            i = 10
            return i + 15
        '\n        For a single output fill in all the parts corresponding to inputs\n        outside the bounding box.\n\n        Parameters\n        ----------\n        valid_output : numpy array\n            The output from the model corresponding to inputs inside the\n            bounding box\n        valid_index : numpy array\n            array of all indices of inputs inside the bounding box\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n\n        Returns\n        -------\n        An output array with all the indices corresponding to inputs\n        outside the bounding box filled in by fill_value\n        '
        output = self._base_output(input_shape, fill_value)
        if not output.shape:
            output = np.array(valid_output)
        else:
            output[valid_index] = valid_output
        if np.isscalar(valid_output):
            output = output.item(0)
        return output

    def _prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        if False:
            while True:
                i = 10
        '\n        Fill in all the outputs of the model corresponding to inputs\n        outside the bounding_box.\n\n        Parameters\n        ----------\n        valid_outputs : list of numpy array\n            The list of outputs from the model corresponding to inputs\n            inside the bounding box\n        valid_index : numpy array\n            array of all indices of inputs inside the bounding box\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n\n        Returns\n        -------\n        List of filled in output arrays.\n        '
        outputs = []
        for valid_output in valid_outputs:
            outputs.append(self._modify_output(valid_output, valid_index, input_shape, fill_value))
        return outputs

    def prepare_outputs(self, valid_outputs, valid_index, input_shape, fill_value):
        if False:
            print('Hello World!')
        '\n        Fill in all the outputs of the model corresponding to inputs\n        outside the bounding_box, adjusting any single output model so that\n        its output becomes a list of containing that output.\n\n        Parameters\n        ----------\n        valid_outputs : list\n            The list of outputs from the model corresponding to inputs\n            inside the bounding box\n        valid_index : array_like\n            array of all indices of inputs inside the bounding box\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n        '
        if self._model.n_outputs == 1:
            valid_outputs = [valid_outputs]
        return self._prepare_outputs(valid_outputs, valid_index, input_shape, fill_value)

    @staticmethod
    def _get_valid_outputs_unit(valid_outputs, with_units: bool) -> UnitBase | None:
        if False:
            print('Hello World!')
        '\n        Get the unit for outputs if one is required.\n\n        Parameters\n        ----------\n        valid_outputs : list of numpy array\n            The list of outputs from the model corresponding to inputs\n            inside the bounding box\n        with_units : bool\n            whether or not a unit is required\n        '
        if with_units:
            return getattr(valid_outputs, 'unit', None)

    def _evaluate_model(self, evaluate: Callable, valid_inputs, valid_index, input_shape, fill_value, with_units: bool):
        if False:
            return 10
        '\n        Evaluate the model using the given evaluate routine.\n\n        Parameters\n        ----------\n        evaluate : Callable\n            callable which takes in the valid inputs to evaluate model\n        valid_inputs : list of numpy arrays\n            The inputs reduced to just those inputs which are all inside\n            their respective bounding box intervals\n        valid_index : numpy array\n            array of all indices inside the bounding box\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n        with_units : bool\n            whether or not a unit is required\n\n        Returns\n        -------\n        outputs :\n            list containing filled in output values\n        valid_outputs_unit :\n            the unit that will be attached to the outputs\n        '
        valid_outputs = evaluate(valid_inputs)
        valid_outputs_unit = self._get_valid_outputs_unit(valid_outputs, with_units)
        return (self.prepare_outputs(valid_outputs, valid_index, input_shape, fill_value), valid_outputs_unit)

    def _evaluate(self, evaluate: Callable, inputs, input_shape, fill_value, with_units: bool):
        if False:
            print('Hello World!')
        'Evaluate model with steps: prepare_inputs -> evaluate -> prepare_outputs.\n\n        Parameters\n        ----------\n        evaluate : Callable\n            callable which takes in the valid inputs to evaluate model\n        valid_inputs : list of numpy arrays\n            The inputs reduced to just those inputs which are all inside\n            their respective bounding box intervals\n        valid_index : numpy array\n            array of all indices inside the bounding box\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n        with_units : bool\n            whether or not a unit is required\n\n        Returns\n        -------\n        outputs :\n            list containing filled in output values\n        valid_outputs_unit :\n            the unit that will be attached to the outputs\n        '
        (valid_inputs, valid_index, all_out) = self.prepare_inputs(input_shape, inputs)
        if all_out:
            return self._all_out_output(input_shape, fill_value)
        else:
            return self._evaluate_model(evaluate, valid_inputs, valid_index, input_shape, fill_value, with_units)

    @staticmethod
    def _set_outputs_unit(outputs, valid_outputs_unit):
        if False:
            print('Hello World!')
        '\n        Set the units on the outputs\n            prepare_inputs -> evaluate -> prepare_outputs -> set output units.\n\n        Parameters\n        ----------\n        outputs :\n            list containing filled in output values\n        valid_outputs_unit :\n            the unit that will be attached to the outputs\n\n        Returns\n        -------\n        List containing filled in output values and units\n        '
        if valid_outputs_unit is not None:
            return Quantity(outputs, valid_outputs_unit, copy=False, subok=True)
        return outputs

    def evaluate(self, evaluate: Callable, inputs, fill_value):
        if False:
            print('Hello World!')
        '\n        Perform full model evaluation steps:\n            prepare_inputs -> evaluate -> prepare_outputs -> set output units.\n\n        Parameters\n        ----------\n        evaluate : callable\n            callable which takes in the valid inputs to evaluate model\n        valid_inputs : list\n            The inputs reduced to just those inputs which are all inside\n            their respective bounding box intervals\n        valid_index : array_like\n            array of all indices inside the bounding box\n        fill_value : float\n            The value which will be assigned to inputs which are outside\n            the bounding box\n        '
        input_shape = self._model.input_shape(inputs)
        (outputs, valid_outputs_unit) = self._evaluate(evaluate, inputs, input_shape, fill_value, self._model.bbox_with_units)
        return tuple(self._set_outputs_unit(outputs, valid_outputs_unit))

class ModelBoundingBox(_BoundingDomain):
    """
    A model's bounding box.

    Parameters
    ----------
    intervals : dict
        A dictionary containing all the intervals for each model input
            keys   -> input index
            values -> interval for that index

    model : `~astropy.modeling.Model`
        The Model this bounding_box is for.

    ignored : list
        A list containing all the inputs (index) which will not be
        checked for whether or not their elements are in/out of an interval.

    order : optional, str
        The ordering that is assumed for the tuple representation of this
        bounding_box. Options: 'C': C/Python order, e.g. z, y, x.
        (default), 'F': Fortran/mathematical notation order, e.g. x, y, z.
    """

    def __init__(self, intervals: dict[int, _Interval], model, ignored: list[int] | None=None, order: str='C'):
        if False:
            i = 10
            return i + 15
        super().__init__(model, ignored, order)
        self._intervals = {}
        if intervals != () and intervals != {}:
            self._validate(intervals, order=order)

    def copy(self, ignored=None):
        if False:
            i = 10
            return i + 15
        intervals = {index: interval.copy() for (index, interval) in self._intervals.items()}
        if ignored is None:
            ignored = self._ignored.copy()
        return ModelBoundingBox(intervals, self._model, ignored=ignored, order=self._order)

    @property
    def intervals(self) -> dict[int, _Interval]:
        if False:
            i = 10
            return i + 15
        'Return bounding_box labeled using input positions.'
        return self._intervals

    @property
    def named_intervals(self) -> dict[str, _Interval]:
        if False:
            print('Hello World!')
        'Return bounding_box labeled using input names.'
        return {self._get_name(index): bbox for (index, bbox) in self._intervals.items()}

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        parts = ['ModelBoundingBox(', '    intervals={']
        for (name, interval) in self.named_intervals.items():
            parts.append(f'        {name}: {interval}')
        parts.append('    }')
        if len(self._ignored) > 0:
            parts.append(f'    ignored={self.ignored_inputs}')
        parts.append(f'    model={self._model.__class__.__name__}(inputs={self._model.inputs})')
        parts.append(f"    order='{self._order}'")
        parts.append(')')
        return '\n'.join(parts)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._intervals)

    def __contains__(self, key):
        if False:
            print('Hello World!')
        try:
            return self._get_index(key) in self._intervals or self._ignored
        except (IndexError, ValueError):
            return False

    def has_interval(self, key):
        if False:
            return 10
        return self._get_index(key) in self._intervals

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        'Get bounding_box entries by either input name or input index.'
        index = self._get_index(key)
        if index in self._ignored:
            return _ignored_interval
        else:
            return self._intervals[self._get_index(key)]

    def bounding_box(self, order: str | None=None):
        if False:
            print('Hello World!')
        "\n        Return the old tuple of tuples representation of the bounding_box\n            order='C' corresponds to the old bounding_box ordering\n            order='F' corresponds to the gwcs bounding_box ordering.\n        "
        if len(self._intervals) == 1:
            return tuple(next(iter(self._intervals.values())))
        else:
            order = self._get_order(order)
            inputs = self._model.inputs
            if order == 'C':
                inputs = inputs[::-1]
            bbox = tuple((tuple(self[input_name]) for input_name in inputs))
            if len(bbox) == 1:
                bbox = bbox[0]
            return bbox

    def __eq__(self, value):
        if False:
            while True:
                i = 10
        'Note equality can be either with old representation or new one.'
        if isinstance(value, tuple):
            return self.bounding_box() == value
        elif isinstance(value, ModelBoundingBox):
            return self.intervals == value.intervals and self.ignored == value.ignored
        else:
            return False

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        'Validate and store interval under key (input index or input name).'
        index = self._get_index(key)
        if index in self._ignored:
            self._ignored.remove(index)
        self._intervals[index] = _Interval.validate(value)

    def __delitem__(self, key):
        if False:
            i = 10
            return i + 15
        'Delete stored interval.'
        index = self._get_index(key)
        if index in self._ignored:
            raise RuntimeError(f'Cannot delete ignored input: {key}!')
        del self._intervals[index]
        self._ignored.append(index)

    def _validate_dict(self, bounding_box: dict):
        if False:
            for i in range(10):
                print('nop')
        'Validate passing dictionary of intervals and setting them.'
        for (key, value) in bounding_box.items():
            self[key] = value

    @property
    def _available_input_index(self):
        if False:
            print('Hello World!')
        model_input_index = [self._get_index(_input) for _input in self._model.inputs]
        return [_input for _input in model_input_index if _input not in self._ignored]

    def _validate_sequence(self, bounding_box, order: str | None=None):
        if False:
            print('Hello World!')
        '\n        Validate passing tuple of tuples representation (or related) and setting them.\n        '
        order = self._get_order(order)
        if order == 'C':
            bounding_box = bounding_box[::-1]
        for (index, value) in enumerate(bounding_box):
            self[self._available_input_index[index]] = value

    @property
    def _n_inputs(self) -> int:
        if False:
            print('Hello World!')
        n_inputs = self._model.n_inputs - len(self._ignored)
        if n_inputs > 0:
            return n_inputs
        else:
            return 0

    def _validate_iterable(self, bounding_box, order: str | None=None):
        if False:
            for i in range(10):
                print('nop')
        'Validate and set any iterable representation.'
        if len(bounding_box) != self._n_inputs:
            raise ValueError(f'Found {len(bounding_box)} intervals, but must have exactly {self._n_inputs}.')
        if isinstance(bounding_box, dict):
            self._validate_dict(bounding_box)
        else:
            self._validate_sequence(bounding_box, order)

    def _validate(self, bounding_box, order: str | None=None):
        if False:
            print('Hello World!')
        'Validate and set any representation.'
        if self._n_inputs == 1 and (not isinstance(bounding_box, dict)):
            self[self._available_input_index[0]] = bounding_box
        else:
            self._validate_iterable(bounding_box, order)

    @classmethod
    def validate(cls, model, bounding_box, ignored: list | None=None, order: str='C', _preserve_ignore: bool=False, **kwargs) -> Self:
        if False:
            return 10
        "\n        Construct a valid bounding box for a model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The model for which this will be a bounding_box\n        bounding_box : dict, tuple\n            A possible representation of the bounding box\n        order : optional, str\n            The order that a tuple representation will be assumed to be\n                Default: 'C'\n        "
        if isinstance(bounding_box, ModelBoundingBox):
            order = bounding_box.order
            if _preserve_ignore:
                ignored = bounding_box.ignored
            bounding_box = bounding_box.named_intervals
        new = cls({}, model, ignored=ignored, order=order)
        new._validate(bounding_box)
        return new

    def fix_inputs(self, model, fixed_inputs: dict, _keep_ignored=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fix the bounding_box for a `fix_inputs` compound model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The new model for which this will be a bounding_box\n        fixed_inputs : dict\n            Dictionary of inputs which have been fixed by this bounding box.\n        keep_ignored : bool\n            Keep the ignored inputs of the bounding box (internal argument only)\n        '
        new = self.copy()
        for _input in fixed_inputs.keys():
            del new[_input]
        if _keep_ignored:
            ignored = new.ignored
        else:
            ignored = None
        return ModelBoundingBox.validate(model, new.named_intervals, ignored=ignored, order=new._order)

    @property
    def dimension(self):
        if False:
            return 10
        return len(self)

    def domain(self, resolution, order: str | None=None):
        if False:
            i = 10
            return i + 15
        inputs = self._model.inputs
        order = self._get_order(order)
        if order == 'C':
            inputs = inputs[::-1]
        return [self[input_name].domain(resolution) for input_name in inputs]

    def _outside(self, input_shape, inputs):
        if False:
            i = 10
            return i + 15
        '\n        Get all the input positions which are outside the bounding_box,\n        so that the corresponding outputs can be filled with the fill\n        value (default NaN).\n\n        Parameters\n        ----------\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        inputs : list\n            List of all the model inputs\n\n        Returns\n        -------\n        outside_index : bool-numpy array\n            True  -> position outside bounding_box\n            False -> position inside  bounding_box\n        all_out : bool\n            if all of the inputs are outside the bounding_box\n        '
        all_out = False
        outside_index = np.zeros(input_shape, dtype=bool)
        for (index, _input) in enumerate(inputs):
            _input = np.asanyarray(_input)
            outside = np.broadcast_to(self[index].outside(_input), input_shape)
            outside_index[outside] = True
            if outside_index.all():
                all_out = True
                break
        return (outside_index, all_out)

    def _valid_index(self, input_shape, inputs):
        if False:
            i = 10
            return i + 15
        '\n        Get the indices of all the inputs inside the bounding_box.\n\n        Parameters\n        ----------\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        inputs : list\n            List of all the model inputs\n\n        Returns\n        -------\n        valid_index : numpy array\n            array of all indices inside the bounding box\n        all_out : bool\n            if all of the inputs are outside the bounding_box\n        '
        (outside_index, all_out) = self._outside(input_shape, inputs)
        valid_index = np.atleast_1d(np.logical_not(outside_index)).nonzero()
        if len(valid_index[0]) == 0:
            all_out = True
        return (valid_index, all_out)

    def prepare_inputs(self, input_shape, inputs) -> tuple[Any, Any, Any]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get prepare the inputs with respect to the bounding box.\n\n        Parameters\n        ----------\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        inputs : list\n            List of all the model inputs\n\n        Returns\n        -------\n        valid_inputs : list\n            The inputs reduced to just those inputs which are all inside\n            their respective bounding box intervals\n        valid_index : array_like\n            array of all indices inside the bounding box\n        all_out: bool\n            if all of the inputs are outside the bounding_box\n        '
        (valid_index, all_out) = self._valid_index(input_shape, inputs)
        valid_inputs = []
        if not all_out:
            for _input in inputs:
                if input_shape:
                    valid_input = np.broadcast_to(np.atleast_1d(_input), input_shape)[valid_index]
                    if np.isscalar(_input):
                        valid_input = valid_input.item(0)
                    valid_inputs.append(valid_input)
                else:
                    valid_inputs.append(_input)
        return (tuple(valid_inputs), valid_index, all_out)

class _BaseSelectorArgument(NamedTuple):
    index: int
    ignore: bool

class _SelectorArgument(_BaseSelectorArgument):
    """
    Contains a single CompoundBoundingBox slicing input.

    Parameters
    ----------
    index : int
        The index of the input in the input list

    ignore : bool
        Whether or not this input will be ignored by the bounding box.

    Methods
    -------
    validate :
        Returns a valid SelectorArgument for a given model.

    get_selector :
        Returns the value of the input for use in finding the correct
        bounding_box.

    get_fixed_value :
        Gets the slicing value from a fix_inputs set of values.
    """

    def __new__(cls, index, ignore):
        if False:
            print('Hello World!')
        self = super().__new__(cls, index, ignore)
        return self

    @classmethod
    def validate(cls, model, argument, ignored: bool=True) -> Self:
        if False:
            i = 10
            return i + 15
        '\n        Construct a valid selector argument for a CompoundBoundingBox.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The model for which this will be an argument for.\n        argument : int or str\n            A representation of which evaluation input to use\n        ignored : optional, bool\n            Whether or not to ignore this argument in the ModelBoundingBox.\n\n        Returns\n        -------\n        Validated selector_argument\n        '
        return cls(get_index(model, argument), ignored)

    def get_selector(self, *inputs):
        if False:
            return 10
        '\n        Get the selector value corresponding to this argument.\n\n        Parameters\n        ----------\n        *inputs :\n            All the processed model evaluation inputs.\n        '
        _selector = inputs[self.index]
        if isiterable(_selector):
            if len(_selector) == 1:
                return _selector[0]
            else:
                return tuple(_selector)
        return _selector

    def name(self, model) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Get the name of the input described by this selector argument.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model this selector argument is for.\n        '
        return get_name(model, self.index)

    def pretty_repr(self, model):
        if False:
            return 10
        '\n        Get a pretty-print representation of this object.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model this selector argument is for.\n        '
        return f"Argument(name='{self.name(model)}', ignore={self.ignore})"

    def get_fixed_value(self, model, values: dict):
        if False:
            while True:
                i = 10
        '\n        Gets the value fixed input corresponding to this argument.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model this selector argument is for.\n\n        values : dict\n            Dictionary of fixed inputs.\n        '
        if self.index in values:
            return values[self.index]
        elif self.name(model) in values:
            return values[self.name(model)]
        else:
            raise RuntimeError(f'{self.pretty_repr(model)} was not found in {values}')

    def is_argument(self, model, argument) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine if passed argument is described by this selector argument.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model this selector argument is for.\n\n        argument : int or str\n            A representation of which evaluation input is being used\n        '
        return self.index == get_index(model, argument)

    def named_tuple(self, model):
        if False:
            i = 10
            return i + 15
        '\n        Get a tuple representation of this argument using the input\n        name from the model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model this selector argument is for.\n        '
        return (self.name(model), self.ignore)

class _SelectorArguments(tuple):
    """
    Contains the CompoundBoundingBox slicing description.

    Parameters
    ----------
    input_ :
        The SelectorArgument values

    Methods
    -------
    validate :
        Returns a valid SelectorArguments for its model.

    get_selector :
        Returns the selector a set of inputs corresponds to.

    is_selector :
        Determines if a selector is correctly formatted for this CompoundBoundingBox.

    get_fixed_value :
        Gets the selector from a fix_inputs set of values.
    """
    _kept_ignore = None

    def __new__(cls, input_: tuple[_SelectorArgument], kept_ignore: list | None=None) -> Self:
        if False:
            while True:
                i = 10
        self = super().__new__(cls, input_)
        if kept_ignore is None:
            self._kept_ignore = []
        else:
            self._kept_ignore = kept_ignore
        return self

    def pretty_repr(self, model):
        if False:
            i = 10
            return i + 15
        '\n        Get a pretty-print representation of this object.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n        '
        parts = ['SelectorArguments(']
        for argument in self:
            parts.append(f'    {argument.pretty_repr(model)}')
        parts.append(')')
        return '\n'.join(parts)

    @property
    def ignore(self):
        if False:
            print('Hello World!')
        'Get the list of ignored inputs.'
        ignore = [argument.index for argument in self if argument.ignore]
        ignore.extend(self._kept_ignore)
        return ignore

    @property
    def kept_ignore(self):
        if False:
            for i in range(10):
                print('nop')
        'The arguments to persist in ignoring.'
        return self._kept_ignore

    @classmethod
    def validate(cls, model, arguments, kept_ignore: list | None=None) -> Self:
        if False:
            i = 10
            return i + 15
        '\n        Construct a valid Selector description for a CompoundBoundingBox.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n\n        arguments :\n            The individual argument information\n\n        kept_ignore :\n            Arguments to persist as ignored\n        '
        inputs = []
        for argument in arguments:
            _input = _SelectorArgument.validate(model, *argument)
            if _input.index in [this.index for this in inputs]:
                raise ValueError(f"Input: '{get_name(model, _input.index)}' has been repeated.")
            inputs.append(_input)
        if len(inputs) == 0:
            raise ValueError('There must be at least one selector argument.')
        if isinstance(arguments, _SelectorArguments):
            if kept_ignore is None:
                kept_ignore = []
            kept_ignore.extend(arguments.kept_ignore)
        return cls(tuple(inputs), kept_ignore)

    def get_selector(self, *inputs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the selector corresponding to these inputs.\n\n        Parameters\n        ----------\n        *inputs :\n            All the processed model evaluation inputs.\n        '
        return tuple((argument.get_selector(*inputs) for argument in self))

    def is_selector(self, _selector):
        if False:
            i = 10
            return i + 15
        '\n        Determine if this is a reasonable selector.\n\n        Parameters\n        ----------\n        _selector : tuple\n            The selector to check\n        '
        return isinstance(_selector, tuple) and len(_selector) == len(self)

    def get_fixed_values(self, model, values: dict):
        if False:
            while True:
                i = 10
        '\n        Gets the value fixed input corresponding to this argument.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n\n        values : dict\n            Dictionary of fixed inputs.\n        '
        return tuple((argument.get_fixed_value(model, values) for argument in self))

    def is_argument(self, model, argument) -> bool:
        if False:
            print('Hello World!')
        '\n        Determine if passed argument is one of the selector arguments.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n\n        argument : int or str\n            A representation of which evaluation input is being used\n        '
        return any((selector_arg.is_argument(model, argument) for selector_arg in self))

    def selector_index(self, model, argument):
        if False:
            while True:
                i = 10
        '\n        Get the index of the argument passed in the selector tuples.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n\n        argument : int or str\n            A representation of which argument is being used\n        '
        for (index, selector_arg) in enumerate(self):
            if selector_arg.is_argument(model, argument):
                return index
        raise ValueError(f'{argument} does not correspond to any selector argument.')

    def reduce(self, model, argument):
        if False:
            i = 10
            return i + 15
        '\n        Reduce the selector arguments by the argument given.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n\n        argument : int or str\n            A representation of which argument is being used\n        '
        arguments = list(self)
        kept_ignore = [arguments.pop(self.selector_index(model, argument)).index]
        kept_ignore.extend(self._kept_ignore)
        return _SelectorArguments.validate(model, tuple(arguments), kept_ignore)

    def add_ignore(self, model, argument):
        if False:
            i = 10
            return i + 15
        '\n        Add argument to the kept_ignore list.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n\n        argument : int or str\n            A representation of which argument is being used\n        '
        if self.is_argument(model, argument):
            raise ValueError(f'{argument}: is a selector argument and cannot be ignored.')
        kept_ignore = [get_index(model, argument)]
        return _SelectorArguments.validate(model, self, kept_ignore)

    def named_tuple(self, model):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a tuple of selector argument tuples using input names.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The Model these selector arguments are for.\n        '
        return tuple((selector_arg.named_tuple(model) for selector_arg in self))

class CompoundBoundingBox(_BoundingDomain):
    """
    A model's compound bounding box.

    Parameters
    ----------
    bounding_boxes : dict
        A dictionary containing all the ModelBoundingBoxes that are possible
            keys   -> _selector (extracted from model inputs)
            values -> ModelBoundingBox

    model : `~astropy.modeling.Model`
        The Model this compound bounding_box is for.

    selector_args : _SelectorArguments
        A description of how to extract the selectors from model inputs.

    create_selector : optional
        A method which takes in the selector and the model to return a
        valid bounding corresponding to that selector. This can be used
        to construct new bounding_boxes for previously undefined selectors.
        These new boxes are then stored for future lookups.

    order : optional, str
        The ordering that is assumed for the tuple representation of the
        bounding_boxes.
    """

    def __init__(self, bounding_boxes: dict[Any, ModelBoundingBox], model, selector_args: _SelectorArguments, create_selector: Callable | None=None, ignored: list[int] | None=None, order: str='C'):
        if False:
            print('Hello World!')
        super().__init__(model, ignored, order)
        self._create_selector = create_selector
        self._selector_args = _SelectorArguments.validate(model, selector_args)
        self._bounding_boxes = {}
        self._validate(bounding_boxes)

    def copy(self):
        if False:
            return 10
        bounding_boxes = {selector: bbox.copy(self.selector_args.ignore) for (selector, bbox) in self._bounding_boxes.items()}
        return CompoundBoundingBox(bounding_boxes, self._model, selector_args=self._selector_args, create_selector=copy.deepcopy(self._create_selector), order=self._order)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        parts = ['CompoundBoundingBox(', '    bounding_boxes={']
        for (_selector, bbox) in self._bounding_boxes.items():
            bbox_repr = bbox.__repr__().split('\n')
            parts.append(f'        {_selector} = {bbox_repr.pop(0)}')
            for part in bbox_repr:
                parts.append(f'            {part}')
        parts.append('    }')
        selector_args_repr = self.selector_args.pretty_repr(self._model).split('\n')
        parts.append(f'    selector_args = {selector_args_repr.pop(0)}')
        for part in selector_args_repr:
            parts.append(f'        {part}')
        parts.append(')')
        return '\n'.join(parts)

    @property
    def bounding_boxes(self) -> dict[Any, ModelBoundingBox]:
        if False:
            for i in range(10):
                print('nop')
        return self._bounding_boxes

    @property
    def selector_args(self) -> _SelectorArguments:
        if False:
            while True:
                i = 10
        return self._selector_args

    @selector_args.setter
    def selector_args(self, value):
        if False:
            while True:
                i = 10
        self._selector_args = _SelectorArguments.validate(self._model, value)
        warnings.warn('Overriding selector_args may cause problems you should re-validate the compound bounding box before use!', RuntimeWarning)

    @property
    def named_selector_tuple(self) -> tuple:
        if False:
            print('Hello World!')
        return self._selector_args.named_tuple(self._model)

    @property
    def create_selector(self):
        if False:
            for i in range(10):
                print('nop')
        return self._create_selector

    @staticmethod
    def _get_selector_key(key):
        if False:
            print('Hello World!')
        if isiterable(key):
            return tuple(key)
        else:
            return (key,)

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        _selector = self._get_selector_key(key)
        if not self.selector_args.is_selector(_selector):
            raise ValueError(f'{_selector} is not a selector!')
        ignored = self.selector_args.ignore + self.ignored
        self._bounding_boxes[_selector] = ModelBoundingBox.validate(self._model, value, ignored, order=self._order)

    def _validate(self, bounding_boxes: dict):
        if False:
            i = 10
            return i + 15
        for (_selector, bounding_box) in bounding_boxes.items():
            self[_selector] = bounding_box

    def __eq__(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, CompoundBoundingBox):
            return self.bounding_boxes == value.bounding_boxes and self.selector_args == value.selector_args and (self.create_selector == value.create_selector)
        else:
            return False

    @classmethod
    def validate(cls, model, bounding_box: dict, selector_args=None, create_selector=None, ignored: list | None=None, order: str='C', _preserve_ignore: bool=False, **kwarg) -> Self:
        if False:
            i = 10
            return i + 15
        "\n        Construct a valid compound bounding box for a model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The model for which this will be a bounding_box\n        bounding_box : dict\n            Dictionary of possible bounding_box representations\n        selector_args : optional\n            Description of the selector arguments\n        create_selector : optional, callable\n            Method for generating new selectors\n        order : optional, str\n            The order that a tuple representation will be assumed to be\n                Default: 'C'\n        "
        if isinstance(bounding_box, CompoundBoundingBox):
            if selector_args is None:
                selector_args = bounding_box.selector_args
            if create_selector is None:
                create_selector = bounding_box.create_selector
            order = bounding_box.order
            if _preserve_ignore:
                ignored = bounding_box.ignored
            bounding_box = bounding_box.bounding_boxes
        if selector_args is None:
            raise ValueError('Selector arguments must be provided (can be passed as part of bounding_box argument)')
        return cls(bounding_box, model, selector_args, create_selector=create_selector, ignored=ignored, order=order)

    def __contains__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return key in self._bounding_boxes

    def _create_bounding_box(self, _selector):
        if False:
            return 10
        self[_selector] = self._create_selector(_selector, model=self._model)
        return self[_selector]

    def __getitem__(self, key):
        if False:
            return 10
        _selector = self._get_selector_key(key)
        if _selector in self:
            return self._bounding_boxes[_selector]
        elif self._create_selector is not None:
            return self._create_bounding_box(_selector)
        else:
            raise RuntimeError(f'No bounding box is defined for selector: {_selector}.')

    def _select_bounding_box(self, inputs) -> ModelBoundingBox:
        if False:
            for i in range(10):
                print('nop')
        _selector = self.selector_args.get_selector(*inputs)
        return self[_selector]

    def prepare_inputs(self, input_shape, inputs) -> tuple[Any, Any, Any]:
        if False:
            print('Hello World!')
        '\n        Get prepare the inputs with respect to the bounding box.\n\n        Parameters\n        ----------\n        input_shape : tuple\n            The shape that all inputs have be reshaped/broadcasted into\n        inputs : list\n            List of all the model inputs\n\n        Returns\n        -------\n        valid_inputs : list\n            The inputs reduced to just those inputs which are all inside\n            their respective bounding box intervals\n        valid_index : array_like\n            array of all indices inside the bounding box\n        all_out: bool\n            if all of the inputs are outside the bounding_box\n        '
        bounding_box = self._select_bounding_box(inputs)
        return bounding_box.prepare_inputs(input_shape, inputs)

    def _matching_bounding_boxes(self, argument, value) -> dict[Any, ModelBoundingBox]:
        if False:
            while True:
                i = 10
        selector_index = self.selector_args.selector_index(self._model, argument)
        matching = {}
        for (selector_key, bbox) in self._bounding_boxes.items():
            if selector_key[selector_index] == value:
                new_selector_key = list(selector_key)
                new_selector_key.pop(selector_index)
                if bbox.has_interval(argument):
                    new_bbox = bbox.fix_inputs(self._model, {argument: value}, _keep_ignored=True)
                else:
                    new_bbox = bbox.copy()
                matching[tuple(new_selector_key)] = new_bbox
        if len(matching) == 0:
            raise ValueError(f'Attempting to fix input {argument}, but there are no bounding boxes for argument value {value}.')
        return matching

    def _fix_input_selector_arg(self, argument, value):
        if False:
            for i in range(10):
                print('nop')
        matching_bounding_boxes = self._matching_bounding_boxes(argument, value)
        if len(self.selector_args) == 1:
            return matching_bounding_boxes[()]
        else:
            return CompoundBoundingBox(matching_bounding_boxes, self._model, self.selector_args.reduce(self._model, argument))

    def _fix_input_bbox_arg(self, argument, value):
        if False:
            for i in range(10):
                print('nop')
        bounding_boxes = {}
        for (selector_key, bbox) in self._bounding_boxes.items():
            bounding_boxes[selector_key] = bbox.fix_inputs(self._model, {argument: value}, _keep_ignored=True)
        return CompoundBoundingBox(bounding_boxes, self._model, self.selector_args.add_ignore(self._model, argument))

    def fix_inputs(self, model, fixed_inputs: dict):
        if False:
            i = 10
            return i + 15
        '\n        Fix the bounding_box for a `fix_inputs` compound model.\n\n        Parameters\n        ----------\n        model : `~astropy.modeling.Model`\n            The new model for which this will be a bounding_box\n        fixed_inputs : dict\n            Dictionary of inputs which have been fixed by this bounding box.\n        '
        fixed_input_keys = list(fixed_inputs.keys())
        argument = fixed_input_keys.pop()
        value = fixed_inputs[argument]
        if self.selector_args.is_argument(self._model, argument):
            bbox = self._fix_input_selector_arg(argument, value)
        else:
            bbox = self._fix_input_bbox_arg(argument, value)
        if len(fixed_input_keys) > 0:
            new_fixed_inputs = fixed_inputs.copy()
            del new_fixed_inputs[argument]
            bbox = bbox.fix_inputs(model, new_fixed_inputs)
        if isinstance(bbox, CompoundBoundingBox):
            selector_args = bbox.named_selector_tuple
            bbox_dict = bbox
        elif isinstance(bbox, ModelBoundingBox):
            selector_args = None
            bbox_dict = bbox.named_intervals
        return bbox.__class__.validate(model, bbox_dict, order=bbox.order, selector_args=selector_args)