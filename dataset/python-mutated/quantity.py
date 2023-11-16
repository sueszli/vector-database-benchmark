"""
This module defines the `Quantity` object, which represents a number with some
associated units. `Quantity` objects support operations like ordinary numbers,
but will deal with unit conversions internally.
"""
import numbers
import operator
import re
import warnings
from fractions import Fraction
import numpy as np
from astropy import config as _config
from astropy.utils.compat.numpycompat import NUMPY_LT_2_0
from astropy.utils.data_info import ParentDtypeInfo
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyWarning
from astropy.utils.misc import isiterable
from .core import Unit, UnitBase, UnitConversionError, UnitsError, UnitTypeError, dimensionless_unscaled, get_current_unit_registry
from .format import Base, Latex
from .quantity_helper import can_have_arbitrary_unit, check_output, converters_and_unit
from .quantity_helper.function_helpers import DISPATCHED_FUNCTIONS, FUNCTION_HELPERS, SUBCLASS_SAFE_FUNCTIONS, UNSUPPORTED_FUNCTIONS
from .structured import StructuredUnit, _structured_unit_like_dtype
from .utils import is_effectively_unity
__all__ = ['Quantity', 'SpecificTypeQuantity', 'QuantityInfoBase', 'QuantityInfo', 'allclose', 'isclose']
__doctest_skip__ = ['Quantity.*']
_UNIT_NOT_INITIALISED = '(Unit not initialised)'
_UFUNCS_FILTER_WARNINGS = {np.arcsin, np.arccos, np.arccosh, np.arctanh}

class Conf(_config.ConfigNamespace):
    """
    Configuration parameters for Quantity.
    """
    latex_array_threshold = _config.ConfigItem(100, 'The maximum size an array Quantity can be before its LaTeX representation for IPython gets "summarized" (meaning only the first and last few elements are shown with "..." between). Setting this to a negative number means that the value will instead be whatever numpy gets from get_printoptions.')
conf = Conf()

class QuantityIterator:
    """
    Flat iterator object to iterate over Quantities.

    A `QuantityIterator` iterator is returned by ``q.flat`` for any Quantity
    ``q``.  It allows iterating over the array as if it were a 1-D array,
    either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    See Also
    --------
    Quantity.flatten : Returns a flattened copy of an array.

    Notes
    -----
    `QuantityIterator` is inspired by `~numpy.ma.core.MaskedIterator`.  It
    is not exported by the `~astropy.units` module.  Instead of
    instantiating a `QuantityIterator` directly, use `Quantity.flat`.
    """

    def __init__(self, q):
        if False:
            return 10
        self._quantity = q
        self._dataiter = q.view(np.ndarray).flat

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __getitem__(self, indx):
        if False:
            i = 10
            return i + 15
        out = self._dataiter.__getitem__(indx)
        if isinstance(out, type(self._quantity)):
            return out
        else:
            return self._quantity._new_view(out)

    def __setitem__(self, index, value):
        if False:
            return 10
        self._dataiter[index] = self._quantity._to_own_unit(value)

    def __next__(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the next value, or raise StopIteration.\n        '
        out = next(self._dataiter)
        return self._quantity._new_view(out)
    next = __next__

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._dataiter)

    @property
    def base(self):
        if False:
            return 10
        'A reference to the array that is iterated over.'
        return self._quantity

    @property
    def coords(self):
        if False:
            print('Hello World!')
        'An N-dimensional tuple of current coordinates.'
        return self._dataiter.coords

    @property
    def index(self):
        if False:
            i = 10
            return i + 15
        'Current flat index into the array.'
        return self._dataiter.index

    def copy(self):
        if False:
            print('Hello World!')
        'Get a copy of the iterator as a 1-D array.'
        return self._quantity.flatten()

class QuantityInfoBase(ParentDtypeInfo):
    attrs_from_parent = {'dtype', 'unit'}
    _supports_indexing = True

    @staticmethod
    def default_format(val):
        if False:
            print('Hello World!')
        return f'{val.value}'

    @staticmethod
    def possible_string_format_functions(format_):
        if False:
            for i in range(10):
                print('nop')
        'Iterate through possible string-derived format functions.\n\n        A string can either be a format specifier for the format built-in,\n        a new-style format string, or an old-style format string.\n\n        This method is overridden in order to suppress printing the unit\n        in each row since it is already at the top in the column header.\n        '
        yield (lambda format_, val: format(val.value, format_))
        yield (lambda format_, val: format_.format(val.value))
        yield (lambda format_, val: format_ % val.value)

class QuantityInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    _represent_as_dict_attrs = ('value', 'unit')
    _construct_from_dict_args = ['value']
    _represent_as_dict_primary_data = 'value'

    def new_like(self, cols, length, metadata_conflicts='warn', name=None):
        if False:
            i = 10
            return i + 15
        "\n        Return a new Quantity instance which is consistent with the\n        input ``cols`` and has ``length`` rows.\n\n        This is intended for creating an empty column object whose elements can\n        be set in-place for table operations like join or vstack.\n\n        Parameters\n        ----------\n        cols : list\n            List of input columns\n        length : int\n            Length of the output column object\n        metadata_conflicts : str ('warn'|'error'|'silent')\n            How to handle metadata conflicts\n        name : str\n            Output column name\n\n        Returns\n        -------\n        col : `~astropy.units.Quantity` (or subclass)\n            Empty instance of this class consistent with ``cols``\n\n        "
        attrs = self.merge_cols_attributes(cols, metadata_conflicts, name, ('meta', 'format', 'description'))
        shape = (length,) + attrs.pop('shape')
        dtype = attrs.pop('dtype')
        data = np.zeros(shape=shape, dtype=dtype)
        map = {key: data if key == 'value' else getattr(cols[-1], key) for key in self._represent_as_dict_attrs}
        map['copy'] = False
        out = self._construct_from_dict(map)
        for (attr, value) in attrs.items():
            setattr(out.info, attr, value)
        return out

    def get_sortable_arrays(self):
        if False:
            while True:
                i = 10
        '\n        Return a list of arrays which can be lexically sorted to represent\n        the order of the parent column.\n\n        For Quantity this is just the quantity itself.\n\n\n        Returns\n        -------\n        arrays : list of ndarray\n        '
        return [self._parent]

class Quantity(np.ndarray):
    """A `~astropy.units.Quantity` represents a number with some associated unit.

    See also: https://docs.astropy.org/en/stable/units/quantity.html

    Parameters
    ----------
    value : number, `~numpy.ndarray`, `~astropy.units.Quantity` (sequence), or str
        The numerical value of this quantity in the units given by unit.  If a
        `Quantity` or sequence of them (or any other valid object with a
        ``unit`` attribute), creates a new `Quantity` object, converting to
        `unit` units as needed.  If a string, it is converted to a number or
        `Quantity`, depending on whether a unit is present.

    unit : unit-like
        An object that represents the unit associated with the input value.
        Must be an `~astropy.units.UnitBase` object or a string parseable by
        the :mod:`~astropy.units` package.

    dtype : ~numpy.dtype, optional
        The dtype of the resulting Numpy array or scalar that will
        hold the value.  If not provided, it is determined from the input,
        except that any integer and (non-Quantity) object inputs are converted
        to float by default.
        If `None`, the normal `numpy.dtype` introspection is used, e.g.
        preventing upcasting of integers.

    copy : bool, optional
        If `True` (default), then the value is copied.  Otherwise, a copy will
        only be made if ``__array__`` returns a copy, if value is a nested
        sequence, or if a copy is needed to satisfy an explicitly given
        ``dtype``.  (The `False` option is intended mostly for internal use,
        to speed up initialization where a copy is known to have been made.
        Use with care.)

    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  As in `~numpy.array`.  This parameter
        is ignored if the input is a `Quantity` and ``copy=False``.

    subok : bool, optional
        If `False` (default), the returned array will be forced to be a
        `Quantity`.  Otherwise, `Quantity` subclasses will be passed through,
        or a subclass appropriate for the unit will be used (such as
        `~astropy.units.Dex` for ``u.dex(u.AA)``).

    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array
        should have.  Ones will be prepended to the shape as needed to meet
        this requirement.  This parameter is ignored if the input is a
        `Quantity` and ``copy=False``.

    Raises
    ------
    TypeError
        If the value provided is not a Python numeric type.
    TypeError
        If the unit provided is not either a :class:`~astropy.units.Unit`
        object or a parseable string unit.

    Notes
    -----
    Quantities can also be created by multiplying a number or array with a
    :class:`~astropy.units.Unit`. See https://docs.astropy.org/en/latest/units/

    Unless the ``dtype`` argument is explicitly specified, integer
    or (non-Quantity) object inputs are converted to `float` by default.
    """
    _equivalencies = []
    _default_unit = dimensionless_unscaled
    _unit = None
    __array_priority__ = 10000

    def __class_getitem__(cls, unit_shape_dtype):
        if False:
            while True:
                i = 10
        'Quantity Type Hints.\n\n        Unit-aware type hints are ``Annotated`` objects that encode the class,\n        the unit, and possibly shape and dtype information, depending on the\n        python and :mod:`numpy` versions.\n\n        Schematically, ``Annotated[cls[shape, dtype], unit]``\n\n        As a classmethod, the type is the class, ie ``Quantity``\n        produces an ``Annotated[Quantity, ...]`` while a subclass\n        like :class:`~astropy.coordinates.Angle` returns\n        ``Annotated[Angle, ...]``.\n\n        Parameters\n        ----------\n        unit_shape_dtype : :class:`~astropy.units.UnitBase`, str, `~astropy.units.PhysicalType`, or tuple\n            Unit specification, can be the physical type (ie str or class).\n            If tuple, then the first element is the unit specification\n            and all other elements are for `numpy.ndarray` type annotations.\n            Whether they are included depends on the python and :mod:`numpy`\n            versions.\n\n        Returns\n        -------\n        `typing.Annotated`, `astropy.units.Unit`, or `astropy.units.PhysicalType`\n            Return type in this preference order:\n            * `typing.Annotated`\n            * `astropy.units.Unit` or `astropy.units.PhysicalType`\n\n        Raises\n        ------\n        TypeError\n            If the unit/physical_type annotation is not Unit-like or\n            PhysicalType-like.\n\n        Examples\n        --------\n        Create a unit-aware Quantity type annotation\n\n            >>> Quantity[Unit("s")]\n            Annotated[Quantity, Unit("s")]\n\n        See Also\n        --------\n        `~astropy.units.quantity_input`\n            Use annotations for unit checks on function arguments and results.\n\n        Notes\n        -----\n        With Python 3.9+ or :mod:`typing_extensions`, |Quantity| types are also\n        static-type compatible.\n        '
        from typing import Annotated
        if isinstance(unit_shape_dtype, tuple):
            target = unit_shape_dtype[0]
            shape_dtype = unit_shape_dtype[1:]
        else:
            target = unit_shape_dtype
            shape_dtype = ()
        try:
            unit = Unit(target)
        except (TypeError, ValueError):
            from astropy.units.physical import get_physical_type
            try:
                unit = get_physical_type(target)
            except (TypeError, ValueError, KeyError):
                raise TypeError('unit annotation is not a Unit or PhysicalType') from None
        return Annotated[cls, unit]

    def __new__(cls, value, unit=None, dtype=np.inexact, copy=True, order=None, subok=False, ndmin=0):
        if False:
            return 10
        if unit is not None:
            unit = Unit(unit)
        float_default = dtype is np.inexact
        if float_default:
            dtype = None
        if isinstance(value, Quantity):
            if unit is not None and unit is not value.unit:
                value = value.to(unit)
                copy = False
            if type(value) is not cls and (not (subok and isinstance(value, cls))):
                value = value.view(cls)
            if float_default and value.dtype.kind in 'iu':
                dtype = float
            return np.array(value, dtype=dtype, copy=copy, order=order, subok=True, ndmin=ndmin)
        value_unit = None
        if not isinstance(value, np.ndarray):
            if isinstance(value, str):
                pattern = '\\s*[+-]?((\\d+\\.?\\d*)|(\\.\\d+)|([nN][aA][nN])|([iI][nN][fF]([iI][nN][iI][tT][yY]){0,1}))([eE][+-]?\\d+)?[.+-]?'
                v = re.match(pattern, value)
                unit_string = None
                try:
                    value = float(v.group())
                except Exception:
                    raise TypeError(f'Cannot parse "{value}" as a {cls.__name__}. It does not start with a number.')
                unit_string = v.string[v.end():].strip()
                if unit_string:
                    value_unit = Unit(unit_string)
                    if unit is None:
                        unit = value_unit
            elif isiterable(value) and len(value) > 0:
                if all((isinstance(v, Quantity) for v in value)):
                    if unit is None:
                        unit = value[0].unit
                    value = [q.to_value(unit) for q in value]
                    value_unit = unit
                elif dtype is None and (not hasattr(value, 'dtype')) and isinstance(unit, StructuredUnit):
                    dtype = unit._recursively_get_dtype(value)
        using_default_unit = False
        if value_unit is None:
            value_unit = getattr(value, 'unit', None)
            if value_unit is None:
                if unit is None:
                    using_default_unit = True
                    unit = cls._default_unit
                value_unit = unit
            else:
                try:
                    value_unit = Unit(value_unit)
                except Exception as exc:
                    raise TypeError(f'The unit attribute {value.unit!r} of the input could not be parsed as an astropy Unit.') from exc
                if unit is None:
                    unit = value_unit
                elif unit is not value_unit:
                    copy = False
        value = np.array(value, dtype=dtype, copy=copy, order=order, subok=True, ndmin=ndmin)
        if using_default_unit and value.dtype.names is not None:
            unit = value_unit = _structured_unit_like_dtype(value_unit, value.dtype)
        if value.dtype.kind in 'OSU' and (not (value.dtype.kind == 'O' and isinstance(value.item(0), numbers.Number))):
            raise TypeError('The value must be a valid Python or Numpy numeric type.')
        if float_default and value.dtype.kind in 'iuO':
            value = value.astype(float)
        if subok:
            qcls = getattr(unit, '_quantity_class', cls)
            if issubclass(qcls, cls):
                cls = qcls
        value = value.view(cls)
        value._set_unit(value_unit)
        if unit is value_unit:
            return value
        else:
            return value.to(unit)

    def __array_finalize__(self, obj):
        if False:
            return 10
        super_array_finalize = super().__array_finalize__
        if super_array_finalize is not None:
            super_array_finalize(obj)
        if obj is None or obj.__class__ is np.ndarray:
            return
        if self._unit is None:
            unit = getattr(obj, '_unit', None)
            if unit is not None:
                self._set_unit(unit)
            if 'info' in obj.__dict__:
                self.info = obj.info

    def __array_wrap__(self, obj, context=None):
        if False:
            print('Hello World!')
        if context is None:
            return self._new_view(obj)
        raise NotImplementedError('__array_wrap__ should not be used with a context any more since all use should go through array_function. Please raise an issue on https://github.com/astropy/astropy')

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Wrap numpy ufuncs, taking care of units.\n\n        Parameters\n        ----------\n        function : callable\n            ufunc to wrap.\n        method : str\n            Ufunc method: ``__call__``, ``at``, ``reduce``, etc.\n        inputs : tuple\n            Input arrays.\n        kwargs : keyword arguments\n            As passed on, with ``out`` containing possible quantity output.\n\n        Returns\n        -------\n        result : `~astropy.units.Quantity` or `NotImplemented`\n            Results of the ufunc, with the unit set properly.\n        '
        try:
            (converters, unit) = converters_and_unit(function, method, *inputs)
            out = kwargs.get('out', None)
            if out is not None:
                if function.nout == 1:
                    out = out[0]
                out_array = check_output(out, unit, inputs, function=function)
                kwargs['out'] = (out_array,) if function.nout == 1 else out_array
            if method == 'reduce' and 'initial' in kwargs and (unit is not None):
                kwargs['initial'] = self._to_own_unit(kwargs['initial'], check_precision=False, unit=unit)
            arrays = []
            for (input_, converter) in zip(inputs, converters):
                input_ = getattr(input_, 'value', input_)
                arrays.append(converter(input_) if converter else input_)
            result = super().__array_ufunc__(function, method, *arrays, **kwargs)
            if unit is None or result is None or result is NotImplemented:
                return result
            return self._result_as_quantity(result, unit, out)
        except (TypeError, ValueError, AttributeError) as e:
            out_normalized = kwargs.get('out', tuple())
            inputs_and_outputs = inputs + out_normalized
            ignored_ufunc = (None, np.ndarray.__array_ufunc__, type(self).__array_ufunc__)
            if not all((getattr(type(io), '__array_ufunc__', None) in ignored_ufunc for io in inputs_and_outputs)):
                return NotImplemented
            else:
                raise e

    def _result_as_quantity(self, result, unit, out):
        if False:
            while True:
                i = 10
        'Turn result into a quantity with the given unit.\n\n        If no output is given, it will take a view of the array as a quantity,\n        and set the unit.  If output is given, those should be quantity views\n        of the result arrays, and the function will just set the unit.\n\n        Parameters\n        ----------\n        result : ndarray or tuple thereof\n            Array(s) which need to be turned into quantity.\n        unit : `~astropy.units.Unit`\n            Unit for the quantities to be returned (or `None` if the result\n            should not be a quantity).  Should be tuple if result is a tuple.\n        out : `~astropy.units.Quantity` or None\n            Possible output quantity. Should be `None` or a tuple if result\n            is a tuple.\n\n        Returns\n        -------\n        out : `~astropy.units.Quantity`\n           With units set.\n        '
        if isinstance(result, (tuple, list)):
            if out is None:
                out = (None,) * len(result)
            result_cls = getattr(result, '_make', result.__class__)
            return result_cls((self._result_as_quantity(result_, unit_, out_) for (result_, unit_, out_) in zip(result, unit, out)))
        if out is None:
            return result if unit is None else self._new_view(result, unit, propagate_info=False)
        elif isinstance(out, Quantity):
            out._set_unit(unit)
        return out

    def __quantity_subclass__(self, unit):
        if False:
            return 10
        '\n        Overridden by subclasses to change what kind of view is\n        created based on the output unit of an operation.\n\n        Parameters\n        ----------\n        unit : UnitBase\n            The unit for which the appropriate class should be returned\n\n        Returns\n        -------\n        tuple :\n            - `~astropy.units.Quantity` subclass\n            - bool: True if subclasses of the given class are ok\n        '
        return (Quantity, True)

    def _new_view(self, obj=None, unit=None, propagate_info=True):
        if False:
            print('Hello World!')
        'Create a Quantity view of some array-like input, and set the unit.\n\n        By default, return a view of ``obj`` of the same class as ``self`` and\n        with the same unit.  Subclasses can override the type of class for a\n        given unit using ``__quantity_subclass__``, and can ensure properties\n        other than the unit are copied using ``__array_finalize__``.\n\n        If the given unit defines a ``_quantity_class`` of which ``self``\n        is not an instance, a view using this class is taken.\n\n        Parameters\n        ----------\n        obj : ndarray or scalar, optional\n            The array to create a view of.  If obj is a numpy or python scalar,\n            it will be converted to an array scalar.  By default, ``self``\n            is converted.\n\n        unit : unit-like, optional\n            The unit of the resulting object.  It is used to select a\n            subclass, and explicitly assigned to the view if given.\n            If not given, the subclass and unit will be that of ``self``.\n\n        propagate_info : bool, optional\n            Whether to transfer ``info`` if present.  Default: `True`, as\n            appropriate for, e.g., unit conversions or slicing, where the\n            nature of the object does not change.\n\n        Returns\n        -------\n        view : `~astropy.units.Quantity` subclass\n\n        '
        if unit is None:
            unit = self.unit
            quantity_subclass = self.__class__
        elif unit is self.unit and self.__class__ is Quantity:
            quantity_subclass = Quantity
        else:
            unit = Unit(unit)
            quantity_subclass = getattr(unit, '_quantity_class', Quantity)
            if isinstance(self, quantity_subclass):
                (quantity_subclass, subok) = self.__quantity_subclass__(unit)
                if subok:
                    quantity_subclass = self.__class__
        if obj is None:
            obj = self.view(np.ndarray)
        else:
            obj = np.array(obj, copy=False, subok=True)
        view = obj.view(quantity_subclass)
        view._set_unit(unit)
        view.__array_finalize__(self)
        if propagate_info and 'info' in self.__dict__:
            view.info = self.info
        return view

    def _set_unit(self, unit):
        if False:
            i = 10
            return i + 15
        'Set the unit.\n\n        This is used anywhere the unit is set or modified, i.e., in the\n        initializer, in ``__imul__`` and ``__itruediv__`` for in-place\n        multiplication and division by another unit, as well as in\n        ``__array_finalize__`` for wrapping up views.  For Quantity, it just\n        sets the unit, but subclasses can override it to check that, e.g.,\n        a unit is consistent.\n        '
        if not isinstance(unit, UnitBase):
            if isinstance(self._unit, StructuredUnit) or isinstance(unit, StructuredUnit):
                unit = StructuredUnit(unit, self.dtype)
            else:
                unit = Unit(str(unit), parse_strict='silent')
                if not isinstance(unit, (UnitBase, StructuredUnit)):
                    raise UnitTypeError(f'{self.__class__.__name__} instances require normal units, not {unit.__class__} instances.')
        self._unit = unit

    def __deepcopy__(self, memo):
        if False:
            while True:
                i = 10
        return self.copy()

    def __reduce__(self):
        if False:
            return 10
        object_state = list(super().__reduce__())
        object_state[2] = (object_state[2], self.__dict__)
        return tuple(object_state)

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        (nd_state, own_state) = state
        super().__setstate__(nd_state)
        self.__dict__.update(own_state)
    info = QuantityInfo()

    def _to_value(self, unit, equivalencies=[]):
        if False:
            i = 10
            return i + 15
        'Helper method for to and to_value.'
        if equivalencies == []:
            equivalencies = self._equivalencies
        if not self.dtype.names or isinstance(self.unit, StructuredUnit):
            return self.unit.to(unit, self.view(np.ndarray), equivalencies=equivalencies)
        else:
            result = np.empty_like(self.view(np.ndarray))
            for name in self.dtype.names:
                result[name] = self[name]._to_value(unit, equivalencies)
            return result

    def to(self, unit, equivalencies=[], copy=True):
        if False:
            i = 10
            return i + 15
        '\n        Return a new `~astropy.units.Quantity` object with the specified unit.\n\n        Parameters\n        ----------\n        unit : unit-like\n            An object that represents the unit to convert to. Must be\n            an `~astropy.units.UnitBase` object or a string parseable\n            by the `~astropy.units` package.\n\n        equivalencies : list of tuple\n            A list of equivalence pairs to try if the units are not\n            directly convertible.  See :ref:`astropy:unit_equivalencies`.\n            If not provided or ``[]``, class default equivalencies will be used\n            (none for `~astropy.units.Quantity`, but may be set for subclasses)\n            If `None`, no equivalencies will be applied at all, not even any\n            set globally or within a context.\n\n        copy : bool, optional\n            If `True` (default), then the value is copied.  Otherwise, a copy\n            will only be made if necessary.\n\n        See Also\n        --------\n        to_value : get the numerical value in a given unit.\n        '
        unit = Unit(unit)
        if copy:
            value = self._to_value(unit, equivalencies)
        else:
            value = self.to_value(unit, equivalencies)
        return self._new_view(value, unit)

    def to_value(self, unit=None, equivalencies=[]):
        if False:
            return 10
        '\n        The numerical value, possibly in a different unit.\n\n        Parameters\n        ----------\n        unit : unit-like, optional\n            The unit in which the value should be given. If not given or `None`,\n            use the current unit.\n\n        equivalencies : list of tuple, optional\n            A list of equivalence pairs to try if the units are not directly\n            convertible (see :ref:`astropy:unit_equivalencies`). If not provided\n            or ``[]``, class default equivalencies will be used (none for\n            `~astropy.units.Quantity`, but may be set for subclasses).\n            If `None`, no equivalencies will be applied at all, not even any\n            set globally or within a context.\n\n        Returns\n        -------\n        value : ndarray or scalar\n            The value in the units specified. For arrays, this will be a view\n            of the data if no unit conversion was necessary.\n\n        See Also\n        --------\n        to : Get a new instance in a different unit.\n        '
        if unit is None or unit is self.unit:
            value = self.view(np.ndarray)
        elif not self.dtype.names:
            unit = Unit(unit)
            try:
                scale = self.unit._to(unit)
            except Exception:
                value = self._to_value(unit, equivalencies)
            else:
                value = self.view(np.ndarray)
                if not is_effectively_unity(scale):
                    value = value * scale
        else:
            value = self._to_value(unit, equivalencies)
        return value if value.shape else value[()]
    value = property(to_value, doc='The numerical value of this instance.\n\n    See also\n    --------\n    to_value : Get the numerical value in a given unit.\n    ')

    @property
    def unit(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A `~astropy.units.UnitBase` object representing the unit of this\n        quantity.\n        '
        return self._unit

    @property
    def equivalencies(self):
        if False:
            i = 10
            return i + 15
        '\n        A list of equivalencies that will be applied by default during\n        unit conversions.\n        '
        return self._equivalencies

    def _recursively_apply(self, func):
        if False:
            i = 10
            return i + 15
        'Apply function recursively to every field.\n\n        Returns a copy with the result.\n        '
        result = np.empty_like(self)
        result_value = result.view(np.ndarray)
        result_unit = ()
        for name in self.dtype.names:
            part = func(self[name])
            result_value[name] = part.value
            result_unit += (part.unit,)
        result._set_unit(result_unit)
        return result

    @property
    def si(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a copy of the current `Quantity` instance with SI units. The\n        value of the resulting object will be scaled.\n        '
        if self.dtype.names:
            return self._recursively_apply(operator.attrgetter('si'))
        si_unit = self.unit.si
        return self._new_view(self.value * si_unit.scale, si_unit / si_unit.scale)

    @property
    def cgs(self):
        if False:
            return 10
        '\n        Returns a copy of the current `Quantity` instance with CGS units. The\n        value of the resulting object will be scaled.\n        '
        if self.dtype.names:
            return self._recursively_apply(operator.attrgetter('cgs'))
        cgs_unit = self.unit.cgs
        return self._new_view(self.value * cgs_unit.scale, cgs_unit / cgs_unit.scale)

    @property
    def isscalar(self):
        if False:
            return 10
        '\n        True if the `value` of this quantity is a scalar, or False if it\n        is an array-like object.\n\n        .. note::\n            This is subtly different from `numpy.isscalar` in that\n            `numpy.isscalar` returns False for a zero-dimensional array\n            (e.g. ``np.array(1)``), while this is True for quantities,\n            since quantities cannot represent true numpy scalars.\n        '
        return not self.shape
    _include_easy_conversion_members = False

    def __dir__(self):
        if False:
            return 10
        '\n        Quantities are able to directly convert to other units that\n        have the same physical type.  This function is implemented in\n        order to make autocompletion still work correctly in IPython.\n        '
        if not self._include_easy_conversion_members:
            return super().__dir__()
        dir_values = set(super().__dir__())
        equivalencies = Unit._normalize_equivalencies(self.equivalencies)
        for equivalent in self.unit._get_units_with_same_physical_type(equivalencies):
            dir_values.update(equivalent.names)
        return sorted(dir_values)

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        '\n        Quantities are able to directly convert to other units that\n        have the same physical type.\n        '
        if not self._include_easy_conversion_members:
            raise AttributeError(f"'{self.__class__.__name__}' object has no '{attr}' member")

        def get_virtual_unit_attribute():
            if False:
                return 10
            registry = get_current_unit_registry().registry
            to_unit = registry.get(attr, None)
            if to_unit is None:
                return None
            try:
                return self.unit.to(to_unit, self.value, equivalencies=self.equivalencies)
            except UnitsError:
                return None
        value = get_virtual_unit_attribute()
        if value is None:
            raise AttributeError(f"{self.__class__.__name__} instance has no attribute '{attr}'")
        else:
            return value

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        try:
            other_value = self._to_own_unit(other)
        except UnitsError:
            return False
        except Exception:
            return NotImplemented
        return self.value.__eq__(other_value)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        try:
            other_value = self._to_own_unit(other)
        except UnitsError:
            return True
        except Exception:
            return NotImplemented
        return self.value.__ne__(other_value)

    def __lshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        try:
            other = Unit(other, parse_strict='silent')
        except UnitTypeError:
            return NotImplemented
        return self.__class__(self, other, copy=False, subok=True)

    def __ilshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        try:
            other = Unit(other, parse_strict='silent')
        except UnitTypeError:
            return NotImplemented
        try:
            factor = self.unit._to(other)
        except UnitConversionError:
            return NotImplemented
        except AttributeError:
            return NotImplemented
        view = self.view(np.ndarray)
        try:
            view *= factor
        except TypeError:
            return NotImplemented
        self._set_unit(other)
        return self

    def __rlshift__(self, other):
        if False:
            i = 10
            return i + 15
        if not self.isscalar:
            return NotImplemented
        return Unit(self).__rlshift__(other)

    def __rrshift__(self, other):
        if False:
            i = 10
            return i + 15
        warnings.warn(">> is not implemented. Did you mean to convert something to this quantity as a unit using '<<'?", AstropyWarning)
        return NotImplemented

    def __rshift__(self, other):
        if False:
            while True:
                i = 10
        return NotImplemented

    def __irshift__(self, other):
        if False:
            while True:
                i = 10
        return NotImplemented

    def __mul__(self, other):
        if False:
            while True:
                i = 10
        'Multiplication between `Quantity` objects and other objects.'
        if isinstance(other, (UnitBase, str)):
            try:
                return self._new_view(self.value.copy(), other * self.unit, propagate_info=False)
            except UnitsError:
                return NotImplemented
        return super().__mul__(other)

    def __imul__(self, other):
        if False:
            while True:
                i = 10
        'In-place multiplication between `Quantity` objects and others.'
        if isinstance(other, (UnitBase, str)):
            self._set_unit(other * self.unit)
            return self
        return super().__imul__(other)

    def __rmul__(self, other):
        if False:
            return 10
        '\n        Right Multiplication between `Quantity` objects and other objects.\n        '
        return self.__mul__(other)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        'Division between `Quantity` objects and other objects.'
        if isinstance(other, (UnitBase, str)):
            try:
                return self._new_view(self.value.copy(), self.unit / other, propagate_info=False)
            except UnitsError:
                return NotImplemented
        return super().__truediv__(other)

    def __itruediv__(self, other):
        if False:
            i = 10
            return i + 15
        'Inplace division between `Quantity` objects and other objects.'
        if isinstance(other, (UnitBase, str)):
            self._set_unit(self.unit / other)
            return self
        return super().__itruediv__(other)

    def __rtruediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Right Division between `Quantity` objects and other objects.'
        if isinstance(other, (UnitBase, str)):
            return self._new_view(1.0 / self.value, other / self.unit, propagate_info=False)
        return super().__rtruediv__(other)

    def __pow__(self, other):
        if False:
            return 10
        if isinstance(other, Fraction):
            return self._new_view(self.value ** float(other), self.unit ** other, propagate_info=False)
        return super().__pow__(other)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(self.value) ^ hash(self.unit)

    def __iter__(self):
        if False:
            while True:
                i = 10
        if self.isscalar:
            raise TypeError(f"'{self.__class__.__name__}' object with a scalar value is not iterable")

        def quantity_iter():
            if False:
                return 10
            for val in self.value:
                yield self._new_view(val)
        return quantity_iter()

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        if isinstance(key, str) and isinstance(self.unit, StructuredUnit):
            return self._new_view(self.view(np.ndarray)[key], self.unit[key], propagate_info=False)
        try:
            out = super().__getitem__(key)
        except IndexError:
            if self.isscalar:
                raise TypeError(f"'{self.__class__.__name__}' object with a scalar value does not support indexing")
            else:
                raise
        if not isinstance(out, np.ndarray):
            out = self._new_view(out)
        return out

    def __setitem__(self, i, value):
        if False:
            while True:
                i = 10
        if isinstance(i, str):
            self[i][...] = value
            return
        if not self.isscalar and 'info' in self.__dict__:
            self.info.adjust_indices(i, value, len(self))
        self.view(np.ndarray).__setitem__(i, self._to_own_unit(value))

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        'This method raises ValueError, since truthiness of quantities is ambiguous,\n        especially for logarithmic units and temperatures. Use explicit comparisons.\n        '
        raise ValueError(f'{type(self).__name__} truthiness is ambiguous, especially for logarithmic units and temperatures. Use explicit comparisons.')

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.isscalar:
            raise TypeError(f"'{self.__class__.__name__}' object with a scalar value has no len()")
        else:
            return len(self.value)

    def __float__(self):
        if False:
            print('Hello World!')
        try:
            return float(self.to_value(dimensionless_unscaled))
        except (UnitsError, TypeError):
            raise TypeError('only dimensionless scalar quantities can be converted to Python scalars')

    def __int__(self):
        if False:
            print('Hello World!')
        try:
            return int(self.to_value(dimensionless_unscaled))
        except (UnitsError, TypeError):
            raise TypeError('only dimensionless scalar quantities can be converted to Python scalars')

    def __index__(self):
        if False:
            return 10
        try:
            assert self.unit.is_unity()
            return self.value.__index__()
        except Exception:
            raise TypeError('only integer dimensionless scalar quantities can be converted to a Python index')

    @property
    def _unitstr(self):
        if False:
            i = 10
            return i + 15
        if self.unit is None:
            unitstr = _UNIT_NOT_INITIALISED
        else:
            unitstr = str(self.unit)
        if unitstr:
            unitstr = ' ' + unitstr
        return unitstr

    def to_string(self, unit=None, precision=None, format=None, subfmt=None):
        if False:
            while True:
                i = 10
        "\n        Generate a string representation of the quantity and its unit.\n\n        The behavior of this function can be altered via the\n        `numpy.set_printoptions` function and its various keywords.  The\n        exception to this is the ``threshold`` keyword, which is controlled via\n        the ``[units.quantity]`` configuration item ``latex_array_threshold``.\n        This is treated separately because the numpy default of 1000 is too big\n        for most browsers to handle.\n\n        Parameters\n        ----------\n        unit : unit-like, optional\n            Specifies the unit.  If not provided,\n            the unit used to initialize the quantity will be used.\n\n        precision : number, optional\n            The level of decimal precision. If `None`, or not provided,\n            it will be determined from NumPy print options.\n\n        format : str, optional\n            The format of the result. If not provided, an unadorned\n            string is returned. Supported values are:\n\n            - 'latex': Return a LaTeX-formatted string\n\n            - 'latex_inline': Return a LaTeX-formatted string that uses\n              negative exponents instead of fractions\n\n        subfmt : str, optional\n            Subformat of the result. For the moment, only used for\n            ``format='latex'`` and ``format='latex_inline'``. Supported\n            values are:\n\n            - 'inline': Use ``$ ... $`` as delimiters.\n\n            - 'display': Use ``$\\displaystyle ... $`` as delimiters.\n\n        Returns\n        -------\n        str\n            A string with the contents of this Quantity\n        "
        if unit is not None and unit != self.unit:
            return self.to(unit).to_string(unit=None, precision=precision, format=format, subfmt=subfmt)
        formats = {None: None, 'latex': {None: ('$', '$'), 'inline': ('$', '$'), 'display': ('$\\displaystyle ', '$')}}
        formats['latex_inline'] = formats['latex']
        if format not in formats:
            raise ValueError(f"Unknown format '{format}'")
        elif format is None:
            if precision is None:
                return f'{self.value}{self._unitstr:s}'
            else:
                return np.array2string(self.value, precision=precision, floatmode='fixed') + self._unitstr
        pops = np.get_printoptions()
        format_spec = f".{(precision if precision is not None else pops['precision'])}g"

        def float_formatter(value):
            if False:
                for i in range(10):
                    print('nop')
            return Latex.format_exponential_notation(value, format_spec=format_spec)

        def complex_formatter(value):
            if False:
                print('Hello World!')
            return '({}{}i)'.format(Latex.format_exponential_notation(value.real, format_spec=format_spec), Latex.format_exponential_notation(value.imag, format_spec='+' + format_spec))
        latex_value = np.array2string(self.view(np.ndarray), threshold=conf.latex_array_threshold if conf.latex_array_threshold > -1 else pops['threshold'], formatter={'float_kind': float_formatter, 'complex_kind': complex_formatter}, max_line_width=np.inf, separator=',~')
        latex_value = latex_value.replace('...', '\\dots')
        if self.unit is None:
            latex_unit = _UNIT_NOT_INITIALISED
        elif format == 'latex':
            latex_unit = self.unit._repr_latex_()[1:-1]
        elif format == 'latex_inline':
            latex_unit = self.unit.to_string(format='latex_inline')[1:-1]
        (delimiter_left, delimiter_right) = formats[format][subfmt]
        return f'{delimiter_left}{latex_value} \\; {latex_unit}{delimiter_right}'

    def __str__(self):
        if False:
            while True:
                i = 10
        return self.to_string()

    def __repr__(self):
        if False:
            print('Hello World!')
        prefixstr = '<' + self.__class__.__name__ + ' '
        arrstr = np.array2string(self.view(np.ndarray), separator=', ', prefix=prefixstr)
        return f'{prefixstr}{arrstr}{self._unitstr:s}>'

    def _repr_latex_(self):
        if False:
            return 10
        '\n        Generate a latex representation of the quantity and its unit.\n\n        Returns\n        -------\n        lstr\n            A LaTeX string with the contents of this Quantity\n        '
        return self.to_string(format='latex', subfmt='inline')

    def __format__(self, format_spec):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.to_string(format=format_spec)
        except ValueError:
            if format_spec in Base.registry:
                if self.unit is dimensionless_unscaled:
                    return f'{self.value}'
                else:
                    return f'{self.value} {format(self.unit, format_spec)}'
            try:
                return f'{format(self.value, format_spec)}{self._unitstr:s}'
            except ValueError:
                return format(f'{self.value}{self._unitstr:s}', format_spec)

    def decompose(self, bases=[]):
        if False:
            print('Hello World!')
        "\n        Generates a new `Quantity` with the units\n        decomposed. Decomposed units have only irreducible units in\n        them (see `astropy.units.UnitBase.decompose`).\n\n        Parameters\n        ----------\n        bases : sequence of `~astropy.units.UnitBase`, optional\n            The bases to decompose into.  When not provided,\n            decomposes down to any irreducible units.  When provided,\n            the decomposed result will only contain the given units.\n            This will raises a `~astropy.units.UnitsError` if it's not possible\n            to do so.\n\n        Returns\n        -------\n        newq : `~astropy.units.Quantity`\n            A new object equal to this quantity with units decomposed.\n        "
        return self._decompose(False, bases=bases)

    def _decompose(self, allowscaledunits=False, bases=[]):
        if False:
            return 10
        "\n        Generates a new `Quantity` with the units decomposed. Decomposed\n        units have only irreducible units in them (see\n        `astropy.units.UnitBase.decompose`).\n\n        Parameters\n        ----------\n        allowscaledunits : bool\n            If True, the resulting `Quantity` may have a scale factor\n            associated with it.  If False, any scaling in the unit will\n            be subsumed into the value of the resulting `Quantity`\n\n        bases : sequence of UnitBase, optional\n            The bases to decompose into.  When not provided,\n            decomposes down to any irreducible units.  When provided,\n            the decomposed result will only contain the given units.\n            This will raises a `~astropy.units.UnitsError` if it's not possible\n            to do so.\n\n        Returns\n        -------\n        newq : `~astropy.units.Quantity`\n            A new object equal to this quantity with units decomposed.\n\n        "
        new_unit = self.unit.decompose(bases=bases)
        if not allowscaledunits and hasattr(new_unit, 'scale'):
            new_value = self.value * new_unit.scale
            new_unit = new_unit / new_unit.scale
            return self._new_view(new_value, new_unit)
        else:
            return self._new_view(self.copy(), new_unit)

    def item(self, *args):
        if False:
            print('Hello World!')
        'Copy an element of an array to a scalar Quantity and return it.\n\n        Like :meth:`~numpy.ndarray.item` except that it always\n        returns a `Quantity`, not a Python scalar.\n\n        '
        return self._new_view(super().item(*args))

    def tolist(self):
        if False:
            while True:
                i = 10
        raise NotImplementedError('cannot make a list of Quantities. Get list of values with q.value.tolist().')

    def _to_own_unit(self, value, check_precision=True, *, unit=None):
        if False:
            while True:
                i = 10
        "Convert value to one's own unit (or that given).\n\n        Here, non-quantities are treated as dimensionless, and care is taken\n        for values of 0, infinity or nan, which are allowed to have any unit.\n\n        Parameters\n        ----------\n        value : anything convertible to `~astropy.units.Quantity`\n            The value to be converted to the requested unit.\n        check_precision : bool\n            Whether to forbid conversion of float to integer if that changes\n            the input number.  Default: `True`.\n        unit : `~astropy.units.Unit` or None\n            The unit to convert to.  By default, the unit of ``self``.\n\n        Returns\n        -------\n        value : number or `~numpy.ndarray`\n            In the requested units.\n\n        "
        if unit is None:
            unit = self.unit
        try:
            _value = value.to_value(unit)
        except AttributeError:
            if value is np.ma.masked or (value is np.ma.masked_print_option and self.dtype.kind == 'O'):
                return value
            try:
                as_quantity = Quantity(value)
                _value = as_quantity.to_value(unit)
            except UnitsError:
                if not hasattr(value, 'unit') and can_have_arbitrary_unit(as_quantity.value):
                    _value = as_quantity.value
                else:
                    raise
        if self.dtype.kind == 'i' and check_precision:
            _value = np.array(_value, copy=False, subok=True)
            if not np.can_cast(_value.dtype, self.dtype):
                self_dtype_array = np.array(_value, self.dtype, subok=True)
                if not np.all((self_dtype_array == _value) | np.isnan(_value)):
                    raise TypeError('cannot convert value type to array type without precision loss')
        if _value.dtype.names is not None:
            _value = _value.astype(self.dtype, copy=False)
        return _value
    if NUMPY_LT_2_0:

        def itemset(self, *args):
            if False:
                for i in range(10):
                    print('nop')
            if len(args) == 0:
                raise ValueError('itemset must have at least one argument')
            self.view(np.ndarray).itemset(*args[:-1] + (self._to_own_unit(args[-1]),))

    def tostring(self, order='C'):
        if False:
            return 10
        'Not implemented, use ``.value.tostring()`` instead.'
        raise NotImplementedError('cannot write Quantities to string.  Write array with q.value.tostring(...).')

    def tobytes(self, order='C'):
        if False:
            i = 10
            return i + 15
        'Not implemented, use ``.value.tobytes()`` instead.'
        raise NotImplementedError('cannot write Quantities to bytes.  Write array with q.value.tobytes(...).')

    def tofile(self, fid, sep='', format='%s'):
        if False:
            return 10
        'Not implemented, use ``.value.tofile()`` instead.'
        raise NotImplementedError('cannot write Quantities to file.  Write array with q.value.tofile(...)')

    def dump(self, file):
        if False:
            for i in range(10):
                print('nop')
        'Not implemented, use ``.value.dump()`` instead.'
        raise NotImplementedError('cannot dump Quantities to file.  Write array with q.value.dump()')

    def dumps(self):
        if False:
            i = 10
            return i + 15
        'Not implemented, use ``.value.dumps()`` instead.'
        raise NotImplementedError('cannot dump Quantities to string.  Write array with q.value.dumps()')

    def fill(self, value):
        if False:
            i = 10
            return i + 15
        self.view(np.ndarray).fill(self._to_own_unit(value))

    @property
    def flat(self):
        if False:
            while True:
                i = 10
        "A 1-D iterator over the Quantity array.\n\n        This returns a ``QuantityIterator`` instance, which behaves the same\n        as the `~numpy.flatiter` instance returned by `~numpy.ndarray.flat`,\n        and is similar to, but not a subclass of, Python's built-in iterator\n        object.\n        "
        return QuantityIterator(self)

    @flat.setter
    def flat(self, value):
        if False:
            return 10
        y = self.ravel()
        y[:] = value

    def take(self, indices, axis=None, out=None, mode='raise'):
        if False:
            for i in range(10):
                print('nop')
        out = super().take(indices, axis=axis, out=out, mode=mode)
        if type(out) is not type(self):
            out = self._new_view(out)
        return out

    def put(self, indices, values, mode='raise'):
        if False:
            while True:
                i = 10
        self.view(np.ndarray).put(indices, self._to_own_unit(values), mode)

    def choose(self, choices, out=None, mode='raise'):
        if False:
            while True:
                i = 10
        raise NotImplementedError('cannot choose based on quantity.  Choose using array with q.value.choose(...)')

    def argsort(self, axis=-1, kind='quicksort', order=None):
        if False:
            while True:
                i = 10
        return self.view(np.ndarray).argsort(axis=axis, kind=kind, order=order)

    def searchsorted(self, v, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return np.searchsorted(np.array(self), self._to_own_unit(v, check_precision=False), *args, **kwargs)

    def argmax(self, axis=None, out=None, *, keepdims=False):
        if False:
            while True:
                i = 10
        return self.view(np.ndarray).argmax(axis=axis, out=out, keepdims=keepdims)

    def argmin(self, axis=None, out=None, *, keepdims=False):
        if False:
            return 10
        return self.view(np.ndarray).argmin(axis=axis, out=out, keepdims=keepdims)

    def __array_function__(self, function, types, args, kwargs):
        if False:
            i = 10
            return i + 15
        'Wrap numpy functions, taking care of units.\n\n        Parameters\n        ----------\n        function : callable\n            Numpy function to wrap\n        types : iterable of classes\n            Classes that provide an ``__array_function__`` override. Can\n            in principle be used to interact with other classes. Below,\n            mostly passed on to `~numpy.ndarray`, which can only interact\n            with subclasses.\n        args : tuple\n            Positional arguments provided in the function call.\n        kwargs : dict\n            Keyword arguments provided in the function call.\n\n        Returns\n        -------\n        result: `~astropy.units.Quantity`, `~numpy.ndarray`\n            As appropriate for the function.  If the function is not\n            supported, `NotImplemented` is returned, which will lead to\n            a `TypeError` unless another argument overrode the function.\n\n        Raises\n        ------\n        ~astropy.units.UnitsError\n            If operands have incompatible units.\n        '
        if function in SUBCLASS_SAFE_FUNCTIONS:
            return super().__array_function__(function, types, args, kwargs)
        elif function in FUNCTION_HELPERS:
            function_helper = FUNCTION_HELPERS[function]
            try:
                (args, kwargs, unit, out) = function_helper(*args, **kwargs)
            except NotImplementedError:
                return self._not_implemented_or_raise(function, types)
            result = super().__array_function__(function, types, args, kwargs)
        elif function in DISPATCHED_FUNCTIONS:
            dispatched_function = DISPATCHED_FUNCTIONS[function]
            try:
                (result, unit, out) = dispatched_function(*args, **kwargs)
            except NotImplementedError:
                return self._not_implemented_or_raise(function, types)
        elif function in UNSUPPORTED_FUNCTIONS:
            return NotImplemented
        else:
            warnings.warn(f"function '{function.__name__}' is not known to astropy's Quantity. Will run it anyway, hoping it will treat ndarray subclasses correctly. Please raise an issue at https://github.com/astropy/astropy/issues.", AstropyWarning)
            return super().__array_function__(function, types, args, kwargs)
        if unit is None or result is NotImplemented:
            return result
        return self._result_as_quantity(result, unit, out=out)

    def _not_implemented_or_raise(self, function, types):
        if False:
            while True:
                i = 10
        if any((issubclass(t, np.ndarray) and (not issubclass(t, Quantity)) for t in types)):
            raise TypeError(f'the Quantity implementation cannot handle {function} with the given arguments.') from None
        else:
            return NotImplemented

    def _wrap_function(self, function, *args, unit=None, out=None, **kwargs):
        if False:
            return 10
        'Wrap a numpy function that processes self, returning a Quantity.\n\n        Parameters\n        ----------\n        function : callable\n            Numpy function to wrap.\n        args : positional arguments\n            Any positional arguments to the function beyond the first argument\n            (which will be set to ``self``).\n        kwargs : keyword arguments\n            Keyword arguments to the function.\n\n        If present, the following arguments are treated specially:\n\n        unit : `~astropy.units.Unit`\n            Unit of the output result.  If not given, the unit of ``self``.\n        out : `~astropy.units.Quantity`\n            A Quantity instance in which to store the output.\n\n        Notes\n        -----\n        Output should always be assigned via a keyword argument, otherwise\n        no proper account of the unit is taken.\n\n        Returns\n        -------\n        out : `~astropy.units.Quantity`\n            Result of the function call, with the unit set properly.\n        '
        if unit is None:
            unit = self.unit
        args = (self.value,) + tuple((arg.value if isinstance(arg, Quantity) else arg for arg in args))
        if out is not None:
            arrays = tuple((arg for arg in args if isinstance(arg, np.ndarray)))
            kwargs['out'] = check_output(out, unit, arrays, function=function)
        result = function(*args, **kwargs)
        return self._result_as_quantity(result, unit, out)

    def trace(self, offset=0, axis1=0, axis2=1, dtype=None, out=None):
        if False:
            for i in range(10):
                print('nop')
        return self._wrap_function(np.trace, offset, axis1, axis2, dtype, out=out)

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
        if False:
            while True:
                i = 10
        return self._wrap_function(np.var, axis, dtype, out=out, ddof=ddof, keepdims=keepdims, where=where, unit=self.unit ** 2)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
        if False:
            for i in range(10):
                print('nop')
        return self._wrap_function(np.std, axis, dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        if False:
            i = 10
            return i + 15
        return self._wrap_function(np.mean, axis, dtype, out=out, keepdims=keepdims, where=where)

    def round(self, decimals=0, out=None):
        if False:
            for i in range(10):
                print('nop')
        return self._wrap_function(np.round, decimals, out=out)

    def dot(self, b, out=None):
        if False:
            print('Hello World!')
        result_unit = self.unit * getattr(b, 'unit', dimensionless_unscaled)
        return self._wrap_function(np.dot, b, out=out, unit=result_unit)

    def all(self, axis=None, out=None):
        if False:
            i = 10
            return i + 15
        raise TypeError('cannot evaluate truth value of quantities. Evaluate array with q.value.all(...)')

    def any(self, axis=None, out=None):
        if False:
            return 10
        raise TypeError('cannot evaluate truth value of quantities. Evaluate array with q.value.any(...)')

    def diff(self, n=1, axis=-1):
        if False:
            while True:
                i = 10
        return self._wrap_function(np.diff, n, axis)

    def ediff1d(self, to_end=None, to_begin=None):
        if False:
            while True:
                i = 10
        return self._wrap_function(np.ediff1d, to_end, to_begin)

    @deprecated('5.3', alternative='np.nansum', obj_type='method')
    def nansum(self, axis=None, out=None, keepdims=False, *, initial=None, where=True):
        if False:
            while True:
                i = 10
        if initial is not None:
            initial = self._to_own_unit(initial)
        return self._wrap_function(np.nansum, axis, out=out, keepdims=keepdims, initial=initial, where=where)

    def insert(self, obj, values, axis=None):
        if False:
            print('Hello World!')
        '\n        Insert values along the given axis before the given indices and return\n        a new `~astropy.units.Quantity` object.\n\n        This is a thin wrapper around the `numpy.insert` function.\n\n        Parameters\n        ----------\n        obj : int, slice or sequence of int\n            Object that defines the index or indices before which ``values`` is\n            inserted.\n        values : array-like\n            Values to insert.  If the type of ``values`` is different\n            from that of quantity, ``values`` is converted to the matching type.\n            ``values`` should be shaped so that it can be broadcast appropriately\n            The unit of ``values`` must be consistent with this quantity.\n        axis : int, optional\n            Axis along which to insert ``values``.  If ``axis`` is None then\n            the quantity array is flattened before insertion.\n\n        Returns\n        -------\n        out : `~astropy.units.Quantity`\n            A copy of quantity with ``values`` inserted.  Note that the\n            insertion does not occur in-place: a new quantity array is returned.\n\n        Examples\n        --------\n        >>> import astropy.units as u\n        >>> q = [1, 2] * u.m\n        >>> q.insert(0, 50 * u.cm)\n        <Quantity [ 0.5,  1.,  2.] m>\n\n        >>> q = [[1, 2], [3, 4]] * u.m\n        >>> q.insert(1, [10, 20] * u.m, axis=0)\n        <Quantity [[  1.,  2.],\n                   [ 10., 20.],\n                   [  3.,  4.]] m>\n\n        >>> q.insert(1, 10 * u.m, axis=1)\n        <Quantity [[  1., 10.,  2.],\n                   [  3., 10.,  4.]] m>\n\n        '
        out_array = np.insert(self.value, obj, self._to_own_unit(values), axis)
        return self._new_view(out_array)

class SpecificTypeQuantity(Quantity):
    """Superclass for Quantities of specific physical type.

    Subclasses of these work just like :class:`~astropy.units.Quantity`, except
    that they are for specific physical types (and may have methods that are
    only appropriate for that type).  Astropy examples are
    :class:`~astropy.coordinates.Angle` and
    :class:`~astropy.coordinates.Distance`

    At a minimum, subclasses should set ``_equivalent_unit`` to the unit
    associated with the physical type.
    """
    _equivalent_unit = None
    _unit = None
    _default_unit = None
    __array_priority__ = Quantity.__array_priority__ + 10

    def __quantity_subclass__(self, unit):
        if False:
            for i in range(10):
                print('nop')
        if unit.is_equivalent(self._equivalent_unit):
            return (type(self), True)
        else:
            return (super().__quantity_subclass__(unit)[0], False)

    def _set_unit(self, unit):
        if False:
            return 10
        if unit is None or not unit.is_equivalent(self._equivalent_unit):
            raise UnitTypeError("{} instances require units equivalent to '{}'".format(type(self).__name__, self._equivalent_unit) + (', but no unit was given.' if unit is None else f", so cannot set it to '{unit}'."))
        super()._set_unit(unit)

def isclose(a, b, rtol=1e-05, atol=None, equal_nan=False):
    if False:
        return 10
    '\n    Return a boolean array where two arrays are element-wise equal\n    within a tolerance.\n\n    Parameters\n    ----------\n    a, b : array-like or `~astropy.units.Quantity`\n        Input values or arrays to compare\n    rtol : array-like or `~astropy.units.Quantity`\n        The relative tolerance for the comparison, which defaults to\n        ``1e-5``.  If ``rtol`` is a :class:`~astropy.units.Quantity`,\n        then it must be dimensionless.\n    atol : number or `~astropy.units.Quantity`\n        The absolute tolerance for the comparison.  The units (or lack\n        thereof) of ``a``, ``b``, and ``atol`` must be consistent with\n        each other.  If `None`, ``atol`` defaults to zero in the\n        appropriate units.\n    equal_nan : `bool`\n        Whether to compare NaN’s as equal. If `True`, NaNs in ``a`` will\n        be considered equal to NaN’s in ``b``.\n\n    Notes\n    -----\n    This is a :class:`~astropy.units.Quantity`-aware version of\n    :func:`numpy.isclose`. However, this differs from the `numpy` function in\n    that the default for the absolute tolerance here is zero instead of\n    ``atol=1e-8`` in `numpy`, as there is no natural way to set a default\n    *absolute* tolerance given two inputs that may have differently scaled\n    units.\n\n    Raises\n    ------\n    `~astropy.units.UnitsError`\n        If the dimensions of ``a``, ``b``, or ``atol`` are incompatible,\n        or if ``rtol`` is not dimensionless.\n\n    See Also\n    --------\n    allclose\n    '
    return np.isclose(*_unquantify_allclose_arguments(a, b, rtol, atol), equal_nan)

def allclose(a, b, rtol=1e-05, atol=None, equal_nan=False) -> bool:
    if False:
        print('Hello World!')
    '\n    Whether two arrays are element-wise equal within a tolerance.\n\n    Parameters\n    ----------\n    a, b : array-like or `~astropy.units.Quantity`\n        Input values or arrays to compare\n    rtol : array-like or `~astropy.units.Quantity`\n        The relative tolerance for the comparison, which defaults to\n        ``1e-5``.  If ``rtol`` is a :class:`~astropy.units.Quantity`,\n        then it must be dimensionless.\n    atol : number or `~astropy.units.Quantity`\n        The absolute tolerance for the comparison.  The units (or lack\n        thereof) of ``a``, ``b``, and ``atol`` must be consistent with\n        each other.  If `None`, ``atol`` defaults to zero in the\n        appropriate units.\n    equal_nan : `bool`\n        Whether to compare NaN’s as equal. If `True`, NaNs in ``a`` will\n        be considered equal to NaN’s in ``b``.\n\n    Notes\n    -----\n    This is a :class:`~astropy.units.Quantity`-aware version of\n    :func:`numpy.allclose`. However, this differs from the `numpy` function in\n    that the default for the absolute tolerance here is zero instead of\n    ``atol=1e-8`` in `numpy`, as there is no natural way to set a default\n    *absolute* tolerance given two inputs that may have differently scaled\n    units.\n\n    Raises\n    ------\n    `~astropy.units.UnitsError`\n        If the dimensions of ``a``, ``b``, or ``atol`` are incompatible,\n        or if ``rtol`` is not dimensionless.\n\n    See Also\n    --------\n    isclose\n    '
    return np.allclose(*_unquantify_allclose_arguments(a, b, rtol, atol), equal_nan)

def _unquantify_allclose_arguments(actual, desired, rtol, atol):
    if False:
        while True:
            i = 10
    actual = Quantity(actual, subok=True, copy=False)
    desired = Quantity(desired, subok=True, copy=False)
    try:
        desired = desired.to(actual.unit)
    except UnitsError:
        raise UnitsError(f"Units for 'desired' ({desired.unit}) and 'actual' ({actual.unit}) are not convertible")
    if atol is None:
        atol = Quantity(0)
    else:
        atol = Quantity(atol, subok=True, copy=False)
        try:
            atol = atol.to(actual.unit)
        except UnitsError:
            raise UnitsError(f"Units for 'atol' ({atol.unit}) and 'actual' ({actual.unit}) are not convertible")
    rtol = Quantity(rtol, subok=True, copy=False)
    try:
        rtol = rtol.to(dimensionless_unscaled)
    except Exception:
        raise UnitsError("'rtol' should be dimensionless")
    return (actual.value, desired.value, rtol.value, atol.value)