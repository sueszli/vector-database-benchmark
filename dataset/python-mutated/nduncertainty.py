import weakref
from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
from astropy import log
from astropy.units import Quantity, Unit, UnitConversionError
__all__ = ['MissingDataAssociationException', 'IncompatibleUncertaintiesException', 'NDUncertainty', 'StdDevUncertainty', 'UnknownUncertainty', 'VarianceUncertainty', 'InverseVariance']
collapse_to_variance_mapping = {np.sum: np.square, np.mean: np.square}

def _move_preserved_axes_first(arr, preserve_axes):
    if False:
        i = 10
        return i + 15
    zeroth_axis_after_reshape = np.prod(np.array(arr.shape)[list(preserve_axes)])
    collapse_axes = [i for i in range(arr.ndim) if i not in preserve_axes]
    return arr.reshape([zeroth_axis_after_reshape] + np.array(arr.shape)[collapse_axes].tolist())

def _unravel_preserved_axes(arr, collapsed_arr, preserve_axes):
    if False:
        i = 10
        return i + 15
    if collapsed_arr.ndim != len(preserve_axes):
        arr_shape = np.array(arr.shape)
        return collapsed_arr.reshape(arr_shape[np.asarray(preserve_axes)])
    return collapsed_arr

def from_variance_for_mean(x, axis):
    if False:
        print('Hello World!')
    if axis is None:
        denom = np.ma.count(x)
    else:
        denom = np.ma.count(x, axis)
    return np.sqrt(np.ma.sum(x, axis)) / denom
collapse_from_variance_mapping = {np.sum: lambda x, axis: np.sqrt(np.ma.sum(x, axis)), np.mean: from_variance_for_mean, np.median: None}

class IncompatibleUncertaintiesException(Exception):
    """This exception should be used to indicate cases in which uncertainties
    with two different classes can not be propagated.
    """

class MissingDataAssociationException(Exception):
    """This exception should be used to indicate that an uncertainty instance
    has not been associated with a parent `~astropy.nddata.NDData` object.
    """

class NDUncertainty(metaclass=ABCMeta):
    """This is the metaclass for uncertainty classes used with `NDData`.

    Parameters
    ----------
    array : any type, optional
        The array or value (the parameter name is due to historical reasons) of
        the uncertainty. `numpy.ndarray`, `~astropy.units.Quantity` or
        `NDUncertainty` subclasses are recommended.
        If the `array` is `list`-like or `numpy.ndarray`-like it will be cast
        to a plain `numpy.ndarray`.
        Default is ``None``.

    unit : unit-like, optional
        Unit for the uncertainty ``array``. Strings that can be converted to a
        `~astropy.units.Unit` are allowed.
        Default is ``None``.

    copy : `bool`, optional
        Indicates whether to save the `array` as a copy. ``True`` copies it
        before saving, while ``False`` tries to save every parameter as
        reference. Note however that it is not always possible to save the
        input as reference.
        Default is ``True``.

    Raises
    ------
    IncompatibleUncertaintiesException
        If given another `NDUncertainty`-like class as ``array`` if their
        ``uncertainty_type`` is different.
    """

    def __init__(self, array=None, copy=True, unit=None):
        if False:
            while True:
                i = 10
        if isinstance(array, NDUncertainty):
            if array.uncertainty_type != self.uncertainty_type:
                raise IncompatibleUncertaintiesException
            if unit is not None and unit != array._unit:
                log.info("overwriting Uncertainty's current unit with specified unit.")
            elif array._unit is not None:
                unit = array.unit
            array = array.array
        elif isinstance(array, Quantity):
            if unit is not None and array.unit is not None and (unit != array.unit):
                log.info("overwriting Quantity's current unit with specified unit.")
            elif array.unit is not None:
                unit = array.unit
            array = array.value
        if unit is None:
            self._unit = None
        else:
            self._unit = Unit(unit)
        if copy:
            array = deepcopy(array)
            unit = deepcopy(unit)
        self.array = array
        self.parent_nddata = None

    @property
    @abstractmethod
    def uncertainty_type(self):
        if False:
            for i in range(10):
                print('nop')
        '`str` : Short description of the type of uncertainty.\n\n        Defined as abstract property so subclasses *have* to override this.\n        '
        return None

    @property
    def supports_correlated(self):
        if False:
            i = 10
            return i + 15
        '`bool` : Supports uncertainty propagation with correlated uncertainties?\n\n        .. versionadded:: 1.2\n        '
        return False

    @property
    def array(self):
        if False:
            for i in range(10):
                print('nop')
        "`numpy.ndarray` : the uncertainty's value."
        return self._array

    @array.setter
    def array(self, value):
        if False:
            i = 10
            return i + 15
        if isinstance(value, (list, np.ndarray)):
            value = np.array(value, subok=False, copy=False)
        self._array = value

    @property
    def unit(self):
        if False:
            for i in range(10):
                print('nop')
        '`~astropy.units.Unit` : The unit of the uncertainty, if any.'
        return self._unit

    @unit.setter
    def unit(self, value):
        if False:
            return 10
        '\n        The unit should be set to a value consistent with the parent NDData\n        unit and the uncertainty type.\n        '
        if value is not None:
            if self._parent_nddata is not None:
                parent_unit = self.parent_nddata.unit
                try:
                    self._data_unit_to_uncertainty_unit(parent_unit).to(value)
                except UnitConversionError:
                    raise UnitConversionError('Unit {} is incompatible with unit {} of parent nddata'.format(value, parent_unit))
            self._unit = Unit(value)
        else:
            self._unit = value

    @property
    def quantity(self):
        if False:
            i = 10
            return i + 15
        '\n        This uncertainty as an `~astropy.units.Quantity` object.\n        '
        return Quantity(self.array, self.unit, copy=False, dtype=self.array.dtype)

    @property
    def parent_nddata(self):
        if False:
            i = 10
            return i + 15
        '`NDData` : reference to `NDData` instance with this uncertainty.\n\n        In case the reference is not set uncertainty propagation will not be\n        possible since propagation might need the uncertain data besides the\n        uncertainty.\n        '
        no_parent_message = 'uncertainty is not associated with an NDData object'
        parent_lost_message = 'the associated NDData object was deleted and cannot be accessed anymore. You can prevent the NDData object from being deleted by assigning it to a variable. If this happened after unpickling make sure you pickle the parent not the uncertainty directly.'
        try:
            parent = self._parent_nddata
        except AttributeError:
            raise MissingDataAssociationException(no_parent_message)
        else:
            if parent is None:
                raise MissingDataAssociationException(no_parent_message)
            elif isinstance(self._parent_nddata, weakref.ref):
                resolved_parent = self._parent_nddata()
                if resolved_parent is None:
                    log.info(parent_lost_message)
                return resolved_parent
            else:
                log.info('parent_nddata should be a weakref to an NDData object.')
                return self._parent_nddata

    @parent_nddata.setter
    def parent_nddata(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value is not None and (not isinstance(value, weakref.ref)):
            value = weakref.ref(value)
        self._parent_nddata = value
        if value is not None:
            parent_unit = self.parent_nddata.unit
            parent_data_unit = getattr(self.parent_nddata.data, 'unit', None)
            if parent_unit is None and parent_data_unit is None:
                self.unit = None
            elif self.unit is None and parent_unit is not None:
                self.unit = self._data_unit_to_uncertainty_unit(parent_unit)
            elif parent_data_unit is not None:
                self.unit = self._data_unit_to_uncertainty_unit(parent_data_unit)
            else:
                unit_from_data = self._data_unit_to_uncertainty_unit(parent_unit)
                try:
                    unit_from_data.to(self.unit)
                except UnitConversionError:
                    raise UnitConversionError(f'Unit {self.unit} of uncertainty incompatible with unit {parent_unit} of data')

    @abstractmethod
    def _data_unit_to_uncertainty_unit(self, value):
        if False:
            return 10
        '\n        Subclasses must override this property. It should take in a data unit\n        and return the correct unit for the uncertainty given the uncertainty\n        type.\n        '
        return None

    def __repr__(self):
        if False:
            print('Hello World!')
        prefix = self.__class__.__name__ + '('
        try:
            body = np.array2string(self.array, separator=', ', prefix=prefix)
        except AttributeError:
            body = str(self.array)
        return f'{prefix}{body})'

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        try:
            return (self._array, self._unit, self.parent_nddata)
        except MissingDataAssociationException:
            return (self._array, self._unit, None)

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        if len(state) != 3:
            raise TypeError('The state should contain 3 items.')
        self._array = state[0]
        self._unit = state[1]
        parent = state[2]
        if parent is not None:
            parent = weakref.ref(parent)
        self._parent_nddata = parent

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        'Normal slicing on the array, keep the unit and return a reference.'
        return self.__class__(self.array[item], unit=self.unit, copy=False)

    def propagate(self, operation, other_nddata, result_data, correlation, axis=None):
        if False:
            i = 10
            return i + 15
        'Calculate the resulting uncertainty given an operation on the data.\n\n        .. versionadded:: 1.2\n\n        Parameters\n        ----------\n        operation : callable\n            The operation that is performed on the `NDData`. Supported are\n            `numpy.add`, `numpy.subtract`, `numpy.multiply` and\n            `numpy.true_divide` (or `numpy.divide`).\n\n        other_nddata : `NDData` instance\n            The second operand in the arithmetic operation.\n\n        result_data : `~astropy.units.Quantity` or ndarray\n            The result of the arithmetic operations on the data.\n\n        correlation : `numpy.ndarray` or number\n            The correlation (rho) is defined between the uncertainties in\n            sigma_AB = sigma_A * sigma_B * rho. A value of ``0`` means\n            uncorrelated operands.\n\n        axis : int or tuple of ints, optional\n            Axis over which to perform a collapsing operation.\n\n        Returns\n        -------\n        resulting_uncertainty : `NDUncertainty` instance\n            Another instance of the same `NDUncertainty` subclass containing\n            the uncertainty of the result.\n\n        Raises\n        ------\n        ValueError\n            If the ``operation`` is not supported or if correlation is not zero\n            but the subclass does not support correlated uncertainties.\n\n        Notes\n        -----\n        First this method checks if a correlation is given and the subclass\n        implements propagation with correlated uncertainties.\n        Then the second uncertainty is converted (or an Exception is raised)\n        to the same class in order to do the propagation.\n        Then the appropriate propagation method is invoked and the result is\n        returned.\n        '
        if not self.supports_correlated:
            if isinstance(correlation, np.ndarray) or correlation != 0:
                raise ValueError('{} does not support uncertainty propagation with correlation.'.format(self.__class__.__name__))
        if other_nddata is not None:
            other_uncert = self._convert_uncertainty(other_nddata.uncertainty)
            if operation.__name__ == 'add':
                result = self._propagate_add(other_uncert, result_data, correlation)
            elif operation.__name__ == 'subtract':
                result = self._propagate_subtract(other_uncert, result_data, correlation)
            elif operation.__name__ == 'multiply':
                result = self._propagate_multiply(other_uncert, result_data, correlation)
            elif operation.__name__ in ['true_divide', 'divide']:
                result = self._propagate_divide(other_uncert, result_data, correlation)
            else:
                raise ValueError(f'unsupported operation: {operation.__name__}')
        else:
            result = self._propagate_collapse(operation, axis)
        return self.__class__(result, copy=False)

    def _convert_uncertainty(self, other_uncert):
        if False:
            i = 10
            return i + 15
        'Checks if the uncertainties are compatible for propagation.\n\n        Checks if the other uncertainty is `NDUncertainty`-like and if so\n        verify that the uncertainty_type is equal. If the latter is not the\n        case try returning ``self.__class__(other_uncert)``.\n\n        Parameters\n        ----------\n        other_uncert : `NDUncertainty` subclass\n            The other uncertainty.\n\n        Returns\n        -------\n        other_uncert : `NDUncertainty` subclass\n            but converted to a compatible `NDUncertainty` subclass if\n            possible and necessary.\n\n        Raises\n        ------\n        IncompatibleUncertaintiesException:\n            If the other uncertainty cannot be converted to a compatible\n            `NDUncertainty` subclass.\n        '
        if isinstance(other_uncert, NDUncertainty):
            if self.uncertainty_type == other_uncert.uncertainty_type:
                return other_uncert
            else:
                return self.__class__(other_uncert)
        else:
            raise IncompatibleUncertaintiesException

    @abstractmethod
    def _propagate_add(self, other_uncert, result_data, correlation):
        if False:
            for i in range(10):
                print('nop')
        return None

    @abstractmethod
    def _propagate_subtract(self, other_uncert, result_data, correlation):
        if False:
            print('Hello World!')
        return None

    @abstractmethod
    def _propagate_multiply(self, other_uncert, result_data, correlation):
        if False:
            i = 10
            return i + 15
        return None

    @abstractmethod
    def _propagate_divide(self, other_uncert, result_data, correlation):
        if False:
            print('Hello World!')
        return None

    def represent_as(self, other_uncert):
        if False:
            i = 10
            return i + 15
        'Convert this uncertainty to a different uncertainty type.\n\n        Parameters\n        ----------\n        other_uncert : `NDUncertainty` subclass\n            The `NDUncertainty` subclass to convert to.\n\n        Returns\n        -------\n        resulting_uncertainty : `NDUncertainty` instance\n            An instance of ``other_uncert`` subclass containing the uncertainty\n            converted to the new uncertainty type.\n\n        Raises\n        ------\n        TypeError\n            If either the initial or final subclasses do not support\n            conversion, a `TypeError` is raised.\n        '
        as_variance = getattr(self, '_convert_to_variance', None)
        if as_variance is None:
            raise TypeError(f'{type(self)} does not support conversion to another uncertainty type.')
        from_variance = getattr(other_uncert, '_convert_from_variance', None)
        if from_variance is None:
            raise TypeError(f'{other_uncert.__name__} does not support conversion from another uncertainty type.')
        return from_variance(as_variance())

class UnknownUncertainty(NDUncertainty):
    """This class implements any unknown uncertainty type.

    The main purpose of having an unknown uncertainty class is to prevent
    uncertainty propagation.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`
    """

    @property
    def supports_correlated(self):
        if False:
            print('Hello World!')
        '`False` : Uncertainty propagation is *not* possible for this class.'
        return False

    @property
    def uncertainty_type(self):
        if False:
            i = 10
            return i + 15
        '``"unknown"`` : `UnknownUncertainty` implements any unknown                            uncertainty type.\n        '
        return 'unknown'

    def _data_unit_to_uncertainty_unit(self, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        No way to convert if uncertainty is unknown.\n        '
        return None

    def _convert_uncertainty(self, other_uncert):
        if False:
            i = 10
            return i + 15
        'Raise an Exception because unknown uncertainty types cannot\n        implement propagation.\n        '
        msg = 'Uncertainties of unknown type cannot be propagated.'
        raise IncompatibleUncertaintiesException(msg)

    def _propagate_add(self, other_uncert, result_data, correlation):
        if False:
            for i in range(10):
                print('nop')
        'Not possible for unknown uncertainty types.'
        return None

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        if False:
            while True:
                i = 10
        return None

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        if False:
            return 10
        return None

    def _propagate_divide(self, other_uncert, result_data, correlation):
        if False:
            while True:
                i = 10
        return None

class _VariancePropagationMixin:
    """
    Propagation of uncertainties for variances, also used to perform error
    propagation for variance-like uncertainties (standard deviation and inverse
    variance).
    """

    def _propagate_collapse(self, numpy_op, axis=None):
        if False:
            i = 10
            return i + 15
        '\n        Error propagation for collapse operations on variance or\n        variance-like uncertainties. Uncertainties are calculated using the\n        formulae for variance but can be used for uncertainty convertible to\n        a variance.\n\n        Parameters\n        ----------\n        numpy_op : function\n            Numpy operation like `np.sum` or `np.max` to use in the collapse\n\n        subtract : bool, optional\n            If ``True``, propagate for subtraction, otherwise propagate for\n            addition.\n\n        axis : tuple, optional\n            Axis on which to compute collapsing operations.\n        '
        try:
            result_unit_sq = self.parent_nddata.unit ** 2
        except (AttributeError, TypeError):
            result_unit_sq = None
        if self.array is not None:
            if numpy_op in [np.min, np.max]:
                return self._get_err_at_extremum(numpy_op, axis=axis)
            else:
                to_variance = collapse_to_variance_mapping[numpy_op]
                from_variance = collapse_from_variance_mapping[numpy_op]
                masked_uncertainty = np.ma.masked_array(self.array, self.parent_nddata.mask)
                if self.unit is not None and to_variance(self.unit) != self.parent_nddata.unit ** 2:
                    this = to_variance(masked_uncertainty << self.unit).to(result_unit_sq).value
                else:
                    this = to_variance(masked_uncertainty)
                return from_variance(this, axis=axis)

    def _get_err_at_extremum(self, extremum, axis):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the value of the ``uncertainty`` array at the indices\n        which satisfy the ``extremum`` function applied to the ``measurement`` array,\n        where we expect ``extremum`` to be np.argmax or np.argmin, and\n        we expect a two-dimensional output.\n\n        Assumes the ``measurement`` and ``uncertainty`` array dimensions\n        are ordered such that the zeroth dimension is the one to preserve.\n        For example, if you start with array with shape (a, b, c), this\n        function applies the ``extremum`` function to the last two dimensions,\n        with shapes b and c.\n\n        This operation is difficult to cast in a vectorized way. Here\n        we implement it with a list comprehension, which is likely not the\n        most performant solution.\n        '
        if axis is not None and (not hasattr(axis, '__len__')):
            axis = [axis]
        if extremum is np.min:
            arg_extremum = np.ma.argmin
        elif extremum is np.max:
            arg_extremum = np.ma.argmax
        all_axes = np.arange(self.array.ndim)
        if axis is None:
            ind = arg_extremum(np.asanyarray(self.parent_nddata).ravel())
            return self.array.ravel()[ind]
        preserve_axes = [ax for ax in all_axes if ax not in axis]
        meas = np.ma.masked_array(_move_preserved_axes_first(self.parent_nddata.data, preserve_axes), _move_preserved_axes_first(self.parent_nddata.mask, preserve_axes))
        err = _move_preserved_axes_first(self.array, preserve_axes)
        result = np.array([e[np.unravel_index(arg_extremum(m), m.shape)] for (m, e) in zip(meas, err)])
        return _unravel_preserved_axes(self.parent_nddata.data, result, preserve_axes)

    def _propagate_add_sub(self, other_uncert, result_data, correlation, subtract=False, to_variance=lambda x: x, from_variance=lambda x: x):
        if False:
            i = 10
            return i + 15
        '\n        Error propagation for addition or subtraction of variance or\n        variance-like uncertainties. Uncertainties are calculated using the\n        formulae for variance but can be used for uncertainty convertible to\n        a variance.\n\n        Parameters\n        ----------\n        other_uncert : `~astropy.nddata.NDUncertainty` instance\n            The uncertainty, if any, of the other operand.\n\n        result_data : `~astropy.nddata.NDData` instance\n            The results of the operation on the data.\n\n        correlation : float or array-like\n            Correlation of the uncertainties.\n\n        subtract : bool, optional\n            If ``True``, propagate for subtraction, otherwise propagate for\n            addition.\n\n        to_variance : function, optional\n            Function that will transform the input uncertainties to variance.\n            The default assumes the uncertainty is the variance.\n\n        from_variance : function, optional\n            Function that will convert from variance to the input uncertainty.\n            The default assumes the uncertainty is the variance.\n        '
        if subtract:
            correlation_sign = -1
        else:
            correlation_sign = 1
        try:
            result_unit_sq = result_data.unit ** 2
        except AttributeError:
            result_unit_sq = None
        if other_uncert.array is not None:
            if other_uncert.unit is not None and result_unit_sq != to_variance(other_uncert.unit):
                other = to_variance(other_uncert.array << other_uncert.unit).to(result_unit_sq).value
            else:
                other = to_variance(other_uncert.array)
        else:
            other = 0
        if self.array is not None:
            if self.unit is not None and to_variance(self.unit) != self.parent_nddata.unit ** 2:
                this = to_variance(self.array << self.unit).to(result_unit_sq).value
            else:
                this = to_variance(self.array)
        else:
            this = 0
        if isinstance(correlation, np.ndarray) or correlation != 0:
            corr = 2 * correlation * np.sqrt(this * other)
            result = this + other + correlation_sign * corr
        else:
            result = this + other
        return from_variance(result)

    def _propagate_multiply_divide(self, other_uncert, result_data, correlation, divide=False, to_variance=lambda x: x, from_variance=lambda x: x):
        if False:
            while True:
                i = 10
        '\n        Error propagation for multiplication or division of variance or\n        variance-like uncertainties. Uncertainties are calculated using the\n        formulae for variance but can be used for uncertainty convertible to\n        a variance.\n\n        Parameters\n        ----------\n        other_uncert : `~astropy.nddata.NDUncertainty` instance\n            The uncertainty, if any, of the other operand.\n\n        result_data : `~astropy.nddata.NDData` instance\n            The results of the operation on the data.\n\n        correlation : float or array-like\n            Correlation of the uncertainties.\n\n        divide : bool, optional\n            If ``True``, propagate for division, otherwise propagate for\n            multiplication.\n\n        to_variance : function, optional\n            Function that will transform the input uncertainties to variance.\n            The default assumes the uncertainty is the variance.\n\n        from_variance : function, optional\n            Function that will convert from variance to the input uncertainty.\n            The default assumes the uncertainty is the variance.\n        '
        if isinstance(result_data, Quantity):
            result_data = result_data.value
        if divide:
            correlation_sign = -1
        else:
            correlation_sign = 1
        if other_uncert.array is not None:
            if other_uncert.unit and to_variance(1 * other_uncert.unit) != ((1 * other_uncert.parent_nddata.unit) ** 2).unit:
                d_b = to_variance(other_uncert.array << other_uncert.unit).to((1 * other_uncert.parent_nddata.unit) ** 2).value
            else:
                d_b = to_variance(other_uncert.array)
            right = np.abs(self.parent_nddata.data ** 2 * d_b)
        else:
            right = 0
        if self.array is not None:
            if self.unit and to_variance(1 * self.unit) != ((1 * self.parent_nddata.unit) ** 2).unit:
                d_a = to_variance(self.array << self.unit).to((1 * self.parent_nddata.unit) ** 2).value
            else:
                d_a = to_variance(self.array)
            left = np.abs(other_uncert.parent_nddata.data ** 2 * d_a)
        else:
            left = 0
        if isinstance(correlation, np.ndarray) or correlation != 0:
            corr = 2 * correlation * np.sqrt(d_a * d_b) * self.parent_nddata.data * other_uncert.parent_nddata.data
        else:
            corr = 0
        if divide:
            return from_variance((left + right + correlation_sign * corr) / other_uncert.parent_nddata.data ** 4)
        else:
            return from_variance(left + right + correlation_sign * corr)

class StdDevUncertainty(_VariancePropagationMixin, NDUncertainty):
    """Standard deviation uncertainty assuming first order gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `StdDevUncertainty`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will have the same unit as the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    `StdDevUncertainty` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, StdDevUncertainty
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=StdDevUncertainty([0.1, 0.1, 0.1]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        StdDevUncertainty([0.1, 0.1, 0.1])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = StdDevUncertainty([0.2], unit='m', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        StdDevUncertainty([0.2])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 2
        >>> ndd.uncertainty
        StdDevUncertainty(2)

    .. note::
        The unit will not be displayed.
    """

    @property
    def supports_correlated(self):
        if False:
            i = 10
            return i + 15
        '`True` : `StdDevUncertainty` allows to propagate correlated                     uncertainties.\n\n        ``correlation`` must be given, this class does not implement computing\n        it by itself.\n        '
        return True

    @property
    def uncertainty_type(self):
        if False:
            print('Hello World!')
        '``"std"`` : `StdDevUncertainty` implements standard deviation.'
        return 'std'

    def _convert_uncertainty(self, other_uncert):
        if False:
            print('Hello World!')
        if isinstance(other_uncert, StdDevUncertainty):
            return other_uncert
        else:
            raise IncompatibleUncertaintiesException

    def _propagate_add(self, other_uncert, result_data, correlation):
        if False:
            return 10
        return super()._propagate_add_sub(other_uncert, result_data, correlation, subtract=False, to_variance=np.square, from_variance=np.sqrt)

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        if False:
            while True:
                i = 10
        return super()._propagate_add_sub(other_uncert, result_data, correlation, subtract=True, to_variance=np.square, from_variance=np.sqrt)

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        if False:
            while True:
                i = 10
        return super()._propagate_multiply_divide(other_uncert, result_data, correlation, divide=False, to_variance=np.square, from_variance=np.sqrt)

    def _propagate_divide(self, other_uncert, result_data, correlation):
        if False:
            while True:
                i = 10
        return super()._propagate_multiply_divide(other_uncert, result_data, correlation, divide=True, to_variance=np.square, from_variance=np.sqrt)

    def _propagate_collapse(self, numpy_operation, axis):
        if False:
            return 10
        return super()._propagate_collapse(numpy_operation, axis=axis)

    def _data_unit_to_uncertainty_unit(self, value):
        if False:
            i = 10
            return i + 15
        return value

    def _convert_to_variance(self):
        if False:
            print('Hello World!')
        new_array = None if self.array is None else self.array ** 2
        new_unit = None if self.unit is None else self.unit ** 2
        return VarianceUncertainty(new_array, unit=new_unit)

    @classmethod
    def _convert_from_variance(cls, var_uncert):
        if False:
            while True:
                i = 10
        new_array = None if var_uncert.array is None else var_uncert.array ** (1 / 2)
        new_unit = None if var_uncert.unit is None else var_uncert.unit ** (1 / 2)
        return cls(new_array, unit=new_unit)

class VarianceUncertainty(_VariancePropagationMixin, NDUncertainty):
    """
    Variance uncertainty assuming first order Gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `VarianceUncertainty`. The class can handle if the uncertainty has a
    unit that differs from (but is convertible to) the parents `NDData` unit.
    The unit of the resulting uncertainty will be the square of the unit of the
    resulting data. Also support for correlation is possible but requires the
    correlation as input. It cannot handle correlation determination itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    Compare this example to that in `StdDevUncertainty`; the uncertainties
    in the examples below are equivalent to the uncertainties in
    `StdDevUncertainty`.

    `VarianceUncertainty` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, VarianceUncertainty
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=VarianceUncertainty([0.01, 0.01, 0.01]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        VarianceUncertainty([0.01, 0.01, 0.01])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = VarianceUncertainty([0.04], unit='m^2', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        VarianceUncertainty([0.04])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 4
        >>> ndd.uncertainty
        VarianceUncertainty(4)

    .. note::
        The unit will not be displayed.
    """

    @property
    def uncertainty_type(self):
        if False:
            return 10
        '``"var"`` : `VarianceUncertainty` implements variance.'
        return 'var'

    @property
    def supports_correlated(self):
        if False:
            for i in range(10):
                print('nop')
        '`True` : `VarianceUncertainty` allows to propagate correlated                     uncertainties.\n\n        ``correlation`` must be given, this class does not implement computing\n        it by itself.\n        '
        return True

    def _propagate_add(self, other_uncert, result_data, correlation):
        if False:
            print('Hello World!')
        return super()._propagate_add_sub(other_uncert, result_data, correlation, subtract=False)

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        if False:
            return 10
        return super()._propagate_add_sub(other_uncert, result_data, correlation, subtract=True)

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        if False:
            print('Hello World!')
        return super()._propagate_multiply_divide(other_uncert, result_data, correlation, divide=False)

    def _propagate_divide(self, other_uncert, result_data, correlation):
        if False:
            i = 10
            return i + 15
        return super()._propagate_multiply_divide(other_uncert, result_data, correlation, divide=True)

    def _data_unit_to_uncertainty_unit(self, value):
        if False:
            while True:
                i = 10
        return value ** 2

    def _convert_to_variance(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    @classmethod
    def _convert_from_variance(cls, var_uncert):
        if False:
            i = 10
            return i + 15
        return var_uncert

def _inverse(x):
    if False:
        print('Hello World!')
    'Just a simple inverse for use in the InverseVariance.'
    return 1 / x

class InverseVariance(_VariancePropagationMixin, NDUncertainty):
    """
    Inverse variance uncertainty assuming first order Gaussian error
    propagation.

    This class implements uncertainty propagation for ``addition``,
    ``subtraction``, ``multiplication`` and ``division`` with other instances
    of `InverseVariance`. The class can handle if the uncertainty has a unit
    that differs from (but is convertible to) the parents `NDData` unit. The
    unit of the resulting uncertainty will the inverse square of the unit of
    the resulting data. Also support for correlation is possible but requires
    the correlation as input. It cannot handle correlation determination
    itself.

    Parameters
    ----------
    args, kwargs :
        see `NDUncertainty`

    Examples
    --------
    Compare this example to that in `StdDevUncertainty`; the uncertainties
    in the examples below are equivalent to the uncertainties in
    `StdDevUncertainty`.

    `InverseVariance` should always be associated with an `NDData`-like
    instance, either by creating it during initialization::

        >>> from astropy.nddata import NDData, InverseVariance
        >>> ndd = NDData([1,2,3], unit='m',
        ...              uncertainty=InverseVariance([100, 100, 100]))
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        InverseVariance([100, 100, 100])

    or by setting it manually on the `NDData` instance::

        >>> ndd.uncertainty = InverseVariance([25], unit='1/m^2', copy=True)
        >>> ndd.uncertainty  # doctest: +FLOAT_CMP
        InverseVariance([25])

    the uncertainty ``array`` can also be set directly::

        >>> ndd.uncertainty.array = 0.25
        >>> ndd.uncertainty
        InverseVariance(0.25)

    .. note::
        The unit will not be displayed.
    """

    @property
    def uncertainty_type(self):
        if False:
            print('Hello World!')
        '``"ivar"`` : `InverseVariance` implements inverse variance.'
        return 'ivar'

    @property
    def supports_correlated(self):
        if False:
            print('Hello World!')
        '`True` : `InverseVariance` allows to propagate correlated                     uncertainties.\n\n        ``correlation`` must be given, this class does not implement computing\n        it by itself.\n        '
        return True

    def _propagate_add(self, other_uncert, result_data, correlation):
        if False:
            for i in range(10):
                print('nop')
        return super()._propagate_add_sub(other_uncert, result_data, correlation, subtract=False, to_variance=_inverse, from_variance=_inverse)

    def _propagate_subtract(self, other_uncert, result_data, correlation):
        if False:
            for i in range(10):
                print('nop')
        return super()._propagate_add_sub(other_uncert, result_data, correlation, subtract=True, to_variance=_inverse, from_variance=_inverse)

    def _propagate_multiply(self, other_uncert, result_data, correlation):
        if False:
            i = 10
            return i + 15
        return super()._propagate_multiply_divide(other_uncert, result_data, correlation, divide=False, to_variance=_inverse, from_variance=_inverse)

    def _propagate_divide(self, other_uncert, result_data, correlation):
        if False:
            i = 10
            return i + 15
        return super()._propagate_multiply_divide(other_uncert, result_data, correlation, divide=True, to_variance=_inverse, from_variance=_inverse)

    def _data_unit_to_uncertainty_unit(self, value):
        if False:
            i = 10
            return i + 15
        return 1 / value ** 2

    def _convert_to_variance(self):
        if False:
            for i in range(10):
                print('nop')
        new_array = None if self.array is None else 1 / self.array
        new_unit = None if self.unit is None else 1 / self.unit
        return VarianceUncertainty(new_array, unit=new_unit)

    @classmethod
    def _convert_from_variance(cls, var_uncert):
        if False:
            for i in range(10):
                print('nop')
        new_array = None if var_uncert.array is None else 1 / var_uncert.array
        new_unit = None if var_uncert.unit is None else 1 / var_uncert.unit
        return cls(new_array, unit=new_unit)