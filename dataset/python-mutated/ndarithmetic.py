import warnings
from copy import deepcopy
import numpy as np
from astropy.nddata.nduncertainty import NDUncertainty
from astropy.units import dimensionless_unscaled
from astropy.utils import format_doc, sharedmethod
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.masked import Masked
__all__ = ['NDArithmeticMixin']
_arit_doc = '\n    Performs {name} by evaluating ``self`` {op} ``operand``.\n\n    Parameters\n    ----------\n    operand, operand2 : `NDData`-like instance\n        If ``operand2`` is ``None`` or not given it will perform the operation\n        ``self`` {op} ``operand``.\n        If ``operand2`` is given it will perform ``operand`` {op} ``operand2``.\n        If the method was called on a class rather than on the instance\n        ``operand2`` must be given.\n\n    propagate_uncertainties : `bool` or ``None``, optional\n        If ``None`` the result will have no uncertainty. If ``False`` the\n        result will have a copied version of the first operand that has an\n        uncertainty. If ``True`` the result will have a correctly propagated\n        uncertainty from the uncertainties of the operands but this assumes\n        that the uncertainties are `NDUncertainty`-like. Default is ``True``.\n\n        .. versionchanged:: 1.2\n            This parameter must be given as keyword-parameter. Using it as\n            positional parameter is deprecated.\n            ``None`` was added as valid parameter value.\n\n    handle_mask : callable, ``\'first_found\'`` or ``None``, optional\n        If ``None`` the result will have no mask. If ``\'first_found\'`` the\n        result will have a copied version of the first operand that has a\n        mask). If it is a callable then the specified callable must\n        create the results ``mask`` and if necessary provide a copy.\n        Default is `numpy.logical_or`.\n\n        .. versionadded:: 1.2\n\n    handle_meta : callable, ``\'first_found\'`` or ``None``, optional\n        If ``None`` the result will have no meta. If ``\'first_found\'`` the\n        result will have a copied version of the first operand that has a\n        (not empty) meta. If it is a callable then the specified callable must\n        create the results ``meta`` and if necessary provide a copy.\n        Default is ``None``.\n\n        .. versionadded:: 1.2\n\n    compare_wcs : callable, ``\'first_found\'`` or ``None``, optional\n        If ``None`` the result will have no wcs and no comparison between\n        the wcs of the operands is made. If ``\'first_found\'`` the\n        result will have a copied version of the first operand that has a\n        wcs. If it is a callable then the specified callable must\n        compare the ``wcs``. The resulting ``wcs`` will be like if ``False``\n        was given otherwise it raises a ``ValueError`` if the comparison was\n        not successful. Default is ``\'first_found\'``.\n\n        .. versionadded:: 1.2\n\n    uncertainty_correlation : number or `~numpy.ndarray`, optional\n        The correlation between the two operands is used for correct error\n        propagation for correlated data as given in:\n        https://en.wikipedia.org/wiki/Propagation_of_uncertainty#Example_formulas\n        Default is 0.\n\n        .. versionadded:: 1.2\n\n\n    kwargs :\n        Any other parameter that should be passed to the callables used.\n\n    Returns\n    -------\n    result : `~astropy.nddata.NDData`-like\n        The resulting dataset\n\n    Notes\n    -----\n    If a ``callable`` is used for ``mask``, ``wcs`` or ``meta`` the\n    callable must accept the corresponding attributes as first two\n    parameters. If the callable also needs additional parameters these can be\n    defined as ``kwargs`` and must start with ``"wcs_"`` (for wcs callable) or\n    ``"meta_"`` (for meta callable). This startstring is removed before the\n    callable is called.\n\n    ``"first_found"`` can also be abbreviated with ``"ff"``.\n    '

class NDArithmeticMixin:
    """
    Mixin class to add arithmetic to an NDData object.

    When subclassing, be sure to list the superclasses in the correct order
    so that the subclass sees NDData as the main superclass. See
    `~astropy.nddata.NDDataArray` for an example.

    Notes
    -----
    This class only aims at covering the most common cases so there are certain
    restrictions on the saved attributes::

        - ``uncertainty`` : has to be something that has a `NDUncertainty`-like
          interface for uncertainty propagation
        - ``mask`` : has to be something that can be used by a bitwise ``or``
          operation.
        - ``wcs`` : has to implement a way of comparing with ``=`` to allow
          the operation.

    But there is a workaround that allows to disable handling a specific
    attribute and to simply set the results attribute to ``None`` or to
    copy the existing attribute (and neglecting the other).
    For example for uncertainties not representing an `NDUncertainty`-like
    interface you can alter the ``propagate_uncertainties`` parameter in
    :meth:`NDArithmeticMixin.add`. ``None`` means that the result will have no
    uncertainty, ``False`` means it takes the uncertainty of the first operand
    (if this does not exist from the second operand) as the result's
    uncertainty. This behavior is also explained in the docstring for the
    different arithmetic operations.

    Decomposing the units is not attempted, mainly due to the internal mechanics
    of `~astropy.units.Quantity`, so the resulting data might have units like
    ``km/m`` if you divided for example 100km by 5m. So this Mixin has adopted
    this behavior.

    Examples
    --------
    Using this Mixin with `~astropy.nddata.NDData`:

        >>> from astropy.nddata import NDData, NDArithmeticMixin
        >>> class NDDataWithMath(NDArithmeticMixin, NDData):
        ...     pass

    Using it with one operand on an instance::

        >>> ndd = NDDataWithMath(100)
        >>> ndd.add(20)
        NDDataWithMath(120)

    Using it with two operand on an instance::

        >>> ndd = NDDataWithMath(-4)
        >>> ndd.divide(1, ndd)
        NDDataWithMath(-0.25)

    Using it as classmethod requires two operands::

        >>> NDDataWithMath.subtract(5, 4)
        NDDataWithMath(1)

    """

    def _arithmetic(self, operation, operand, propagate_uncertainties=True, handle_mask=np.logical_or, handle_meta=None, uncertainty_correlation=0, compare_wcs='first_found', operation_ignores_mask=False, axis=None, **kwds):
        if False:
            return 10
        "\n        Base method which calculates the result of the arithmetic operation.\n\n        This method determines the result of the arithmetic operation on the\n        ``data`` including their units and then forwards to other methods\n        to calculate the other properties for the result (like uncertainty).\n\n        Parameters\n        ----------\n        operation : callable\n            The operation that is performed on the `NDData`. Supported are\n            `numpy.add`, `numpy.subtract`, `numpy.multiply` and\n            `numpy.true_divide`.\n\n        operand : same type (class) as self\n            see :meth:`NDArithmeticMixin.add`\n\n        propagate_uncertainties : `bool` or ``None``, optional\n            see :meth:`NDArithmeticMixin.add`\n\n        handle_mask : callable, ``'first_found'`` or ``None``, optional\n            see :meth:`NDArithmeticMixin.add`\n\n        handle_meta : callable, ``'first_found'`` or ``None``, optional\n            see :meth:`NDArithmeticMixin.add`\n\n        compare_wcs : callable, ``'first_found'`` or ``None``, optional\n            see :meth:`NDArithmeticMixin.add`\n\n        uncertainty_correlation : ``Number`` or `~numpy.ndarray`, optional\n            see :meth:`NDArithmeticMixin.add`\n\n        operation_ignores_mask : bool, optional\n            When True, masked values will be excluded from operations;\n            otherwise the operation will be performed on all values,\n            including masked ones.\n\n        axis : int or tuple of ints, optional\n            axis or axes over which to perform collapse operations like min, max, sum or mean.\n\n        kwargs :\n            Any other parameter that should be passed to the\n            different :meth:`NDArithmeticMixin._arithmetic_mask` (or wcs, ...)\n            methods.\n\n        Returns\n        -------\n        result : ndarray or `~astropy.units.Quantity`\n            The resulting data as array (in case both operands were without\n            unit) or as quantity if at least one had a unit.\n\n        kwargs : `dict`\n            The kwargs should contain all the other attributes (besides data\n            and unit) needed to create a new instance for the result. Creating\n            the new instance is up to the calling method, for example\n            :meth:`NDArithmeticMixin.add`.\n\n        "
        kwds2 = {'mask': {}, 'meta': {}, 'wcs': {}, 'data': {}, 'uncertainty': {}}
        for i in kwds:
            splitted = i.split('_', 1)
            try:
                kwds2[splitted[0]][splitted[1]] = kwds[i]
            except KeyError:
                raise KeyError(f'Unknown prefix {splitted[0]} for parameter {i}')
        kwargs = {}
        if compare_wcs is None:
            kwargs['wcs'] = None
        elif compare_wcs in ['ff', 'first_found']:
            if self.wcs is None and hasattr(operand, 'wcs'):
                kwargs['wcs'] = deepcopy(operand.wcs)
            else:
                kwargs['wcs'] = deepcopy(self.wcs)
        else:
            kwargs['wcs'] = self._arithmetic_wcs(operation, operand, compare_wcs, **kwds2['wcs'])
        use_masked_arith = operand is None and self.mask is not None
        if use_masked_arith:
            if not operation_ignores_mask:
                if self.unit is not None and (not hasattr(self.data, 'unit')):
                    masked_input = Masked(self.data << self.unit, mask=self.mask)
                else:
                    masked_input = Masked(self.data, mask=self.mask)
            else:
                masked_input = np.ma.masked_array(self.data, self.mask)
            result = operation(masked_input, axis=axis)
            if not hasattr(result, 'mask'):
                result = np.ma.masked_array(result, mask=np.zeros_like(result, dtype=bool))
        else:
            result = self._arithmetic_data(operation, operand, axis=axis, **kwds2['data'])
        if not hasattr(result, 'unit') and hasattr(self, 'unit'):
            kwargs['unit'] = self.unit
        if propagate_uncertainties is None:
            kwargs['uncertainty'] = None
        elif not propagate_uncertainties:
            if self.uncertainty is None:
                kwargs['uncertainty'] = deepcopy(operand.uncertainty)
            else:
                kwargs['uncertainty'] = deepcopy(self.uncertainty)
        else:
            kwargs['uncertainty'] = self._arithmetic_uncertainty(operation, operand, result, uncertainty_correlation, axis=axis, **kwds2['uncertainty'])
        if self.psf is not None or (operand is not None and operand.psf is not None):
            warnings.warn(f'Not setting psf attribute during {operation.__name__}.', AstropyUserWarning)
        if handle_mask is None:
            pass
        elif hasattr(result, 'mask'):
            kwargs['mask'] = None
        elif handle_mask in ['ff', 'first_found']:
            if self.mask is None:
                kwargs['mask'] = deepcopy(operand.mask)
            else:
                kwargs['mask'] = deepcopy(self.mask)
        else:
            kwargs['mask'] = self._arithmetic_mask(operation, operand, handle_mask, axis=axis, **kwds2['mask'])
        if handle_meta is None:
            kwargs['meta'] = None
        elif handle_meta in ['ff', 'first_found']:
            if not self.meta:
                kwargs['meta'] = deepcopy(operand.meta)
            else:
                kwargs['meta'] = deepcopy(self.meta)
        else:
            kwargs['meta'] = self._arithmetic_meta(operation, operand, handle_meta, **kwds2['meta'])
        return (result, kwargs)

    def _arithmetic_data(self, operation, operand, **kwds):
        if False:
            print('Hello World!')
        '\n        Calculate the resulting data.\n\n        Parameters\n        ----------\n        operation : callable\n            see `NDArithmeticMixin._arithmetic` parameter description.\n\n        operand : `NDData`-like instance\n            The second operand wrapped in an instance of the same class as\n            self.\n\n        kwds :\n            Additional parameters.\n\n        Returns\n        -------\n        result_data : ndarray or `~astropy.units.Quantity`\n            If both operands had no unit the resulting data is a simple numpy\n            array, but if any of the operands had a unit the return is a\n            Quantity.\n        '
        if self.unit is None and hasattr(operand, 'data'):
            if hasattr(operand, 'unit') and operand.unit is None:
                result = operation(self.data, operand.data)
            else:
                result = operation(self.data << dimensionless_unscaled, operand.data << operand.unit)
        elif hasattr(operand, 'unit'):
            if operand.unit is not None:
                result = operation(self.data << self.unit, operand.data << operand.unit)
            else:
                result = operation(self.data << self.unit, operand.data << dimensionless_unscaled)
        elif operand is not None:
            result = operation(self.data << self.unit, operand.data << operand.unit)
        else:
            result = operation(self.data, axis=kwds['axis'])
        return result

    def _arithmetic_uncertainty(self, operation, operand, result, correlation, **kwds):
        if False:
            return 10
        '\n        Calculate the resulting uncertainty.\n\n        Parameters\n        ----------\n        operation : callable\n            see :meth:`NDArithmeticMixin._arithmetic` parameter description.\n\n        operand : `NDData`-like instance\n            The second operand wrapped in an instance of the same class as\n            self.\n\n        result : `~astropy.units.Quantity` or `~numpy.ndarray`\n            The result of :meth:`NDArithmeticMixin._arithmetic_data`.\n\n        correlation : number or `~numpy.ndarray`\n            see :meth:`NDArithmeticMixin.add` parameter description.\n\n        kwds :\n            Additional parameters.\n\n        Returns\n        -------\n        result_uncertainty : `NDUncertainty` subclass instance or None\n            The resulting uncertainty already saved in the same `NDUncertainty`\n            subclass that ``self`` had (or ``operand`` if self had no\n            uncertainty). ``None`` only if both had no uncertainty.\n        '
        if self.uncertainty is not None and (not isinstance(self.uncertainty, NDUncertainty)):
            raise TypeError('Uncertainty propagation is only defined for subclasses of NDUncertainty.')
        if operand is not None and operand.uncertainty is not None and (not isinstance(operand.uncertainty, NDUncertainty)):
            raise TypeError('Uncertainty propagation is only defined for subclasses of NDUncertainty.')
        if self.uncertainty is None and (not hasattr(operand, 'uncertainty') or operand.uncertainty is None):
            return None
        elif self.uncertainty is None:
            self.uncertainty = operand.uncertainty.__class__(None)
            result_uncert = self.uncertainty.propagate(operation, operand, result, correlation)
            self.uncertainty = None
            return result_uncert
        elif operand is not None and operand.uncertainty is None:
            operand.uncertainty = self.uncertainty.__class__(None)
            result_uncert = self.uncertainty.propagate(operation, operand, result, correlation)
            operand.uncertainty = None
            return result_uncert
        else:
            axis_kwarg = dict(axis=kwds['axis']) if 'axis' in kwds else dict()
            return self.uncertainty.propagate(operation, operand, result, correlation, **axis_kwarg)

    def _arithmetic_mask(self, operation, operand, handle_mask, axis=None, **kwds):
        if False:
            i = 10
            return i + 15
        '\n        Calculate the resulting mask.\n\n        This is implemented as the piecewise ``or`` operation if both have a\n        mask.\n\n        Parameters\n        ----------\n        operation : callable\n            see :meth:`NDArithmeticMixin._arithmetic` parameter description.\n            By default, the ``operation`` will be ignored.\n\n        operand : `NDData`-like instance\n            The second operand wrapped in an instance of the same class as\n            self.\n\n        handle_mask : callable\n            see :meth:`NDArithmeticMixin.add`\n\n        kwds :\n            Additional parameters given to ``handle_mask``.\n\n        Returns\n        -------\n        result_mask : any type\n            If only one mask was present this mask is returned.\n            If neither had a mask ``None`` is returned. Otherwise\n            ``handle_mask`` must create (and copy) the returned mask.\n        '
        if self.mask is None and operand is not None and (operand.mask is None) or handle_mask is None:
            return None
        elif self.mask is None and operand is not None:
            return deepcopy(operand.mask)
        elif operand is None or getattr(operand, 'mask', None) is None:
            return deepcopy(self.mask)
        else:
            return handle_mask(self.mask, operand.mask, **kwds)

    def _arithmetic_wcs(self, operation, operand, compare_wcs, **kwds):
        if False:
            return 10
        '\n        Calculate the resulting wcs.\n\n        There is actually no calculation involved but it is a good place to\n        compare wcs information of both operands. This is currently not working\n        properly with `~astropy.wcs.WCS` (which is the suggested class for\n        storing as wcs property) but it will not break it neither.\n\n        Parameters\n        ----------\n        operation : callable\n            see :meth:`NDArithmeticMixin._arithmetic` parameter description.\n            By default, the ``operation`` will be ignored.\n\n        operand : `NDData` instance or subclass\n            The second operand wrapped in an instance of the same class as\n            self.\n\n        compare_wcs : callable\n            see :meth:`NDArithmeticMixin.add` parameter description.\n\n        kwds :\n            Additional parameters given to ``compare_wcs``.\n\n        Raises\n        ------\n        ValueError\n            If ``compare_wcs`` returns ``False``.\n\n        Returns\n        -------\n        result_wcs : any type\n            The ``wcs`` of the first operand is returned.\n        '
        if not compare_wcs(self.wcs, operand.wcs, **kwds):
            raise ValueError('WCS are not equal.')
        return deepcopy(self.wcs)

    def _arithmetic_meta(self, operation, operand, handle_meta, **kwds):
        if False:
            while True:
                i = 10
        '\n        Calculate the resulting meta.\n\n        Parameters\n        ----------\n        operation : callable\n            see :meth:`NDArithmeticMixin._arithmetic` parameter description.\n            By default, the ``operation`` will be ignored.\n\n        operand : `NDData`-like instance\n            The second operand wrapped in an instance of the same class as\n            self.\n\n        handle_meta : callable\n            see :meth:`NDArithmeticMixin.add`\n\n        kwds :\n            Additional parameters given to ``handle_meta``.\n\n        Returns\n        -------\n        result_meta : any type\n            The result of ``handle_meta``.\n        '
        return handle_meta(self.meta, operand.meta, **kwds)

    @sharedmethod
    @format_doc(_arit_doc, name='addition', op='+')
    def add(self, operand, operand2=None, **kwargs):
        if False:
            print('Hello World!')
        return self._prepare_then_do_arithmetic(np.add, operand, operand2, **kwargs)

    @sharedmethod
    @format_doc(_arit_doc, name='subtraction', op='-')
    def subtract(self, operand, operand2=None, **kwargs):
        if False:
            print('Hello World!')
        return self._prepare_then_do_arithmetic(np.subtract, operand, operand2, **kwargs)

    @sharedmethod
    @format_doc(_arit_doc, name='multiplication', op='*')
    def multiply(self, operand, operand2=None, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._prepare_then_do_arithmetic(np.multiply, operand, operand2, **kwargs)

    @sharedmethod
    @format_doc(_arit_doc, name='division', op='/')
    def divide(self, operand, operand2=None, **kwargs):
        if False:
            while True:
                i = 10
        return self._prepare_then_do_arithmetic(np.true_divide, operand, operand2, **kwargs)

    @sharedmethod
    def sum(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._prepare_then_do_arithmetic(np.sum, **kwargs)

    @sharedmethod
    def mean(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._prepare_then_do_arithmetic(np.mean, **kwargs)

    @sharedmethod
    def min(self, **kwargs):
        if False:
            print('Hello World!')
        propagate_uncertainties = kwargs.pop('propagate_uncertainties', None)
        return self._prepare_then_do_arithmetic(np.min, propagate_uncertainties=propagate_uncertainties, **kwargs)

    @sharedmethod
    def max(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        propagate_uncertainties = kwargs.pop('propagate_uncertainties', None)
        return self._prepare_then_do_arithmetic(np.max, propagate_uncertainties=propagate_uncertainties, **kwargs)

    @sharedmethod
    def _prepare_then_do_arithmetic(self_or_cls, operation, operand=None, operand2=None, **kwargs):
        if False:
            print('Hello World!')
        'Intermediate method called by public arithmetic (i.e. ``add``)\n        before the processing method (``_arithmetic``) is invoked.\n\n        .. warning::\n            Do not override this method in subclasses.\n\n        This method checks if it was called as instance or as class method and\n        then wraps the operands and the result from ``_arithmetic`` in the\n        appropriate subclass.\n\n        Parameters\n        ----------\n        self_or_cls : instance or class\n            ``sharedmethod`` behaves like a normal method if called on the\n            instance (then this parameter is ``self``) but like a classmethod\n            when called on the class (then this parameter is ``cls``).\n\n        operations : callable\n            The operation (normally a numpy-ufunc) that represents the\n            appropriate action.\n\n        operand, operand2, kwargs :\n            See for example ``add``.\n\n        Result\n        ------\n        result : `~astropy.nddata.NDData`-like\n            Depending how this method was called either ``self_or_cls``\n            (called on class) or ``self_or_cls.__class__`` (called on instance)\n            is the NDData-subclass that is used as wrapper for the result.\n        '
        if isinstance(self_or_cls, NDArithmeticMixin):
            cls = self_or_cls.__class__
            if operand2 is None:
                operand2 = operand
                operand = self_or_cls
            else:
                operand = cls(operand)
        else:
            cls = self_or_cls
            if operand2 is None:
                raise TypeError("operand2 must be given when the method isn't called on an instance.")
            operand = cls(operand)
        if operand2 is not None and (not issubclass(operand2.__class__, NDArithmeticMixin)):
            operand2 = cls(operand2)
            (result, init_kwds) = operand._arithmetic(operation, operand2, **kwargs)
        elif issubclass(operand2.__class__, NDArithmeticMixin):
            (result, init_kwds) = cls._arithmetic(operand, operation, operand2, **kwargs)
        else:
            (result, init_kwds) = self_or_cls._arithmetic(operation, operand2, **kwargs)
        return cls(result, **init_kwds)