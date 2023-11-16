from astropy import log
from astropy.wcs.wcsapi import BaseHighLevelWCS, BaseLowLevelWCS, HighLevelWCSWrapper, SlicedLowLevelWCS
__all__ = ['NDSlicingMixin']

class NDSlicingMixin:
    """Mixin to provide slicing on objects using the `NDData`
    interface.

    The ``data``, ``mask``, ``uncertainty`` and ``wcs`` will be sliced, if
    set and sliceable. The ``unit`` and ``meta`` will be untouched. The return
    will be a reference and not a copy, if possible.

    Examples
    --------
    Using this Mixin with `~astropy.nddata.NDData`:

        >>> from astropy.nddata import NDData, NDSlicingMixin
        >>> class NDDataSliceable(NDSlicingMixin, NDData):
        ...     pass

    Slicing an instance containing data::

        >>> nd = NDDataSliceable([1,2,3,4,5])
        >>> nd[1:3]
        NDDataSliceable([2, 3])

    Also the other attributes are sliced for example the ``mask``::

        >>> import numpy as np
        >>> mask = np.array([True, False, True, True, False])
        >>> nd2 = NDDataSliceable(nd, mask=mask)
        >>> nd2slc = nd2[1:3]
        >>> nd2slc[nd2slc.mask]
        NDDataSliceable([—])

    Be aware that changing values of the sliced instance will change the values
    of the original::

        >>> nd3 = nd2[1:3]
        >>> nd3.data[0] = 100
        >>> nd2
        NDDataSliceable([———, 100, ———, ———,   5])

    See Also
    --------
    NDDataRef
    NDDataArray
    """

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        if self.data.shape == ():
            raise TypeError('scalars cannot be sliced.')
        kwargs = self._slice(item)
        return self.__class__(**kwargs)

    def _slice(self, item):
        if False:
            while True:
                i = 10
        'Collects the sliced attributes and passes them back as `dict`.\n\n        It passes uncertainty, mask and wcs to their appropriate ``_slice_*``\n        method, while ``meta`` and ``unit`` are simply taken from the original.\n        The data is assumed to be sliceable and is sliced directly.\n\n        When possible the return should *not* be a copy of the data but a\n        reference.\n\n        Parameters\n        ----------\n        item : slice\n            The slice passed to ``__getitem__``.\n\n        Returns\n        -------\n        dict :\n            Containing all the attributes after slicing - ready to\n            use them to create ``self.__class__.__init__(**kwargs)`` in\n            ``__getitem__``.\n        '
        kwargs = {}
        kwargs['data'] = self.data[item]
        kwargs['uncertainty'] = self._slice_uncertainty(item)
        kwargs['mask'] = self._slice_mask(item)
        kwargs['wcs'] = self._slice_wcs(item)
        kwargs['unit'] = self.unit
        kwargs['meta'] = self.meta
        return kwargs

    def _slice_uncertainty(self, item):
        if False:
            return 10
        if self.uncertainty is None:
            return None
        try:
            return self.uncertainty[item]
        except (TypeError, KeyError):
            log.info('uncertainty cannot be sliced.')
        return self.uncertainty

    def _slice_mask(self, item):
        if False:
            print('Hello World!')
        if self.mask is None:
            return None
        try:
            return self.mask[item]
        except (TypeError, KeyError):
            log.info('mask cannot be sliced.')
        return self.mask

    def _slice_wcs(self, item):
        if False:
            for i in range(10):
                print('nop')
        if self.wcs is None:
            return None
        try:
            llwcs = SlicedLowLevelWCS(self.wcs.low_level_wcs, item)
            return HighLevelWCSWrapper(llwcs)
        except Exception as err:
            self._handle_wcs_slicing_error(err, item)

    def _handle_wcs_slicing_error(self, err, item):
        if False:
            print('Hello World!')
        raise ValueError(f"Slicing the WCS object with the slice '{item}' failed, if you want to slice the NDData object without the WCS, you can remove by setting `NDData.wcs = None` and then retry.") from err