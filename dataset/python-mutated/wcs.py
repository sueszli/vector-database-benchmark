import builtins
import copy
import io
import itertools
import os
import re
import textwrap
import uuid
import warnings
import numpy as np
from packaging.version import Version
from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyUserWarning, AstropyWarning
from . import _wcs, docstrings
from .wcsapi.fitswcs import FITSWCSAPIMixin, SlicedFITSWCS
__all__ = ['FITSFixedWarning', 'WCS', 'find_all_wcs', 'DistortionLookupTable', 'Sip', 'Tabprm', 'Wcsprm', 'Auxprm', 'Celprm', 'Prjprm', 'Wtbarr', 'WCSBase', 'validate', 'WcsError', 'SingularMatrixError', 'InconsistentAxisTypesError', 'InvalidTransformError', 'InvalidCoordinateError', 'InvalidPrjParametersError', 'NoSolutionError', 'InvalidSubimageSpecificationError', 'NoConvergence', 'NonseparableSubimageCoordinateSystemError', 'NoWcsKeywordsFoundError', 'InvalidTabularParametersError']
__doctest_skip__ = ['WCS.all_world2pix']
if _wcs is not None:
    if Version(_wcs.__version__) < Version('5.8'):
        raise ImportError('astropy.wcs is built with wcslib {0}, but only versions 5.8 and later on the 5.x series are known to work.  The version of wcslib that ships with astropy may be used.')
    if not _wcs._sanity_check():
        raise RuntimeError('astropy.wcs did not pass its sanity check for your build on your platform.')
    _WCSSUB_TIME_SUPPORT = Version(_wcs.__version__) >= Version('7.8')
    _WCS_TPD_WARN_LT71 = Version(_wcs.__version__) < Version('7.1')
    _WCS_TPD_WARN_LT74 = Version(_wcs.__version__) < Version('7.4')
    WCSBase = _wcs._Wcs
    DistortionLookupTable = _wcs.DistortionLookupTable
    Sip = _wcs.Sip
    Wcsprm = _wcs.Wcsprm
    Auxprm = _wcs.Auxprm
    Celprm = _wcs.Celprm
    Prjprm = _wcs.Prjprm
    Tabprm = _wcs.Tabprm
    Wtbarr = _wcs.Wtbarr
    WcsError = _wcs.WcsError
    SingularMatrixError = _wcs.SingularMatrixError
    InconsistentAxisTypesError = _wcs.InconsistentAxisTypesError
    InvalidTransformError = _wcs.InvalidTransformError
    InvalidCoordinateError = _wcs.InvalidCoordinateError
    NoSolutionError = _wcs.NoSolutionError
    InvalidSubimageSpecificationError = _wcs.InvalidSubimageSpecificationError
    NonseparableSubimageCoordinateSystemError = _wcs.NonseparableSubimageCoordinateSystemError
    NoWcsKeywordsFoundError = _wcs.NoWcsKeywordsFoundError
    InvalidTabularParametersError = _wcs.InvalidTabularParametersError
    InvalidPrjParametersError = _wcs.InvalidPrjParametersError
    for (key, val) in _wcs.__dict__.items():
        if key.startswith(('WCSSUB_', 'WCSHDR_', 'WCSHDO_', 'WCSCOMPARE_', 'PRJ_')):
            locals()[key] = val
            __all__.append(key)

    def _load_tab_bintable(hdulist, extnam, extver, extlev, kind, ttype, row, ndim):
        if False:
            i = 10
            return i + 15
        arr = hdulist[extnam, extver].data[ttype][row - 1]
        if arr.ndim != ndim:
            if kind == 'c' and ndim == 2:
                arr = arr.reshape((arr.size, 1))
            else:
                raise ValueError('Bad TDIM')
        return np.ascontiguousarray(arr, dtype=np.double)
    _wcs.set_wtbarr_fitsio_callback(_load_tab_bintable)
else:
    WCSBase = object
    Wcsprm = object
    DistortionLookupTable = object
    Sip = object
    Tabprm = object
    Wtbarr = object
    WcsError = None
    SingularMatrixError = None
    InconsistentAxisTypesError = None
    InvalidTransformError = None
    InvalidCoordinateError = None
    NoSolutionError = None
    InvalidSubimageSpecificationError = None
    NonseparableSubimageCoordinateSystemError = None
    NoWcsKeywordsFoundError = None
    InvalidTabularParametersError = None
    _WCSSUB_TIME_SUPPORT = False
    _WCS_TPD_WARN_LT71 = False
    _WCS_TPD_WARN_LT74 = False
WCSHDO_SIP = 524288
SIP_KW = re.compile('^[AB]P?_1?[0-9]_1?[0-9][A-Z]?$')

def _parse_keysel(keysel):
    if False:
        return 10
    keysel_flags = 0
    if keysel is not None:
        for element in keysel:
            if element.lower() == 'image':
                keysel_flags |= _wcs.WCSHDR_IMGHEAD
            elif element.lower() == 'binary':
                keysel_flags |= _wcs.WCSHDR_BIMGARR
            elif element.lower() == 'pixel':
                keysel_flags |= _wcs.WCSHDR_PIXLIST
            else:
                raise ValueError("keysel must be a list of 'image', 'binary' and/or 'pixel'")
    else:
        keysel_flags = -1
    return keysel_flags

class NoConvergence(Exception):
    """
    An error class used to report non-convergence and/or divergence
    of numerical methods. It is used to report errors in the
    iterative solution used by
    the :py:meth:`~astropy.wcs.WCS.all_world2pix`.

    Attributes
    ----------
    best_solution : `numpy.ndarray`
        Best solution achieved by the numerical method.

    accuracy : `numpy.ndarray`
        Accuracy of the ``best_solution``.

    niter : `int`
        Number of iterations performed by the numerical method
        to compute ``best_solution``.

    divergent : None, `numpy.ndarray`
        Indices of the points in ``best_solution`` array
        for which the solution appears to be divergent. If the
        solution does not diverge, ``divergent`` will be set to `None`.

    slow_conv : None, `numpy.ndarray`
        Indices of the solutions in ``best_solution`` array
        for which the solution failed to converge within the
        specified maximum number of iterations. If there are no
        non-converging solutions (i.e., if the required accuracy
        has been achieved for all input data points)
        then ``slow_conv`` will be set to `None`.

    """

    def __init__(self, *args, best_solution=None, accuracy=None, niter=None, divergent=None, slow_conv=None):
        if False:
            while True:
                i = 10
        super().__init__(*args)
        self.best_solution = best_solution
        self.accuracy = accuracy
        self.niter = niter
        self.divergent = divergent
        self.slow_conv = slow_conv

class FITSFixedWarning(AstropyWarning):
    """
    The warning raised when the contents of the FITS header have been
    modified to be standards compliant.
    """
    pass

class WCS(FITSWCSAPIMixin, WCSBase):
    """WCS objects perform standard WCS transformations, and correct for
    `SIP`_ and `distortion paper`_ table-lookup transformations, based
    on the WCS keywords and supplementary data read from a FITS file.

    See also: https://docs.astropy.org/en/stable/wcs/

    Parameters
    ----------
    header : `~astropy.io.fits.Header`, `~astropy.io.fits.hdu.image.PrimaryHDU`, `~astropy.io.fits.hdu.image.ImageHDU`, str, dict-like, or None, optional
        If *header* is not provided or None, the object will be
        initialized to default values.

    fobj : `~astropy.io.fits.HDUList`, optional
        It is needed when header keywords point to a `distortion
        paper`_ lookup table stored in a different extension.

    key : str, optional
        The name of a particular WCS transform to use.  This may be
        either ``' '`` or ``'A'``-``'Z'`` and corresponds to the
        ``"a"`` part of the ``CTYPEia`` cards.  *key* may only be
        provided if *header* is also provided.

    minerr : float, optional
        The minimum value a distortion correction must have in order
        to be applied. If the value of ``CQERRja`` is smaller than
        *minerr*, the corresponding distortion is not applied.

    relax : bool or int, optional
        Degree of permissiveness:

        - `True` (default): Admit all recognized informal extensions
          of the WCS standard.

        - `False`: Recognize only FITS keywords defined by the
          published WCS standard.

        - `int`: a bit field selecting specific extensions to accept.
          See :ref:`astropy:relaxread` for details.

    naxis : int or sequence, optional
        Extracts specific coordinate axes using
        :meth:`~astropy.wcs.Wcsprm.sub`.  If a header is provided, and
        *naxis* is not ``None``, *naxis* will be passed to
        :meth:`~astropy.wcs.Wcsprm.sub` in order to select specific
        axes from the header.  See :meth:`~astropy.wcs.Wcsprm.sub` for
        more details about this parameter.

    keysel : sequence of str, optional
        A sequence of flags used to select the keyword types
        considered by wcslib.  When ``None``, only the standard image
        header keywords are considered (and the underlying wcspih() C
        function is called).  To use binary table image array or pixel
        list keywords, *keysel* must be set.

        Each element in the list should be one of the following
        strings:

        - 'image': Image header keywords

        - 'binary': Binary table image array keywords

        - 'pixel': Pixel list keywords

        Keywords such as ``EQUIna`` or ``RFRQna`` that are common to
        binary table image arrays and pixel lists (including
        ``WCSNna`` and ``TWCSna``) are selected by both 'binary' and
        'pixel'.

    colsel : sequence of int, optional
        A sequence of table column numbers used to restrict the WCS
        transformations considered to only those pertaining to the
        specified columns.  If `None`, there is no restriction.

    fix : bool, optional
        When `True` (default), call `~astropy.wcs.Wcsprm.fix` on
        the resulting object to fix any non-standard uses in the
        header.  `FITSFixedWarning` Warnings will be emitted if any
        changes were made.

    translate_units : str, optional
        Specify which potentially unsafe translations of non-standard
        unit strings to perform.  By default, performs none.  See
        `WCS.fix` for more information about this parameter.  Only
        effective when ``fix`` is `True`.

    Raises
    ------
    MemoryError
         Memory allocation failed.

    ValueError
         Invalid key.

    KeyError
         Key not found in FITS header.

    ValueError
         Lookup table distortion present in the header but *fobj* was
         not provided.

    Notes
    -----
    1. astropy.wcs supports arbitrary *n* dimensions for the core WCS
       (the transformations handled by WCSLIB).  However, the
       `distortion paper`_ lookup table and `SIP`_ distortions must be
       two dimensional.  Therefore, if you try to create a WCS object
       where the core WCS has a different number of dimensions than 2
       and that object also contains a `distortion paper`_ lookup
       table or `SIP`_ distortion, a `ValueError`
       exception will be raised.  To avoid this, consider using the
       *naxis* kwarg to select two dimensions from the core WCS.

    2. The number of coordinate axes in the transformation is not
       determined directly from the ``NAXIS`` keyword but instead from
       the highest of:

           - ``NAXIS`` keyword

           - ``WCSAXESa`` keyword

           - The highest axis number in any parameterized WCS keyword.
             The keyvalue, as well as the keyword, must be
             syntactically valid otherwise it will not be considered.

       If none of these keyword types is present, i.e. if the header
       only contains auxiliary WCS keywords for a particular
       coordinate representation, then no coordinate description is
       constructed for it.

       The number of axes, which is set as the ``naxis`` member, may
       differ for different coordinate representations of the same
       image.

    3. When the header includes duplicate keywords, in most cases the
       last encountered is used.

    4. `~astropy.wcs.Wcsprm.set` is called immediately after
       construction, so any invalid keywords or transformations will
       be raised by the constructor, not when subsequently calling a
       transformation method.

    """

    def __init__(self, header=None, fobj=None, key=' ', minerr=0.0, relax=True, naxis=None, keysel=None, colsel=None, fix=True, translate_units='', _do_set=True):
        if False:
            while True:
                i = 10
        close_fds = []
        self._init_kwargs = {'keysel': copy.copy(keysel), 'colsel': copy.copy(colsel)}
        if header is None:
            if naxis is None:
                naxis = 2
            wcsprm = _wcs.Wcsprm(header=None, key=key, relax=relax, naxis=naxis)
            self.naxis = wcsprm.naxis
            det2im = (None, None)
            cpdis = (None, None)
            sip = None
        else:
            keysel_flags = _parse_keysel(keysel)
            if isinstance(header, (str, bytes)):
                try:
                    is_path = os.path.exists(header)
                except (OSError, ValueError):
                    is_path = False
                if is_path:
                    if fobj is not None:
                        raise ValueError('Can not provide both a FITS filename to argument 1 and a FITS file object to argument 2')
                    fobj = fits.open(header)
                    close_fds.append(fobj)
                    header = fobj[0].header
            elif isinstance(header, fits.hdu.image._ImageBaseHDU):
                header = header.header
            elif not isinstance(header, fits.Header):
                try:
                    orig_header = header
                    header = fits.Header()
                    for dict_key in orig_header.keys():
                        header[dict_key] = orig_header[dict_key]
                except TypeError:
                    raise TypeError('header must be a string, an astropy.io.fits.Header object, or a dict-like object')
            if isinstance(header, fits.Header):
                header_string = header.tostring().rstrip()
            else:
                header_string = header
            if isinstance(header_string, str):
                header_bytes = header_string.encode('ascii')
            else:
                header_bytes = header_string
                header_string = header_string.decode('ascii')
            if not (fobj is None or isinstance(fobj, fits.HDUList)):
                raise AssertionError("'fobj' must be either None or an astropy.io.fits.HDUList object.")
            est_naxis = 2
            try:
                tmp_header = fits.Header.fromstring(header_string)
                self._remove_sip_kw(tmp_header)
                tmp_header_bytes = tmp_header.tostring().rstrip()
                if isinstance(tmp_header_bytes, str):
                    tmp_header_bytes = tmp_header_bytes.encode('ascii')
                tmp_wcsprm = _wcs.Wcsprm(header=tmp_header_bytes, key=key, relax=relax, keysel=keysel_flags, colsel=colsel, warnings=False, hdulist=fobj)
                if naxis is not None:
                    try:
                        tmp_wcsprm = tmp_wcsprm.sub(naxis)
                    except ValueError:
                        pass
                    est_naxis = tmp_wcsprm.naxis if tmp_wcsprm.naxis else 2
            except _wcs.NoWcsKeywordsFoundError:
                pass
            self.naxis = est_naxis
            header = fits.Header.fromstring(header_string)
            det2im = self._read_det2im_kw(header, fobj, err=minerr)
            cpdis = self._read_distortion_kw(header, fobj, dist='CPDIS', err=minerr)
            self._fix_pre2012_scamp_tpv(header)
            sip = self._read_sip_kw(header, wcskey=key)
            self._remove_sip_kw(header)
            header_string = header.tostring()
            header_string = header_string.replace('END' + ' ' * 77, '')
            if isinstance(header_string, str):
                header_bytes = header_string.encode('ascii')
            else:
                header_bytes = header_string
                header_string = header_string.decode('ascii')
            try:
                wcsprm = _wcs.Wcsprm(header=header_bytes, key=key, relax=relax, keysel=keysel_flags, colsel=colsel, hdulist=fobj)
            except _wcs.NoWcsKeywordsFoundError:
                if colsel is None:
                    wcsprm = _wcs.Wcsprm(header=None, key=key, relax=relax, keysel=keysel_flags, colsel=colsel, hdulist=fobj)
                else:
                    raise
            if naxis is not None:
                wcsprm = wcsprm.sub(naxis)
            self.naxis = wcsprm.naxis
            if wcsprm.naxis != 2 and (det2im[0] or det2im[1] or cpdis[0] or cpdis[1] or sip):
                raise ValueError(f'\nFITS WCS distortion paper lookup tables and SIP distortions only work\nin 2 dimensions.  However, WCSLIB has detected {wcsprm.naxis} dimensions in the\ncore WCS keywords.  To use core WCS in conjunction with FITS WCS\ndistortion paper lookup tables or SIP distortion, you must select or\nreduce these to 2 dimensions using the naxis kwarg.\n')
            header_naxis = header.get('NAXIS', None)
            if header_naxis is not None and header_naxis < wcsprm.naxis:
                warnings.warn(f'The WCS transformation has more axes ({wcsprm.naxis:d}) than the image it is associated with ({header_naxis:d})', FITSFixedWarning)
        self._get_naxis(header)
        WCSBase.__init__(self, sip, cpdis, wcsprm, det2im)
        if fix:
            if header is None:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FITSFixedWarning)
                    self.fix(translate_units=translate_units)
            else:
                self.fix(translate_units=translate_units)
        if _do_set:
            self.wcs.set()
        for fd in close_fds:
            fd.close()
        self._pixel_bounds = None

    def __copy__(self):
        if False:
            for i in range(10):
                print('nop')
        new_copy = self.__class__()
        WCSBase.__init__(new_copy, self.sip, (self.cpdis1, self.cpdis2), self.wcs, (self.det2im1, self.det2im2))
        new_copy.__dict__.update(self.__dict__)
        return new_copy

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        from copy import deepcopy
        new_copy = self.__class__()
        new_copy.naxis = deepcopy(self.naxis, memo)
        WCSBase.__init__(new_copy, deepcopy(self.sip, memo), (deepcopy(self.cpdis1, memo), deepcopy(self.cpdis2, memo)), deepcopy(self.wcs, memo), (deepcopy(self.det2im1, memo), deepcopy(self.det2im2, memo)))
        for (key, val) in self.__dict__.items():
            new_copy.__dict__[key] = deepcopy(val, memo)
        return new_copy

    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a shallow copy of the object.\n\n        Convenience method so user doesn't have to import the\n        :mod:`copy` stdlib module.\n\n        .. warning::\n            Use `deepcopy` instead of `copy` unless you know why you need a\n            shallow copy.\n        "
        return copy.copy(self)

    def deepcopy(self):
        if False:
            while True:
                i = 10
        "\n        Return a deep copy of the object.\n\n        Convenience method so user doesn't have to import the\n        :mod:`copy` stdlib module.\n        "
        return copy.deepcopy(self)

    def sub(self, axes=None):
        if False:
            print('Hello World!')
        copy = self.deepcopy()
        cname_uuid = [str(uuid.uuid4()) for i in range(copy.wcs.naxis)]
        copy.wcs.cname = cname_uuid
        copy.wcs = copy.wcs.sub(axes)
        copy.naxis = copy.wcs.naxis
        keep = [cname_uuid.index(cname) if cname in cname_uuid else None for cname in copy.wcs.cname]
        copy.wcs.cname = ['' if i is None else self.wcs.cname[i] for i in keep]
        if self.pixel_shape:
            copy.pixel_shape = tuple((None if i is None else self.pixel_shape[i] for i in keep))
        if self.pixel_bounds:
            copy.pixel_bounds = [None if i is None else self.pixel_bounds[i] for i in keep]
        return copy
    if _wcs is not None:
        sub.__doc__ = _wcs.Wcsprm.sub.__doc__

    def _fix_scamp(self):
        if False:
            while True:
                i = 10
        "\n        Remove SCAMP's PVi_m distortion parameters if SIP distortion parameters\n        are also present. Some projects (e.g., Palomar Transient Factory)\n        convert SCAMP's distortion parameters (which abuse the PVi_m cards) to\n        SIP. However, wcslib gets confused by the presence of both SCAMP and\n        SIP distortion parameters.\n\n        See https://github.com/astropy/astropy/issues/299.\n\n        SCAMP uses TAN projection exclusively. The case of CTYPE ending\n        in -TAN should have been handled by ``_fix_pre2012_scamp_tpv()`` before\n        calling this function.\n        "
        if self.wcs is None:
            return
        ctype = [ct.strip().upper() for ct in self.wcs.ctype]
        if sum((ct.endswith('-TPV') for ct in ctype)) == 2:
            if self.sip is not None:
                self.sip = None
                warnings.warn('Removed redundant SIP distortion parameters because CTYPE explicitly specifies TPV distortions', FITSFixedWarning)
            return
        pv = self.wcs.get_pv()
        if not pv:
            return
        if self.sip is None:
            return
        has_scamp = False
        for i in {v[0] for v in pv}:
            js = tuple((v[1] for v in pv if v[0] == i))
            if '-TAN' in self.wcs.ctype[i - 1].upper() and js and (max(js) >= 5):
                has_scamp = True
                break
        if has_scamp and all((ct.endswith('-SIP') for ct in ctype)):
            self.wcs.set_pv([])
            warnings.warn('Removed redundant SCAMP distortion parameters because SIP parameters are also present', FITSFixedWarning)
            return

    def fix(self, translate_units='', naxis=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform the fix operations from wcslib, and warn about any\n        changes it has made.\n\n        Parameters\n        ----------\n        translate_units : str, optional\n            Specify which potentially unsafe translations of\n            non-standard unit strings to perform.  By default,\n            performs none.\n\n            Although ``"S"`` is commonly used to represent seconds,\n            its translation to ``"s"`` is potentially unsafe since the\n            standard recognizes ``"S"`` formally as Siemens, however\n            rarely that may be used.  The same applies to ``"H"`` for\n            hours (Henry), and ``"D"`` for days (Debye).\n\n            This string controls what to do in such cases, and is\n            case-insensitive.\n\n            - If the string contains ``"s"``, translate ``"S"`` to\n              ``"s"``.\n\n            - If the string contains ``"h"``, translate ``"H"`` to\n              ``"h"``.\n\n            - If the string contains ``"d"``, translate ``"D"`` to\n              ``"d"``.\n\n            Thus ``\'\'`` doesn\'t do any unsafe translations, whereas\n            ``\'shd\'`` does all of them.\n\n        naxis : int array, optional\n            Image axis lengths.  If this array is set to zero or\n            ``None``, then `~astropy.wcs.Wcsprm.cylfix` will not be\n            invoked.\n        '
        if self.wcs is not None:
            self._fix_scamp()
            fixes = self.wcs.fix(translate_units, naxis)
            for (key, val) in fixes.items():
                if val != 'No change':
                    if key == 'datfix' and '1858-11-17' in val and (not np.count_nonzero(self.wcs.mjdref)):
                        continue
                    warnings.warn(f"'{key}' made the change '{val}'.", FITSFixedWarning)

    def calc_footprint(self, header=None, undistort=True, axes=None, center=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculates the footprint of the image on the sky.\n\n        A footprint is defined as the positions of the corners of the\n        image on the sky after all available distortions have been\n        applied.\n\n        Parameters\n        ----------\n        header : `~astropy.io.fits.Header` object, optional\n            Used to get ``NAXIS1`` and ``NAXIS2``\n            header and axes are mutually exclusive, alternative ways\n            to provide the same information.\n\n        undistort : bool, optional\n            If `True`, take SIP and distortion lookup table into\n            account\n\n        axes : (int, int), optional\n            If provided, use the given sequence as the shape of the\n            image.  Otherwise, use the ``NAXIS1`` and ``NAXIS2``\n            keywords from the header that was used to create this\n            `WCS` object.\n\n        center : bool, optional\n            If `True` use the center of the pixel, otherwise use the corner.\n\n        Returns\n        -------\n        coord : (4, 2) array of (*x*, *y*) coordinates.\n            The order is clockwise starting with the bottom left corner.\n        '
        if axes is not None:
            (naxis1, naxis2) = axes
        elif header is None:
            try:
                (naxis1, naxis2) = self.pixel_shape
            except (AttributeError, TypeError):
                warnings.warn('Need a valid header in order to calculate footprint\n', AstropyUserWarning)
                return None
        else:
            naxis1 = header.get('NAXIS1', None)
            naxis2 = header.get('NAXIS2', None)
        if naxis1 is None or naxis2 is None:
            raise ValueError('Image size could not be determined.')
        if center:
            corners = np.array([[1, 1], [1, naxis2], [naxis1, naxis2], [naxis1, 1]], dtype=np.float64)
        else:
            corners = np.array([[0.5, 0.5], [0.5, naxis2 + 0.5], [naxis1 + 0.5, naxis2 + 0.5], [naxis1 + 0.5, 0.5]], dtype=np.float64)
        if undistort:
            return self.all_pix2world(corners, 1)
        else:
            return self.wcs_pix2world(corners, 1)

    def _read_det2im_kw(self, header, fobj, err=0.0):
        if False:
            print('Hello World!')
        '\n        Create a `distortion paper`_ type lookup table for detector to\n        image plane correction.\n        '
        if fobj is None:
            return (None, None)
        if not isinstance(fobj, fits.HDUList):
            return (None, None)
        try:
            axiscorr = header['AXISCORR']
            d2imdis = self._read_d2im_old_format(header, fobj, axiscorr)
            return d2imdis
        except KeyError:
            pass
        dist = 'D2IMDIS'
        d_kw = 'D2IM'
        err_kw = 'D2IMERR'
        tables = {}
        for i in range(1, self.naxis + 1):
            d_error = header.get(err_kw + str(i), 0.0)
            if d_error < err:
                tables[i] = None
                continue
            distortion = dist + str(i)
            if distortion in header:
                dis = header[distortion].lower()
                if dis == 'lookup':
                    del header[distortion]
                    assert isinstance(fobj, fits.HDUList), 'An astropy.io.fits.HDUListis required for Lookup table distortion.'
                    dp = (d_kw + str(i)).strip()
                    dp_extver_key = dp + '.EXTVER'
                    if dp_extver_key in header:
                        d_extver = header[dp_extver_key]
                        del header[dp_extver_key]
                    else:
                        d_extver = 1
                    dp_axis_key = dp + f'.AXIS.{i:d}'
                    if i == header[dp_axis_key]:
                        d_data = fobj['D2IMARR', d_extver].data
                    else:
                        d_data = fobj['D2IMARR', d_extver].data.transpose()
                    del header[dp_axis_key]
                    d_header = fobj['D2IMARR', d_extver].header
                    d_crpix = (d_header.get('CRPIX1', 0.0), d_header.get('CRPIX2', 0.0))
                    d_crval = (d_header.get('CRVAL1', 0.0), d_header.get('CRVAL2', 0.0))
                    d_cdelt = (d_header.get('CDELT1', 1.0), d_header.get('CDELT2', 1.0))
                    d_lookup = DistortionLookupTable(d_data, d_crpix, d_crval, d_cdelt)
                    tables[i] = d_lookup
                else:
                    warnings.warn('Polynomial distortion is not implemented.\n', AstropyUserWarning)
                for key in set(header):
                    if key.startswith(dp + '.'):
                        del header[key]
            else:
                tables[i] = None
        if not tables:
            return (None, None)
        else:
            return (tables.get(1), tables.get(2))

    def _read_d2im_old_format(self, header, fobj, axiscorr):
        if False:
            return 10
        warnings.warn('The use of ``AXISCORR`` for D2IM correction has been deprecated.`~astropy.wcs` will read in files with ``AXISCORR`` but ``to_fits()`` will write out files without it.', AstropyDeprecationWarning)
        cpdis = [None, None]
        crpix = [0.0, 0.0]
        crval = [0.0, 0.0]
        cdelt = [1.0, 1.0]
        try:
            d2im_data = fobj['D2IMARR', 1].data
        except KeyError:
            return (None, None)
        except AttributeError:
            return (None, None)
        d2im_data = np.array([d2im_data])
        d2im_hdr = fobj['D2IMARR', 1].header
        naxis = d2im_hdr['NAXIS']
        for i in range(1, naxis + 1):
            crpix[i - 1] = d2im_hdr.get('CRPIX' + str(i), 0.0)
            crval[i - 1] = d2im_hdr.get('CRVAL' + str(i), 0.0)
            cdelt[i - 1] = d2im_hdr.get('CDELT' + str(i), 1.0)
        cpdis = DistortionLookupTable(d2im_data, crpix, crval, cdelt)
        if axiscorr == 1:
            return (cpdis, None)
        elif axiscorr == 2:
            return (None, cpdis)
        else:
            warnings.warn('Expected AXISCORR to be 1 or 2', AstropyUserWarning)
            return (None, None)

    def _write_det2im(self, hdulist):
        if False:
            return 10
        '\n        Writes a `distortion paper`_ type lookup table to the given\n        `~astropy.io.fits.HDUList`.\n        '
        if self.det2im1 is None and self.det2im2 is None:
            return
        dist = 'D2IMDIS'
        d_kw = 'D2IM'

        def write_d2i(num, det2im):
            if False:
                return 10
            if det2im is None:
                return
            hdulist[0].header[f'{dist}{num:d}'] = ('LOOKUP', 'Detector to image correction type')
            hdulist[0].header[f'{d_kw}{num:d}.EXTVER'] = (num, 'Version number of WCSDVARR extension')
            hdulist[0].header[f'{d_kw}{num:d}.NAXES'] = (len(det2im.data.shape), 'Number of independent variables in D2IM function')
            for i in range(det2im.data.ndim):
                jth = {1: '1st', 2: '2nd', 3: '3rd'}.get(i + 1, f'{i + 1}th')
                hdulist[0].header[f'{d_kw}{num:d}.AXIS.{i + 1:d}'] = (i + 1, f'Axis number of the {jth} variable in a D2IM function')
            image = fits.ImageHDU(det2im.data, name='D2IMARR')
            header = image.header
            header['CRPIX1'] = (det2im.crpix[0], 'Coordinate system reference pixel')
            header['CRPIX2'] = (det2im.crpix[1], 'Coordinate system reference pixel')
            header['CRVAL1'] = (det2im.crval[0], 'Coordinate system value at reference pixel')
            header['CRVAL2'] = (det2im.crval[1], 'Coordinate system value at reference pixel')
            header['CDELT1'] = (det2im.cdelt[0], 'Coordinate increment along axis')
            header['CDELT2'] = (det2im.cdelt[1], 'Coordinate increment along axis')
            image.ver = int(hdulist[0].header[f'{d_kw}{num:d}.EXTVER'])
            hdulist.append(image)
        write_d2i(1, self.det2im1)
        write_d2i(2, self.det2im2)

    def _read_distortion_kw(self, header, fobj, dist='CPDIS', err=0.0):
        if False:
            while True:
                i = 10
        '\n        Reads `distortion paper`_ table-lookup keywords and data, and\n        returns a 2-tuple of `~astropy.wcs.DistortionLookupTable`\n        objects.\n\n        If no `distortion paper`_ keywords are found, ``(None, None)``\n        is returned.\n        '
        if isinstance(header, (str, bytes)):
            return (None, None)
        if dist == 'CPDIS':
            d_kw = 'DP'
            err_kw = 'CPERR'
        else:
            d_kw = 'DQ'
            err_kw = 'CQERR'
        tables = {}
        for i in range(1, self.naxis + 1):
            d_error_key = err_kw + str(i)
            if d_error_key in header:
                d_error = header[d_error_key]
                del header[d_error_key]
            else:
                d_error = 0.0
            if d_error < err:
                tables[i] = None
                continue
            distortion = dist + str(i)
            if distortion in header:
                dis = header[distortion].lower()
                del header[distortion]
                if dis == 'lookup':
                    if not isinstance(fobj, fits.HDUList):
                        raise ValueError('an astropy.io.fits.HDUList is required for Lookup table distortion.')
                    dp = (d_kw + str(i)).strip()
                    dp_extver_key = dp + '.EXTVER'
                    if dp_extver_key in header:
                        d_extver = header[dp_extver_key]
                        del header[dp_extver_key]
                    else:
                        d_extver = 1
                    dp_axis_key = dp + f'.AXIS.{i:d}'
                    if i == header[dp_axis_key]:
                        d_data = fobj['WCSDVARR', d_extver].data
                    else:
                        d_data = fobj['WCSDVARR', d_extver].data.transpose()
                    del header[dp_axis_key]
                    d_header = fobj['WCSDVARR', d_extver].header
                    d_crpix = (d_header.get('CRPIX1', 0.0), d_header.get('CRPIX2', 0.0))
                    d_crval = (d_header.get('CRVAL1', 0.0), d_header.get('CRVAL2', 0.0))
                    d_cdelt = (d_header.get('CDELT1', 1.0), d_header.get('CDELT2', 1.0))
                    d_lookup = DistortionLookupTable(d_data, d_crpix, d_crval, d_cdelt)
                    tables[i] = d_lookup
                    for key in set(header):
                        if key.startswith(dp + '.'):
                            del header[key]
                else:
                    warnings.warn('Polynomial distortion is not implemented.\n', AstropyUserWarning)
            else:
                tables[i] = None
        if not tables:
            return (None, None)
        else:
            return (tables.get(1), tables.get(2))

    def _write_distortion_kw(self, hdulist, dist='CPDIS'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write out `distortion paper`_ keywords to the given\n        `~astropy.io.fits.HDUList`.\n        '
        if self.cpdis1 is None and self.cpdis2 is None:
            return
        if dist == 'CPDIS':
            d_kw = 'DP'
        else:
            d_kw = 'DQ'

        def write_dist(num, cpdis):
            if False:
                print('Hello World!')
            if cpdis is None:
                return
            hdulist[0].header[f'{dist}{num:d}'] = ('LOOKUP', 'Prior distortion function type')
            hdulist[0].header[f'{d_kw}{num:d}.EXTVER'] = (num, 'Version number of WCSDVARR extension')
            hdulist[0].header[f'{d_kw}{num:d}.NAXES'] = (len(cpdis.data.shape), f'Number of independent variables in {dist} function')
            for i in range(cpdis.data.ndim):
                jth = {1: '1st', 2: '2nd', 3: '3rd'}.get(i + 1, f'{i + 1}th')
                hdulist[0].header[f'{d_kw}{num:d}.AXIS.{i + 1:d}'] = (i + 1, f'Axis number of the {jth} variable in a {dist} function')
            image = fits.ImageHDU(cpdis.data, name='WCSDVARR')
            header = image.header
            header['CRPIX1'] = (cpdis.crpix[0], 'Coordinate system reference pixel')
            header['CRPIX2'] = (cpdis.crpix[1], 'Coordinate system reference pixel')
            header['CRVAL1'] = (cpdis.crval[0], 'Coordinate system value at reference pixel')
            header['CRVAL2'] = (cpdis.crval[1], 'Coordinate system value at reference pixel')
            header['CDELT1'] = (cpdis.cdelt[0], 'Coordinate increment along axis')
            header['CDELT2'] = (cpdis.cdelt[1], 'Coordinate increment along axis')
            image.ver = int(hdulist[0].header[f'{d_kw}{num:d}.EXTVER'])
            hdulist.append(image)
        write_dist(1, self.cpdis1)
        write_dist(2, self.cpdis2)

    def _fix_pre2012_scamp_tpv(self, header, wcskey=''):
        if False:
            for i in range(10):
                print('nop')
        "\n        Replace -TAN with TPV (for pre-2012 SCAMP headers that use -TAN\n        in CTYPE). Ignore SIP if present. This follows recommendations in\n        Section 7 in\n        http://web.ipac.caltech.edu/staff/shupe/reprints/SIP_to_PV_SPIE2012.pdf.\n\n        This is to deal with pre-2012 headers that may contain TPV with a\n        CTYPE that ends in '-TAN' (post-2012 they should end in '-TPV' when\n        SCAMP has adopted the new TPV convention).\n        "
        if isinstance(header, (str, bytes)):
            return
        wcskey = wcskey.strip().upper()
        cntype = [(nax, header.get(f'CTYPE{nax}{wcskey}', '').strip()) for nax in range(1, self.naxis + 1)]
        tan_axes = [ct[0] for ct in cntype if ct[1].endswith('-TAN')]
        if len(tan_axes) == 2:
            tan_to_tpv = False
            for nax in tan_axes:
                js = []
                for p in header[f'PV{nax}_*{wcskey}'].keys():
                    prefix = f'PV{nax}_'
                    if p.startswith(prefix):
                        p = p[len(prefix):]
                        p = p.rstrip(wcskey)
                        try:
                            p = int(p)
                        except ValueError:
                            continue
                        js.append(p)
                if js and max(js) >= 5:
                    tan_to_tpv = True
                    break
            if tan_to_tpv:
                warnings.warn("Removed redundant SIP distortion parameters because SCAMP' PV distortions are also present", FITSFixedWarning)
                self._remove_sip_kw(header, del_order=True)
                for i in tan_axes:
                    kwd = f'CTYPE{i:d}{wcskey}'
                    if kwd in header:
                        header[kwd] = header[kwd].strip().upper().replace('-TAN', '-TPV')

    @staticmethod
    def _remove_sip_kw(header, del_order=False):
        if False:
            print('Hello World!')
        '\n        Remove SIP information from a header.\n        '
        for key in {m.group() for m in map(SIP_KW.match, list(header)) if m is not None}:
            del header[key]
        if del_order:
            for kwd in ['A_ORDER', 'B_ORDER', 'AP_ORDER', 'BP_ORDER']:
                if kwd in header:
                    del header[kwd]

    def _read_sip_kw(self, header, wcskey=''):
        if False:
            print('Hello World!')
        '\n        Reads `SIP`_ header keywords and returns a `~astropy.wcs.Sip`\n        object.\n\n        If no `SIP`_ header keywords are found, ``None`` is returned.\n        '
        if isinstance(header, (str, bytes)):
            return None
        if 'A_ORDER' in header and header['A_ORDER'] > 1:
            if 'B_ORDER' not in header:
                raise ValueError('A_ORDER provided without corresponding B_ORDER keyword for SIP distortion')
            m = int(header['A_ORDER'])
            a = np.zeros((m + 1, m + 1), np.double)
            for i in range(m + 1):
                for j in range(m - i + 1):
                    key = f'A_{i}_{j}'
                    if key in header:
                        a[i, j] = header[key]
                        del header[key]
            m = int(header['B_ORDER'])
            if m > 1:
                b = np.zeros((m + 1, m + 1), np.double)
                for i in range(m + 1):
                    for j in range(m - i + 1):
                        key = f'B_{i}_{j}'
                        if key in header:
                            b[i, j] = header[key]
                            del header[key]
            else:
                a = None
                b = None
            del header['A_ORDER']
            del header['B_ORDER']
            ctype = [header[f'CTYPE{nax}{wcskey}'] for nax in range(1, self.naxis + 1)]
            if any((not ctyp.endswith('-SIP') for ctyp in ctype)):
                message = '\n                Inconsistent SIP distortion information is present in the FITS header and the WCS object:\n                SIP coefficients were detected, but CTYPE is missing a "-SIP" suffix.\n                astropy.wcs is using the SIP distortion coefficients,\n                therefore the coordinates calculated here might be incorrect.\n\n                If you do not want to apply the SIP distortion coefficients,\n                please remove the SIP coefficients from the FITS header or the\n                WCS object.  As an example, if the image is already distortion-corrected\n                (e.g., drizzled) then distortion components should not apply and the SIP\n                coefficients should be removed.\n\n                While the SIP distortion coefficients are being applied here, if that was indeed the intent,\n                for consistency please append "-SIP" to the CTYPE in the FITS header or the WCS object.\n\n                '
                log.info(message)
        elif 'B_ORDER' in header and header['B_ORDER'] > 1:
            raise ValueError('B_ORDER provided without corresponding A_ORDER keyword for SIP distortion')
        else:
            a = None
            b = None
        if 'AP_ORDER' in header and header['AP_ORDER'] > 1:
            if 'BP_ORDER' not in header:
                raise ValueError('AP_ORDER provided without corresponding BP_ORDER keyword for SIP distortion')
            m = int(header['AP_ORDER'])
            ap = np.zeros((m + 1, m + 1), np.double)
            for i in range(m + 1):
                for j in range(m - i + 1):
                    key = f'AP_{i}_{j}'
                    if key in header:
                        ap[i, j] = header[key]
                        del header[key]
            m = int(header['BP_ORDER'])
            if m > 1:
                bp = np.zeros((m + 1, m + 1), np.double)
                for i in range(m + 1):
                    for j in range(m - i + 1):
                        key = f'BP_{i}_{j}'
                        if key in header:
                            bp[i, j] = header[key]
                            del header[key]
            else:
                ap = None
                bp = None
            del header['AP_ORDER']
            del header['BP_ORDER']
        elif 'BP_ORDER' in header and header['BP_ORDER'] > 1:
            raise ValueError('BP_ORDER provided without corresponding AP_ORDER keyword for SIP distortion')
        else:
            ap = None
            bp = None
        if a is None and b is None and (ap is None) and (bp is None):
            return None
        if f'CRPIX1{wcskey}' not in header or f'CRPIX2{wcskey}' not in header:
            raise ValueError('Header has SIP keywords without CRPIX keywords')
        crpix1 = header.get(f'CRPIX1{wcskey}')
        crpix2 = header.get(f'CRPIX2{wcskey}')
        return Sip(a, b, ap, bp, (crpix1, crpix2))

    def _write_sip_kw(self):
        if False:
            i = 10
            return i + 15
        '\n        Write out SIP keywords.  Returns a dictionary of key-value\n        pairs.\n        '
        if self.sip is None:
            return {}
        keywords = {}

        def write_array(name, a):
            if False:
                for i in range(10):
                    print('nop')
            if a is None:
                return
            size = a.shape[0]
            trdir = 'sky to detector' if name[-1] == 'P' else 'detector to sky'
            comment = f"SIP polynomial order, axis {ord(name[0]) - ord('A'):d}, {trdir:s}"
            keywords[f'{name}_ORDER'] = (size - 1, comment)
            comment = 'SIP distortion coefficient'
            for i in range(size):
                for j in range(size - i):
                    if a[i, j] != 0.0:
                        keywords[f'{name}_{i:d}_{j:d}'] = (a[i, j], comment)
        write_array('A', self.sip.a)
        write_array('B', self.sip.b)
        write_array('AP', self.sip.ap)
        write_array('BP', self.sip.bp)
        return keywords

    def _denormalize_sky(self, sky):
        if False:
            print('Hello World!')
        if self.wcs.lngtyp != 'RA':
            raise ValueError("WCS does not have longitude type of 'RA', therefore (ra, dec) data can not be used as input")
        if self.wcs.lattyp != 'DEC':
            raise ValueError("WCS does not have longitude type of 'DEC', therefore (ra, dec) data can not be used as input")
        if self.wcs.naxis == 2:
            if self.wcs.lng == 0 and self.wcs.lat == 1:
                return sky
            elif self.wcs.lng == 1 and self.wcs.lat == 0:
                return sky[:, ::-1]
            else:
                raise ValueError('WCS does not have longitude and latitude celestial axes, therefore (ra, dec) data can not be used as input')
        else:
            if self.wcs.lng < 0 or self.wcs.lat < 0:
                raise ValueError('WCS does not have both longitude and latitude celestial axes, therefore (ra, dec) data can not be used as input')
            out = np.zeros((sky.shape[0], self.wcs.naxis))
            out[:, self.wcs.lng] = sky[:, 0]
            out[:, self.wcs.lat] = sky[:, 1]
            return out

    def _normalize_sky(self, sky):
        if False:
            while True:
                i = 10
        if self.wcs.lngtyp != 'RA':
            raise ValueError("WCS does not have longitude type of 'RA', therefore (ra, dec) data can not be returned")
        if self.wcs.lattyp != 'DEC':
            raise ValueError("WCS does not have longitude type of 'DEC', therefore (ra, dec) data can not be returned")
        if self.wcs.naxis == 2:
            if self.wcs.lng == 0 and self.wcs.lat == 1:
                return sky
            elif self.wcs.lng == 1 and self.wcs.lat == 0:
                return sky[:, ::-1]
            else:
                raise ValueError('WCS does not have longitude and latitude celestial axes, therefore (ra, dec) data can not be returned')
        else:
            if self.wcs.lng < 0 or self.wcs.lat < 0:
                raise ValueError('WCS does not have both longitude and latitude celestial axes, therefore (ra, dec) data can not be returned')
            out = np.empty((sky.shape[0], 2))
            out[:, 0] = sky[:, self.wcs.lng]
            out[:, 1] = sky[:, self.wcs.lat]
            return out

    def _array_converter(self, func, sky, *args, ra_dec_order=False):
        if False:
            i = 10
            return i + 15
        '\n        A helper function to support reading either a pair of arrays\n        or a single Nx2 array.\n        '

        def _return_list_of_arrays(axes, origin):
            if False:
                while True:
                    i = 10
            if any((x.size == 0 for x in axes)):
                return axes
            try:
                axes = np.broadcast_arrays(*axes)
            except ValueError:
                raise ValueError('Coordinate arrays are not broadcastable to each other')
            xy = np.hstack([x.reshape((x.size, 1)) for x in axes])
            if ra_dec_order and sky == 'input':
                xy = self._denormalize_sky(xy)
            output = func(xy, origin)
            if ra_dec_order and sky == 'output':
                output = self._normalize_sky(output)
                return (output[:, 0].reshape(axes[0].shape), output[:, 1].reshape(axes[0].shape))
            return [output[:, i].reshape(axes[0].shape) for i in range(output.shape[1])]

        def _return_single_array(xy, origin):
            if False:
                while True:
                    i = 10
            if xy.shape[-1] != self.naxis:
                raise ValueError(f'When providing two arguments, the array must be of shape (N, {self.naxis})')
            if 0 in xy.shape:
                return xy
            if ra_dec_order and sky == 'input':
                xy = self._denormalize_sky(xy)
            result = func(xy, origin)
            if ra_dec_order and sky == 'output':
                result = self._normalize_sky(result)
            return result
        if len(args) == 2:
            try:
                (xy, origin) = args
                xy = np.asarray(xy)
                origin = int(origin)
            except Exception:
                raise TypeError(f'When providing two arguments, they must be (coords[N][{self.naxis}], origin)')
            if xy.shape == () or len(xy.shape) == 1:
                return _return_list_of_arrays([xy], origin)
            return _return_single_array(xy, origin)
        elif len(args) == self.naxis + 1:
            axes = args[:-1]
            origin = args[-1]
            try:
                axes = [np.asarray(x) for x in axes]
                origin = int(origin)
            except Exception:
                raise TypeError('When providing more than two arguments, they must be a 1-D array for each axis, followed by an origin.')
            return _return_list_of_arrays(axes, origin)
        raise TypeError(f'WCS projection has {self.naxis} dimensions, so expected 2 (an Nx{self.naxis} array and the origin argument) or {self.naxis + 1} arguments (the position in each dimension, and the origin argument). Instead, {len(args)} arguments were given.')

    def all_pix2world(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._array_converter(self._all_pix2world, 'output', *args, **kwargs)
    all_pix2world.__doc__ = f"""\n        Transforms pixel coordinates to world coordinates.\n\n        Performs all of the following in series:\n\n            - Detector to image plane correction (if present in the\n              FITS file)\n\n            - `SIP`_ distortion correction (if present in the FITS\n              file)\n\n            - `distortion paper`_ table-lookup correction (if present\n              in the FITS file)\n\n            - `wcslib`_ "core" WCS transformation\n\n        Parameters\n        ----------\n        {docstrings.TWO_OR_MORE_ARGS('naxis', 8)}\n\n            For a transformation that is not two-dimensional, the\n            two-argument form must be used.\n\n        {docstrings.RA_DEC_ORDER(8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('sky coordinates, in degrees', 8)}\n\n        Notes\n        -----\n        The order of the axes for the result is determined by the\n        ``CTYPEia`` keywords in the FITS header, therefore it may not\n        always be of the form (*ra*, *dec*).  The\n        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,\n        `~astropy.wcs.Wcsprm.lattyp` and `~astropy.wcs.Wcsprm.lngtyp`\n        members can be used to determine the order of the axes.\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        SingularMatrixError\n            Linear transformation matrix is singular.\n\n        InconsistentAxisTypesError\n            Inconsistent or unrecognized coordinate axis types.\n\n        ValueError\n            Invalid parameter value.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n\n        ValueError\n            x- and y-coordinate arrays are not the same size.\n\n        InvalidTransformError\n            Invalid coordinate transformation parameters.\n\n        InvalidTransformError\n            Ill-conditioned coordinate transformation parameters.\n        """

    def wcs_pix2world(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if self.wcs is None:
            raise ValueError('No basic WCS settings were created.')
        return self._array_converter(lambda xy, o: self.wcs.p2s(xy, o)['world'], 'output', *args, **kwargs)
    wcs_pix2world.__doc__ = f"\n        Transforms pixel coordinates to world coordinates by doing\n        only the basic `wcslib`_ transformation.\n\n        No `SIP`_ or `distortion paper`_ table lookup correction is\n        applied.  To perform distortion correction, see\n        `~astropy.wcs.WCS.all_pix2world`,\n        `~astropy.wcs.WCS.sip_pix2foc`, `~astropy.wcs.WCS.p4_pix2foc`,\n        or `~astropy.wcs.WCS.pix2foc`.\n\n        Parameters\n        ----------\n        {docstrings.TWO_OR_MORE_ARGS('naxis', 8)}\n\n            For a transformation that is not two-dimensional, the\n            two-argument form must be used.\n\n        {docstrings.RA_DEC_ORDER(8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('world coordinates, in degrees', 8)}\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        SingularMatrixError\n            Linear transformation matrix is singular.\n\n        InconsistentAxisTypesError\n            Inconsistent or unrecognized coordinate axis types.\n\n        ValueError\n            Invalid parameter value.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n\n        ValueError\n            x- and y-coordinate arrays are not the same size.\n\n        InvalidTransformError\n            Invalid coordinate transformation parameters.\n\n        InvalidTransformError\n            Ill-conditioned coordinate transformation parameters.\n\n        Notes\n        -----\n        The order of the axes for the result is determined by the\n        ``CTYPEia`` keywords in the FITS header, therefore it may not\n        always be of the form (*ra*, *dec*).  The\n        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,\n        `~astropy.wcs.Wcsprm.lattyp` and `~astropy.wcs.Wcsprm.lngtyp`\n        members can be used to determine the order of the axes.\n\n        "

    def _all_world2pix(self, world, origin, tolerance, maxiter, adaptive, detect_divergence, quiet):
        if False:
            return 10
        pix0 = self.wcs_world2pix(world, origin)
        if not self.has_distortion:
            return pix0
        pix = pix0.copy()
        dpix = self.pix2foc(pix, origin) - pix0
        pix -= dpix
        dn = np.sum(dpix * dpix, axis=1)
        dnprev = dn.copy()
        tol2 = tolerance ** 2
        k = 1
        ind = None
        inddiv = None
        old_invalid = np.geterr()['invalid']
        old_over = np.geterr()['over']
        np.seterr(invalid='ignore', over='ignore')
        if not adaptive:
            while np.nanmax(dn) >= tol2 and k < maxiter:
                dpix = self.pix2foc(pix, origin) - pix0
                dn = np.sum(dpix * dpix, axis=1)
                if detect_divergence:
                    divergent = dn >= dnprev
                    if np.any(divergent):
                        slowconv = dn >= tol2
                        (inddiv,) = np.where(divergent & slowconv)
                        if inddiv.shape[0] > 0:
                            conv = dn < dnprev
                            iconv = np.where(conv)
                            dpixgood = dpix[iconv]
                            pix[iconv] -= dpixgood
                            dpix[iconv] = dpixgood
                            (ind,) = np.where(slowconv & conv)
                            pix0 = pix0[ind]
                            dnprev[ind] = dn[ind]
                            k += 1
                            adaptive = True
                            break
                    dnprev = dn
                pix -= dpix
                k += 1
        if adaptive:
            if ind is None:
                (ind,) = np.where(np.isfinite(pix).all(axis=1))
                pix0 = pix0[ind]
            while ind.shape[0] > 0 and k < maxiter:
                dpixnew = self.pix2foc(pix[ind], origin) - pix0
                dnnew = np.sum(np.square(dpixnew), axis=1)
                dnprev[ind] = dn[ind].copy()
                dn[ind] = dnnew
                if detect_divergence:
                    conv = dnnew < dnprev[ind]
                    iconv = np.where(conv)
                    iiconv = ind[iconv]
                    dpixgood = dpixnew[iconv]
                    pix[iiconv] -= dpixgood
                    dpix[iiconv] = dpixgood
                    (subind,) = np.where((dnnew >= tol2) & conv)
                else:
                    pix[ind] -= dpixnew
                    dpix[ind] = dpixnew
                    (subind,) = np.where(dnnew >= tol2)
                ind = ind[subind]
                pix0 = pix0[subind]
                k += 1
        invalid = ~np.all(np.isfinite(pix), axis=1) & np.all(np.isfinite(world), axis=1)
        (inddiv,) = np.where((dn >= tol2) & (dn >= dnprev) | invalid)
        if inddiv.shape[0] == 0:
            inddiv = None
        if k >= maxiter:
            (ind,) = np.where((dn >= tol2) & (dn < dnprev) & ~invalid)
            if ind.shape[0] == 0:
                ind = None
        else:
            ind = None
        np.seterr(invalid=old_invalid, over=old_over)
        if (ind is not None or inddiv is not None) and (not quiet):
            if inddiv is None:
                raise NoConvergence(f"'WCS.all_world2pix' failed to converge to the requested accuracy after {k:d} iterations.", best_solution=pix, accuracy=np.abs(dpix), niter=k, slow_conv=ind, divergent=None)
            else:
                raise NoConvergence(f"'WCS.all_world2pix' failed to converge to the requested accuracy.\nAfter {k:d} iterations, the solution is diverging at least for one input point.", best_solution=pix, accuracy=np.abs(dpix), niter=k, slow_conv=ind, divergent=inddiv)
        return pix

    def all_world2pix(self, *args, tolerance=0.0001, maxiter=20, adaptive=False, detect_divergence=True, quiet=False, **kwargs):
        if False:
            i = 10
            return i + 15
        if self.wcs is None:
            raise ValueError('No basic WCS settings were created.')
        return self._array_converter(lambda *args, **kwargs: self._all_world2pix(*args, tolerance=tolerance, maxiter=maxiter, adaptive=adaptive, detect_divergence=detect_divergence, quiet=quiet), 'input', *args, **kwargs)
    all_world2pix.__doc__ = f"""\n        all_world2pix(*arg, tolerance=1.0e-4, maxiter=20,\n        adaptive=False, detect_divergence=True, quiet=False)\n\n        Transforms world coordinates to pixel coordinates, using\n        numerical iteration to invert the full forward transformation\n        `~astropy.wcs.WCS.all_pix2world` with complete\n        distortion model.\n\n\n        Parameters\n        ----------\n        {docstrings.TWO_OR_MORE_ARGS('naxis', 8)}\n\n            For a transformation that is not two-dimensional, the\n            two-argument form must be used.\n\n        {docstrings.RA_DEC_ORDER(8)}\n\n        tolerance : float, optional (default = 1.0e-4)\n            Tolerance of solution. Iteration terminates when the\n            iterative solver estimates that the "true solution" is\n            within this many pixels current estimate, more\n            specifically, when the correction to the solution found\n            during the previous iteration is smaller\n            (in the sense of the L2 norm) than ``tolerance``.\n\n        maxiter : int, optional (default = 20)\n            Maximum number of iterations allowed to reach a solution.\n\n        quiet : bool, optional (default = False)\n            Do not throw :py:class:`NoConvergence` exceptions when\n            the method does not converge to a solution with the\n            required accuracy within a specified number of maximum\n            iterations set by ``maxiter`` parameter. Instead,\n            simply return the found solution.\n\n        Other Parameters\n        ----------------\n        adaptive : bool, optional (default = False)\n            Specifies whether to adaptively select only points that\n            did not converge to a solution within the required\n            accuracy for the next iteration. Default is recommended\n            for HST as well as most other instruments.\n\n            .. note::\n               The :py:meth:`all_world2pix` uses a vectorized\n               implementation of the method of consecutive\n               approximations (see ``Notes`` section below) in which it\n               iterates over *all* input points *regardless* until\n               the required accuracy has been reached for *all* input\n               points. In some cases it may be possible that\n               *almost all* points have reached the required accuracy\n               but there are only a few of input data points for\n               which additional iterations may be needed (this\n               depends mostly on the characteristics of the geometric\n               distortions for a given instrument). In this situation\n               it may be advantageous to set ``adaptive`` = `True` in\n               which case :py:meth:`all_world2pix` will continue\n               iterating *only* over the points that have not yet\n               converged to the required accuracy. However, for the\n               HST's ACS/WFC detector, which has the strongest\n               distortions of all HST instruments, testing has\n               shown that enabling this option would lead to a about\n               50-100% penalty in computational time (depending on\n               specifics of the image, geometric distortions, and\n               number of input points to be converted). Therefore,\n               for HST and possibly instruments, it is recommended\n               to set ``adaptive`` = `False`. The only danger in\n               getting this setting wrong will be a performance\n               penalty.\n\n            .. note::\n               When ``detect_divergence`` is `True`,\n               :py:meth:`all_world2pix` will automatically switch\n               to the adaptive algorithm once divergence has been\n               detected.\n\n        detect_divergence : bool, optional (default = True)\n            Specifies whether to perform a more detailed analysis\n            of the convergence to a solution. Normally\n            :py:meth:`all_world2pix` may not achieve the required\n            accuracy if either the ``tolerance`` or ``maxiter`` arguments\n            are too low. However, it may happen that for some\n            geometric distortions the conditions of convergence for\n            the method of consecutive approximations used by\n            :py:meth:`all_world2pix` may not be satisfied, in which\n            case consecutive approximations to the solution will\n            diverge regardless of the ``tolerance`` or ``maxiter``\n            settings.\n\n            When ``detect_divergence`` is `False`, these divergent\n            points will be detected as not having achieved the\n            required accuracy (without further details). In addition,\n            if ``adaptive`` is `False` then the algorithm will not\n            know that the solution (for specific points) is diverging\n            and will continue iterating and trying to "improve"\n            diverging solutions. This may result in ``NaN`` or\n            ``Inf`` values in the return results (in addition to a\n            performance penalties). Even when ``detect_divergence``\n            is `False`, :py:meth:`all_world2pix`, at the end of the\n            iterative process, will identify invalid results\n            (``NaN`` or ``Inf``) as "diverging" solutions and will\n            raise :py:class:`NoConvergence` unless the ``quiet``\n            parameter is set to `True`.\n\n            When ``detect_divergence`` is `True`,\n            :py:meth:`all_world2pix` will detect points for which\n            current correction to the coordinates is larger than\n            the correction applied during the previous iteration\n            **if** the requested accuracy **has not yet been\n            achieved**. In this case, if ``adaptive`` is `True`,\n            these points will be excluded from further iterations and\n            if ``adaptive`` is `False`, :py:meth:`all_world2pix` will\n            automatically switch to the adaptive algorithm. Thus, the\n            reported divergent solution will be the latest converging\n            solution computed immediately *before* divergence\n            has been detected.\n\n            .. note::\n               When accuracy has been achieved, small increases in\n               current corrections may be possible due to rounding\n               errors (when ``adaptive`` is `False`) and such\n               increases will be ignored.\n\n            .. note::\n               Based on our testing using HST ACS/WFC images, setting\n               ``detect_divergence`` to `True` will incur about 5-20%\n               performance penalty with the larger penalty\n               corresponding to ``adaptive`` set to `True`.\n               Because the benefits of enabling this\n               feature outweigh the small performance penalty,\n               especially when ``adaptive`` = `False`, it is\n               recommended to set ``detect_divergence`` to `True`,\n               unless extensive testing of the distortion models for\n               images from specific instruments show a good stability\n               of the numerical method for a wide range of\n               coordinates (even outside the image itself).\n\n            .. note::\n               Indices of the diverging inverse solutions will be\n               reported in the ``divergent`` attribute of the\n               raised :py:class:`NoConvergence` exception object.\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('pixel coordinates', 8)}\n\n        Notes\n        -----\n        The order of the axes for the input world array is determined by\n        the ``CTYPEia`` keywords in the FITS header, therefore it may\n        not always be of the form (*ra*, *dec*).  The\n        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,\n        `~astropy.wcs.Wcsprm.lattyp`, and\n        `~astropy.wcs.Wcsprm.lngtyp`\n        members can be used to determine the order of the axes.\n\n        Using the method of fixed-point iterations approximations we\n        iterate starting with the initial approximation, which is\n        computed using the non-distortion-aware\n        :py:meth:`wcs_world2pix` (or equivalent).\n\n        The :py:meth:`all_world2pix` function uses a vectorized\n        implementation of the method of consecutive approximations and\n        therefore it is highly efficient (>30x) when *all* data points\n        that need to be converted from sky coordinates to image\n        coordinates are passed at *once*. Therefore, it is advisable,\n        whenever possible, to pass as input a long array of all points\n        that need to be converted to :py:meth:`all_world2pix` instead\n        of calling :py:meth:`all_world2pix` for each data point. Also\n        see the note to the ``adaptive`` parameter.\n\n        Raises\n        ------\n        NoConvergence\n            The method did not converge to a\n            solution to the required accuracy within a specified\n            number of maximum iterations set by the ``maxiter``\n            parameter. To turn off this exception, set ``quiet`` to\n            `True`. Indices of the points for which the requested\n            accuracy was not achieved (if any) will be listed in the\n            ``slow_conv`` attribute of the\n            raised :py:class:`NoConvergence` exception object.\n\n            See :py:class:`NoConvergence` documentation for\n            more details.\n\n        MemoryError\n            Memory allocation failed.\n\n        SingularMatrixError\n            Linear transformation matrix is singular.\n\n        InconsistentAxisTypesError\n            Inconsistent or unrecognized coordinate axis types.\n\n        ValueError\n            Invalid parameter value.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n\n        ValueError\n            x- and y-coordinate arrays are not the same size.\n\n        InvalidTransformError\n            Invalid coordinate transformation parameters.\n\n        InvalidTransformError\n            Ill-conditioned coordinate transformation parameters.\n\n        Examples\n        --------\n        >>> import astropy.io.fits as fits\n        >>> import astropy.wcs as wcs\n        >>> import numpy as np\n        >>> import os\n\n        >>> filename = os.path.join(wcs.__path__[0], 'tests/data/j94f05bgq_flt.fits')\n        >>> hdulist = fits.open(filename)\n        >>> w = wcs.WCS(hdulist[('sci',1)].header, hdulist)\n        >>> hdulist.close()\n\n        >>> ra, dec = w.all_pix2world([1,2,3], [1,1,1], 1)\n        >>> print(ra)  # doctest: +FLOAT_CMP\n        [ 5.52645627  5.52649663  5.52653698]\n        >>> print(dec)  # doctest: +FLOAT_CMP\n        [-72.05171757 -72.05171276 -72.05170795]\n        >>> radec = w.all_pix2world([[1,1], [2,1], [3,1]], 1)\n        >>> print(radec)  # doctest: +FLOAT_CMP\n        [[  5.52645627 -72.05171757]\n         [  5.52649663 -72.05171276]\n         [  5.52653698 -72.05170795]]\n        >>> x, y = w.all_world2pix(ra, dec, 1)\n        >>> print(x)  # doctest: +FLOAT_CMP\n        [ 1.00000238  2.00000237  3.00000236]\n        >>> print(y)  # doctest: +FLOAT_CMP\n        [ 0.99999996  0.99999997  0.99999997]\n        >>> xy = w.all_world2pix(radec, 1)\n        >>> print(xy)  # doctest: +FLOAT_CMP\n        [[ 1.00000238  0.99999996]\n         [ 2.00000237  0.99999997]\n         [ 3.00000236  0.99999997]]\n        >>> xy = w.all_world2pix(radec, 1, maxiter=3,\n        ...                      tolerance=1.0e-10, quiet=False)\n        Traceback (most recent call last):\n        ...\n        NoConvergence: 'WCS.all_world2pix' failed to converge to the\n        requested accuracy. After 3 iterations, the solution is\n        diverging at least for one input point.\n\n        >>> # Now try to use some diverging data:\n        >>> divradec = w.all_pix2world([[1.0, 1.0],\n        ...                             [10000.0, 50000.0],\n        ...                             [3.0, 1.0]], 1)\n        >>> print(divradec)  # doctest: +FLOAT_CMP\n        [[  5.52645627 -72.05171757]\n         [  7.15976932 -70.8140779 ]\n         [  5.52653698 -72.05170795]]\n\n        >>> # First, turn detect_divergence on:\n        >>> try:  # doctest: +FLOAT_CMP\n        ...   xy = w.all_world2pix(divradec, 1, maxiter=20,\n        ...                        tolerance=1.0e-4, adaptive=False,\n        ...                        detect_divergence=True,\n        ...                        quiet=False)\n        ... except wcs.wcs.NoConvergence as e:\n        ...   print("Indices of diverging points: {{0}}"\n        ...         .format(e.divergent))\n        ...   print("Indices of poorly converging points: {{0}}"\n        ...         .format(e.slow_conv))\n        ...   print("Best solution:\\n{{0}}".format(e.best_solution))\n        ...   print("Achieved accuracy:\\n{{0}}".format(e.accuracy))\n        Indices of diverging points: [1]\n        Indices of poorly converging points: None\n        Best solution:\n        [[  1.00000238e+00   9.99999965e-01]\n         [ -1.99441636e+06   1.44309097e+06]\n         [  3.00000236e+00   9.99999966e-01]]\n        Achieved accuracy:\n        [[  6.13968380e-05   8.59638593e-07]\n         [  8.59526812e+11   6.61713548e+11]\n         [  6.09398446e-05   8.38759724e-07]]\n        >>> raise e\n        Traceback (most recent call last):\n        ...\n        NoConvergence: 'WCS.all_world2pix' failed to converge to the\n        requested accuracy.  After 5 iterations, the solution is\n        diverging at least for one input point.\n\n        >>> # This time turn detect_divergence off:\n        >>> try:  # doctest: +FLOAT_CMP\n        ...   xy = w.all_world2pix(divradec, 1, maxiter=20,\n        ...                        tolerance=1.0e-4, adaptive=False,\n        ...                        detect_divergence=False,\n        ...                        quiet=False)\n        ... except wcs.wcs.NoConvergence as e:\n        ...   print("Indices of diverging points: {{0}}"\n        ...         .format(e.divergent))\n        ...   print("Indices of poorly converging points: {{0}}"\n        ...         .format(e.slow_conv))\n        ...   print("Best solution:\\n{{0}}".format(e.best_solution))\n        ...   print("Achieved accuracy:\\n{{0}}".format(e.accuracy))\n        Indices of diverging points: [1]\n        Indices of poorly converging points: None\n        Best solution:\n        [[ 1.00000009  1.        ]\n         [        nan         nan]\n         [ 3.00000009  1.        ]]\n        Achieved accuracy:\n        [[  2.29417358e-06   3.21222995e-08]\n         [             nan              nan]\n         [  2.27407877e-06   3.13005639e-08]]\n        >>> raise e\n        Traceback (most recent call last):\n        ...\n        NoConvergence: 'WCS.all_world2pix' failed to converge to the\n        requested accuracy.  After 6 iterations, the solution is\n        diverging at least for one input point.\n\n        """

    def wcs_world2pix(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if self.wcs is None:
            raise ValueError('No basic WCS settings were created.')
        return self._array_converter(lambda xy, o: self.wcs.s2p(xy, o)['pixcrd'], 'input', *args, **kwargs)
    wcs_world2pix.__doc__ = f"\n        Transforms world coordinates to pixel coordinates, using only\n        the basic `wcslib`_ WCS transformation.  No `SIP`_ or\n        `distortion paper`_ table lookup transformation is applied.\n\n        Parameters\n        ----------\n        {docstrings.TWO_OR_MORE_ARGS('naxis', 8)}\n\n            For a transformation that is not two-dimensional, the\n            two-argument form must be used.\n\n        {docstrings.RA_DEC_ORDER(8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('pixel coordinates', 8)}\n\n        Notes\n        -----\n        The order of the axes for the input world array is determined by\n        the ``CTYPEia`` keywords in the FITS header, therefore it may\n        not always be of the form (*ra*, *dec*).  The\n        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,\n        `~astropy.wcs.Wcsprm.lattyp` and `~astropy.wcs.Wcsprm.lngtyp`\n        members can be used to determine the order of the axes.\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        SingularMatrixError\n            Linear transformation matrix is singular.\n\n        InconsistentAxisTypesError\n            Inconsistent or unrecognized coordinate axis types.\n\n        ValueError\n            Invalid parameter value.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n\n        ValueError\n            x- and y-coordinate arrays are not the same size.\n\n        InvalidTransformError\n            Invalid coordinate transformation parameters.\n\n        InvalidTransformError\n            Ill-conditioned coordinate transformation parameters.\n        "

    def pix2foc(self, *args):
        if False:
            print('Hello World!')
        return self._array_converter(self._pix2foc, None, *args)
    pix2foc.__doc__ = f"\n        Convert pixel coordinates to focal plane coordinates using the\n        `SIP`_ polynomial distortion convention and `distortion\n        paper`_ table-lookup correction.\n\n        The output is in absolute pixel coordinates, not relative to\n        ``CRPIX``.\n\n        Parameters\n        ----------\n\n        {docstrings.TWO_OR_MORE_ARGS('2', 8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('focal coordinates', 8)}\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n        "

    def p4_pix2foc(self, *args):
        if False:
            i = 10
            return i + 15
        return self._array_converter(self._p4_pix2foc, None, *args)
    p4_pix2foc.__doc__ = f"\n        Convert pixel coordinates to focal plane coordinates using\n        `distortion paper`_ table-lookup correction.\n\n        The output is in absolute pixel coordinates, not relative to\n        ``CRPIX``.\n\n        Parameters\n        ----------\n\n        {docstrings.TWO_OR_MORE_ARGS('2', 8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('focal coordinates', 8)}\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n        "

    def det2im(self, *args):
        if False:
            print('Hello World!')
        return self._array_converter(self._det2im, None, *args)
    det2im.__doc__ = f"\n        Convert detector coordinates to image plane coordinates using\n        `distortion paper`_ table-lookup correction.\n\n        The output is in absolute pixel coordinates, not relative to\n        ``CRPIX``.\n\n        Parameters\n        ----------\n\n        {docstrings.TWO_OR_MORE_ARGS('2', 8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('pixel coordinates', 8)}\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n        "

    def sip_pix2foc(self, *args):
        if False:
            print('Hello World!')
        if self.sip is None:
            if len(args) == 2:
                return args[0]
            elif len(args) == 3:
                return args[:2]
            else:
                raise TypeError('Wrong number of arguments')
        return self._array_converter(self.sip.pix2foc, None, *args)
    sip_pix2foc.__doc__ = f"\n        Convert pixel coordinates to focal plane coordinates using the\n        `SIP`_ polynomial distortion convention.\n\n        The output is in pixel coordinates, relative to ``CRPIX``.\n\n        FITS WCS `distortion paper`_ table lookup correction is not\n        applied, even if that information existed in the FITS file\n        that initialized this :class:`~astropy.wcs.WCS` object.  To\n        correct for that, use `~astropy.wcs.WCS.pix2foc` or\n        `~astropy.wcs.WCS.p4_pix2foc`.\n\n        Parameters\n        ----------\n\n        {docstrings.TWO_OR_MORE_ARGS('2', 8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('focal coordinates', 8)}\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n        "

    def sip_foc2pix(self, *args):
        if False:
            while True:
                i = 10
        if self.sip is None:
            if len(args) == 2:
                return args[0]
            elif len(args) == 3:
                return args[:2]
            else:
                raise TypeError('Wrong number of arguments')
        return self._array_converter(self.sip.foc2pix, None, *args)
    sip_foc2pix.__doc__ = f"\n        Convert focal plane coordinates to pixel coordinates using the\n        `SIP`_ polynomial distortion convention.\n\n        FITS WCS `distortion paper`_ table lookup distortion\n        correction is not applied, even if that information existed in\n        the FITS file that initialized this `~astropy.wcs.WCS` object.\n\n        Parameters\n        ----------\n\n        {docstrings.TWO_OR_MORE_ARGS('2', 8)}\n\n        Returns\n        -------\n\n        {docstrings.RETURNS('pixel coordinates', 8)}\n\n        Raises\n        ------\n        MemoryError\n            Memory allocation failed.\n\n        ValueError\n            Invalid coordinate transformation parameters.\n        "

    def proj_plane_pixel_scales(self):
        if False:
            while True:
                i = 10
        '\n        Calculate pixel scales along each axis of the image pixel at\n        the ``CRPIX`` location once it is projected onto the\n        "plane of intermediate world coordinates" as defined in\n        `Greisen & Calabretta 2002, A&A, 395, 1061 <https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1061G>`_.\n\n        .. note::\n            This method is concerned **only** about the transformation\n            "image plane"->"projection plane" and **not** about the\n            transformation "celestial sphere"->"projection plane"->"image plane".\n            Therefore, this function ignores distortions arising due to\n            non-linear nature of most projections.\n\n        .. note::\n            This method only returns sensible answers if the WCS contains\n            celestial axes, i.e., the `~astropy.wcs.WCS.celestial` WCS object.\n\n        Returns\n        -------\n        scale : list of `~astropy.units.Quantity`\n            A vector of projection plane increments corresponding to each\n            pixel side (axis).\n\n        See Also\n        --------\n        astropy.wcs.utils.proj_plane_pixel_scales\n\n        '
        from astropy.wcs.utils import proj_plane_pixel_scales
        values = proj_plane_pixel_scales(self)
        units = [u.Unit(x) for x in self.wcs.cunit]
        return [value * unit for (value, unit) in zip(values, units)]

    def proj_plane_pixel_area(self):
        if False:
            i = 10
            return i + 15
        '\n        For a **celestial** WCS (see `astropy.wcs.WCS.celestial`), returns pixel\n        area of the image pixel at the ``CRPIX`` location once it is projected\n        onto the "plane of intermediate world coordinates" as defined in\n        `Greisen & Calabretta 2002, A&A, 395, 1061 <https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1061G>`_.\n\n        .. note::\n            This function is concerned **only** about the transformation\n            "image plane"->"projection plane" and **not** about the\n            transformation "celestial sphere"->"projection plane"->"image plane".\n            Therefore, this function ignores distortions arising due to\n            non-linear nature of most projections.\n\n        .. note::\n            This method only returns sensible answers if the WCS contains\n            celestial axes, i.e., the `~astropy.wcs.WCS.celestial` WCS object.\n\n        Returns\n        -------\n        area : `~astropy.units.Quantity`\n            Area (in the projection plane) of the pixel at ``CRPIX`` location.\n\n        Raises\n        ------\n        ValueError\n            Pixel area is defined only for 2D pixels. Most likely the\n            `~astropy.wcs.Wcsprm.cd` matrix of the `~astropy.wcs.WCS.celestial`\n            WCS is not a square matrix of second order.\n\n        Notes\n        -----\n        Depending on the application, square root of the pixel area can be used to\n        represent a single pixel scale of an equivalent square pixel\n        whose area is equal to the area of a generally non-square pixel.\n\n        See Also\n        --------\n        astropy.wcs.utils.proj_plane_pixel_area\n\n        '
        from astropy.wcs.utils import proj_plane_pixel_area
        value = proj_plane_pixel_area(self)
        unit = u.Unit(self.wcs.cunit[0]) * u.Unit(self.wcs.cunit[1])
        return value * unit

    def to_fits(self, relax=False, key=None):
        if False:
            return 10
        '\n        Generate an `~astropy.io.fits.HDUList` object with all of the\n        information stored in this object.  This should be logically identical\n        to the input FITS file, but it will be normalized in a number of ways.\n\n        See `to_header` for some warnings about the output produced.\n\n        Parameters\n        ----------\n        relax : bool or int, optional\n            Degree of permissiveness:\n\n            - `False` (default): Write all extensions that are\n              considered to be safe and recommended.\n\n            - `True`: Write all recognized informal extensions of the\n              WCS standard.\n\n            - `int`: a bit field selecting specific extensions to\n              write.  See :ref:`astropy:relaxwrite` for details.\n\n        key : str\n            The name of a particular WCS transform to use.  This may be\n            either ``\' \'`` or ``\'A\'``-``\'Z\'`` and corresponds to the ``"a"``\n            part of the ``CTYPEia`` cards.\n\n        Returns\n        -------\n        hdulist : `~astropy.io.fits.HDUList`\n        '
        header = self.to_header(relax=relax, key=key)
        hdu = fits.PrimaryHDU(header=header)
        hdulist = fits.HDUList(hdu)
        self._write_det2im(hdulist)
        self._write_distortion_kw(hdulist)
        return hdulist

    def to_header(self, relax=None, key=None):
        if False:
            while True:
                i = 10
        'Generate an `astropy.io.fits.Header` object with the basic WCS\n        and SIP information stored in this object.  This should be\n        logically identical to the input FITS file, but it will be\n        normalized in a number of ways.\n\n        .. warning::\n\n          This function does not write out FITS WCS `distortion\n          paper`_ information, since that requires multiple FITS\n          header data units.  To get a full representation of\n          everything in this object, use `to_fits`.\n\n        Parameters\n        ----------\n        relax : bool or int, optional\n            Degree of permissiveness:\n\n            - `False` (default): Write all extensions that are\n              considered to be safe and recommended.\n\n            - `True`: Write all recognized informal extensions of the\n              WCS standard.\n\n            - `int`: a bit field selecting specific extensions to\n              write.  See :ref:`astropy:relaxwrite` for details.\n\n            If the ``relax`` keyword argument is not given and any\n            keywords were omitted from the output, an\n            `~astropy.utils.exceptions.AstropyWarning` is displayed.\n            To override this, explicitly pass a value to ``relax``.\n\n        key : str\n            The name of a particular WCS transform to use.  This may be\n            either ``\' \'`` or ``\'A\'``-``\'Z\'`` and corresponds to the ``"a"``\n            part of the ``CTYPEia`` cards.\n\n        Returns\n        -------\n        header : `astropy.io.fits.Header`\n\n        Notes\n        -----\n        The output header will almost certainly differ from the input in a\n        number of respects:\n\n          1. The output header only contains WCS-related keywords.  In\n             particular, it does not contain syntactically-required\n             keywords such as ``SIMPLE``, ``NAXIS``, ``BITPIX``, or\n             ``END``.\n\n          2. Deprecated (e.g. ``CROTAn``) or non-standard usage will\n             be translated to standard (this is partially dependent on\n             whether ``fix`` was applied).\n\n          3. Quantities will be converted to the units used internally,\n             basically SI with the addition of degrees.\n\n          4. Floating-point quantities may be given to a different decimal\n             precision.\n\n          5. Elements of the ``PCi_j`` matrix will be written if and\n             only if they differ from the unit matrix.  Thus, if the\n             matrix is unity then no elements will be written.\n\n          6. Additional keywords such as ``WCSAXES``, ``CUNITia``,\n             ``LONPOLEa`` and ``LATPOLEa`` may appear.\n\n          7. The original keycomments will be lost, although\n             `to_header` tries hard to write meaningful comments.\n\n          8. Keyword order may be changed.\n\n        '
        precision = WCSHDO_P14
        display_warning = False
        if relax is None:
            display_warning = True
            relax = False
        if relax not in (True, False):
            do_sip = relax & WCSHDO_SIP
            relax &= ~WCSHDO_SIP
        else:
            do_sip = relax
            relax = WCSHDO_all if relax is True else WCSHDO_safe
        relax = precision | relax
        if self.wcs is not None:
            if key is not None:
                orig_key = self.wcs.alt
                self.wcs.alt = key
            header_string = self.wcs.to_header(relax)
            header = fits.Header.fromstring(header_string)
            keys_to_remove = ['', ' ', 'COMMENT']
            for kw in keys_to_remove:
                if kw in header:
                    del header[kw]
            if _WCS_TPD_WARN_LT71:
                for (kw, val) in header.items():
                    if kw[:5] in ('CPDIS', 'CQDIS') and val == 'TPD':
                        warnings.warn(f'WCS contains a TPD distortion model in {kw}. WCSLIB {_wcs.__version__} is writing this in a format incompatible with current versions - please update to 7.4 or use the bundled WCSLIB.', AstropyWarning)
            elif _WCS_TPD_WARN_LT74:
                for (kw, val) in header.items():
                    if kw[:5] in ('CPDIS', 'CQDIS') and val == 'TPD':
                        warnings.warn(f'WCS contains a TPD distortion model in {kw}, which requires WCSLIB 7.4 or later to store in a FITS header (having {_wcs.__version__}).', AstropyWarning)
        else:
            header = fits.Header()
        if do_sip and self.sip is not None:
            if self.wcs is not None and any((not ctyp.endswith('-SIP') for ctyp in self.wcs.ctype)):
                self._fix_ctype(header, add_sip=True)
            for (kw, val) in self._write_sip_kw().items():
                header[kw] = val
        if not do_sip and self.wcs is not None and any(self.wcs.ctype) and (self.sip is not None):
            header = self._fix_ctype(header, add_sip=False)
        if display_warning:
            full_header = self.to_header(relax=True, key=key)
            missing_keys = []
            for (kw, val) in full_header.items():
                if kw not in header:
                    missing_keys.append(kw)
            if len(missing_keys):
                warnings.warn(f"Some non-standard WCS keywords were excluded: {', '.join(missing_keys)} Use the ``relax`` kwarg to control this.", AstropyWarning)
            if any(self.wcs.ctype) and self.sip is not None:
                header = self._fix_ctype(header, add_sip=False, log_message=False)
        if key is not None:
            self.wcs.alt = orig_key
        return header

    def _fix_ctype(self, header, add_sip=True, log_message=True):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        header : `~astropy.io.fits.Header`\n            FITS header.\n        add_sip : bool\n            Flag indicating whether "-SIP" should be added or removed from CTYPE keywords.\n\n            Remove "-SIP" from CTYPE when writing out a header with relax=False.\n            This needs to be done outside ``to_header`` because ``to_header`` runs\n            twice when ``relax=False`` and the second time ``relax`` is set to ``True``\n            to display the missing keywords.\n\n            If the user requested SIP distortion to be written out add "-SIP" to\n            CTYPE if it is missing.\n        '
        _add_sip_to_ctype = '\n        Inconsistent SIP distortion information is present in the current WCS:\n        SIP coefficients were detected, but CTYPE is missing "-SIP" suffix,\n        therefore the current WCS is internally inconsistent.\n\n        Because relax has been set to True, the resulting output WCS will have\n        "-SIP" appended to CTYPE in order to make the header internally consistent.\n\n        However, this may produce incorrect astrometry in the output WCS, if\n        in fact the current WCS is already distortion-corrected.\n\n        Therefore, if current WCS is already distortion-corrected (eg, drizzled)\n        then SIP distortion components should not apply. In that case, for a WCS\n        that is already distortion-corrected, please remove the SIP coefficients\n        from the header.\n\n        '
        if log_message:
            if add_sip:
                log.info(_add_sip_to_ctype)
        for i in range(1, self.naxis + 1):
            kw = f'CTYPE{i}{self.wcs.alt}'.strip()
            if kw in header:
                if add_sip:
                    val = header[kw].strip('-SIP') + '-SIP'
                else:
                    val = header[kw].strip('-SIP')
                header[kw] = val
            else:
                continue
        return header

    def to_header_string(self, relax=None):
        if False:
            print('Hello World!')
        '\n        Identical to `to_header`, but returns a string containing the\n        header cards.\n        '
        return str(self.to_header(relax))

    def footprint_to_file(self, filename='footprint.reg', color='green', width=2, coordsys=None):
        if False:
            return 10
        "\n        Writes out a `ds9`_ style regions file. It can be loaded\n        directly by `ds9`_.\n\n        Parameters\n        ----------\n        filename : str, optional\n            Output file name - default is ``'footprint.reg'``\n\n        color : str, optional\n            Color to use when plotting the line.\n\n        width : int, optional\n            Width of the region line.\n\n        coordsys : str, optional\n            Coordinate system. If not specified (default), the ``radesys``\n            value is used. For all possible values, see\n            http://ds9.si.edu/doc/ref/region.html#RegionFileFormat\n\n        "
        comments = '# Region file format: DS9 version 4.0 \n# global color=green font="helvetica 12 bold select=1 highlite=1 edit=1 move=1 delete=1 include=1 fixed=0 source\n'
        coordsys = coordsys or self.wcs.radesys
        if coordsys not in ('PHYSICAL', 'IMAGE', 'FK4', 'B1950', 'FK5', 'J2000', 'GALACTIC', 'ECLIPTIC', 'ICRS', 'LINEAR', 'AMPLIFIER', 'DETECTOR'):
            raise ValueError(f"Coordinate system '{coordsys}' is not supported. A valid one can be given with the 'coordsys' argument.")
        with open(filename, mode='w') as f:
            f.write(comments)
            f.write(f'{coordsys}\n')
            f.write('polygon(')
            ftpr = self.calc_footprint()
            if ftpr is not None:
                ftpr.tofile(f, sep=',')
                f.write(f') # color={color}, width={width:d} \n')

    def _get_naxis(self, header=None):
        if False:
            while True:
                i = 10
        _naxis = []
        if header is not None and (not isinstance(header, (str, bytes))):
            for naxis in itertools.count(1):
                try:
                    _naxis.append(header[f'NAXIS{naxis}'])
                except KeyError:
                    break
        if len(_naxis) == 0:
            _naxis = [0, 0]
        elif len(_naxis) == 1:
            _naxis.append(0)
        self._naxis = _naxis

    def printwcs(self):
        if False:
            return 10
        print(repr(self))

    def __repr__(self):
        if False:
            while True:
                i = 10
        '\n        Return a short description. Simply porting the behavior from\n        the `printwcs()` method.\n        '
        description = ['WCS Keywords\n', f'Number of WCS axes: {self.naxis!r}']
        sfmt = ' : ' + ''.join([f'{{{i}!r}} ' for i in range(self.naxis)])
        keywords = ['CTYPE', 'CRVAL', 'CRPIX']
        values = [self.wcs.ctype, self.wcs.crval, self.wcs.crpix]
        for (keyword, value) in zip(keywords, values):
            description.append(keyword + sfmt.format(*value))
        if hasattr(self.wcs, 'pc'):
            for i in range(self.naxis):
                s = ''
                for j in range(self.naxis):
                    s += ''.join(['PC', str(i + 1), '_', str(j + 1), ' '])
                s += sfmt
                description.append(s.format(*self.wcs.pc[i]))
            s = 'CDELT' + sfmt
            description.append(s.format(*self.wcs.cdelt))
        elif hasattr(self.wcs, 'cd'):
            for i in range(self.naxis):
                s = ''
                for j in range(self.naxis):
                    s += ''.join(['CD', str(i + 1), '_', str(j + 1), ' '])
                s += sfmt
                description.append(s.format(*self.wcs.cd[i]))
        description.append(f"NAXIS : {'  '.join(map(str, self._naxis))}")
        return '\n'.join(description)

    def get_axis_types(self):
        if False:
            i = 10
            return i + 15
        '\n        Similar to `self.wcsprm.axis_types <astropy.wcs.Wcsprm.axis_types>`\n        but provides the information in a more Python-friendly format.\n\n        Returns\n        -------\n        result : list of dict\n\n            Returns a list of dictionaries, one for each axis, each\n            containing attributes about the type of that axis.\n\n            Each dictionary has the following keys:\n\n            - \'coordinate_type\':\n\n              - None: Non-specific coordinate type.\n\n              - \'stokes\': Stokes coordinate.\n\n              - \'celestial\': Celestial coordinate (including ``CUBEFACE``).\n\n              - \'spectral\': Spectral coordinate.\n\n            - \'scale\':\n\n              - \'linear\': Linear axis.\n\n              - \'quantized\': Quantized axis (``STOKES``, ``CUBEFACE``).\n\n              - \'non-linear celestial\': Non-linear celestial axis.\n\n              - \'non-linear spectral\': Non-linear spectral axis.\n\n              - \'logarithmic\': Logarithmic axis.\n\n              - \'tabular\': Tabular axis.\n\n            - \'group\'\n\n              - Group number, e.g. lookup table number\n\n            - \'number\'\n\n              - For celestial axes:\n\n                - 0: Longitude coordinate.\n\n                - 1: Latitude coordinate.\n\n                - 2: ``CUBEFACE`` number.\n\n              - For lookup tables:\n\n                - the axis number in a multidimensional table.\n\n            ``CTYPEia`` in ``"4-3"`` form with unrecognized algorithm code will\n            generate an error.\n        '
        if self.wcs is None:
            raise AttributeError('This WCS object does not have a wcsprm object.')
        coordinate_type_map = {0: None, 1: 'stokes', 2: 'celestial', 3: 'spectral'}
        scale_map = {0: 'linear', 1: 'quantized', 2: 'non-linear celestial', 3: 'non-linear spectral', 4: 'logarithmic', 5: 'tabular'}
        result = []
        for axis_type in self.wcs.axis_types:
            subresult = {}
            coordinate_type = axis_type // 1000 % 10
            subresult['coordinate_type'] = coordinate_type_map[coordinate_type]
            scale = axis_type // 100 % 10
            subresult['scale'] = scale_map[scale]
            group = axis_type // 10 % 10
            subresult['group'] = group
            number = axis_type % 10
            subresult['number'] = number
            result.append(subresult)
        return result

    def __reduce__(self):
        if False:
            i = 10
            return i + 15
        '\n        Support pickling of WCS objects.  This is done by serializing\n        to an in-memory FITS file and dumping that as a string.\n        '
        hdulist = self.to_fits(relax=True)
        buffer = io.BytesIO()
        hdulist.writeto(buffer)
        dct = self.__dict__.copy()
        dct['_alt_wcskey'] = self.wcs.alt
        return (__WCS_unpickle__, (self.__class__, dct, buffer.getvalue()))

    def dropaxis(self, dropax):
        if False:
            i = 10
            return i + 15
        '\n        Remove an axis from the WCS.\n\n        Parameters\n        ----------\n        wcs : `~astropy.wcs.WCS`\n            The WCS with naxis to be chopped to naxis-1\n        dropax : int\n            The index of the WCS to drop, counting from 0 (i.e., python convention,\n            not FITS convention)\n\n        Returns\n        -------\n        `~astropy.wcs.WCS`\n            A new `~astropy.wcs.WCS` instance with one axis fewer\n        '
        inds = list(range(self.wcs.naxis))
        inds.pop(dropax)
        return self.sub([i + 1 for i in inds])

    def swapaxes(self, ax0, ax1):
        if False:
            return 10
        '\n        Swap axes in a WCS.\n\n        Parameters\n        ----------\n        wcs : `~astropy.wcs.WCS`\n            The WCS to have its axes swapped\n        ax0 : int\n        ax1 : int\n            The indices of the WCS to be swapped, counting from 0 (i.e., python\n            convention, not FITS convention)\n\n        Returns\n        -------\n        `~astropy.wcs.WCS`\n            A new `~astropy.wcs.WCS` instance with the same number of axes,\n            but two swapped\n        '
        inds = list(range(self.wcs.naxis))
        (inds[ax0], inds[ax1]) = (inds[ax1], inds[ax0])
        return self.sub([i + 1 for i in inds])

    def reorient_celestial_first(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Reorient the WCS such that the celestial axes are first, followed by\n        the spectral axis, followed by any others.\n        Assumes at least celestial axes are present.\n        '
        return self.sub([WCSSUB_CELESTIAL, WCSSUB_SPECTRAL, WCSSUB_STOKES, WCSSUB_TIME])

    def slice(self, view, numpy_order=True):
        if False:
            return 10
        '\n        Slice a WCS instance using a Numpy slice. The order of the slice should\n        be reversed (as for the data) compared to the natural WCS order.\n\n        Parameters\n        ----------\n        view : tuple\n            A tuple containing the same number of slices as the WCS system.\n            The ``step`` method, the third argument to a slice, is not\n            presently supported.\n        numpy_order : bool\n            Use numpy order, i.e. slice the WCS so that an identical slice\n            applied to a numpy array will slice the array and WCS in the same\n            way. If set to `False`, the WCS will be sliced in FITS order,\n            meaning the first slice will be applied to the *last* numpy index\n            but the *first* WCS axis.\n\n        Returns\n        -------\n        wcs_new : `~astropy.wcs.WCS`\n            A new resampled WCS axis\n        '
        if hasattr(view, '__len__') and len(view) > self.wcs.naxis:
            raise ValueError('Must have # of slices <= # of WCS axes')
        elif not hasattr(view, '__len__'):
            view = [view]
        if not all((isinstance(x, slice) for x in view)):
            return SlicedFITSWCS(self, view)
        wcs_new = self.deepcopy()
        if wcs_new.sip is not None:
            sip_crpix = wcs_new.sip.crpix.tolist()
        for (i, iview) in enumerate(view):
            if iview.step is not None and iview.step < 0:
                raise NotImplementedError('Reversing an axis is not implemented.')
            if numpy_order:
                wcs_index = self.wcs.naxis - 1 - i
            else:
                wcs_index = i
            if iview.step is not None and iview.start is None:
                iview = slice(0, iview.stop, iview.step)
            if iview.start is not None:
                if iview.step not in (None, 1):
                    crpix = self.wcs.crpix[wcs_index]
                    cdelt = self.wcs.cdelt[wcs_index]
                    crp = (crpix - iview.start - 1.0) / iview.step + 0.5 + 1.0 / iview.step / 2.0
                    wcs_new.wcs.crpix[wcs_index] = crp
                    if wcs_new.sip is not None:
                        sip_crpix[wcs_index] = crp
                    wcs_new.wcs.cdelt[wcs_index] = cdelt * iview.step
                else:
                    wcs_new.wcs.crpix[wcs_index] -= iview.start
                    if wcs_new.sip is not None:
                        sip_crpix[wcs_index] -= iview.start
            try:
                nitems = len(builtins.range(self._naxis[wcs_index])[iview])
            except TypeError as exc:
                if 'indices must be integers' not in str(exc):
                    raise
                warnings.warn(f"NAXIS{wcs_index} attribute is not updated because at least one index ('{iview}') is no integer.", AstropyUserWarning)
            else:
                wcs_new._naxis[wcs_index] = nitems
        if wcs_new.sip is not None:
            wcs_new.sip = Sip(self.sip.a, self.sip.b, self.sip.ap, self.sip.bp, sip_crpix)
        return wcs_new

    def __getitem__(self, item):
        if False:
            while True:
                i = 10
        return self.slice(item)

    def __iter__(self):
        if False:
            print('Hello World!')
        raise TypeError(f"'{self.__class__.__name__}' object is not iterable")

    @property
    def axis_type_names(self):
        if False:
            return 10
        '\n        World names for each coordinate axis.\n\n        Returns\n        -------\n        list of str\n            A list of names along each axis.\n        '
        names = list(self.wcs.cname)
        types = self.wcs.ctype
        for i in range(len(names)):
            if len(names[i]) > 0:
                continue
            names[i] = types[i].split('-')[0]
        return names

    @property
    def celestial(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A copy of the current WCS with only the celestial axes included.\n        '
        return self.sub([WCSSUB_CELESTIAL])

    @property
    def is_celestial(self):
        if False:
            for i in range(10):
                print('nop')
        return self.has_celestial and self.naxis == 2

    @property
    def has_celestial(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.wcs.lng >= 0 and self.wcs.lat >= 0
        except InconsistentAxisTypesError:
            return False

    @property
    def spectral(self):
        if False:
            i = 10
            return i + 15
        '\n        A copy of the current WCS with only the spectral axes included.\n        '
        return self.sub([WCSSUB_SPECTRAL])

    @property
    def is_spectral(self):
        if False:
            while True:
                i = 10
        return self.has_spectral and self.naxis == 1

    @property
    def has_spectral(self):
        if False:
            i = 10
            return i + 15
        try:
            return self.wcs.spec >= 0
        except InconsistentAxisTypesError:
            return False

    @property
    def temporal(self):
        if False:
            i = 10
            return i + 15
        '\n        A copy of the current WCS with only the time axes included.\n        '
        if not _WCSSUB_TIME_SUPPORT:
            raise NotImplementedError(f"Support for 'temporal' axis requires WCSLIB version 7.8 or greater but linked WCSLIB version is {_wcs.__version__}")
        return self.sub([WCSSUB_TIME])

    @property
    def is_temporal(self):
        if False:
            while True:
                i = 10
        return self.has_temporal and self.naxis == 1

    @property
    def has_temporal(self):
        if False:
            while True:
                i = 10
        return any((t // 1000 == 4 for t in self.wcs.axis_types))

    @property
    def has_distortion(self):
        if False:
            return 10
        '\n        Returns `True` if any distortion terms are present.\n        '
        return self.sip is not None or self.cpdis1 is not None or self.cpdis2 is not None or (self.det2im1 is not None and self.det2im2 is not None)

    @property
    def pixel_scale_matrix(self):
        if False:
            while True:
                i = 10
        try:
            cdelt = np.diag(self.wcs.get_cdelt())
            pc = self.wcs.get_pc()
        except InconsistentAxisTypesError:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', 'cdelt will be ignored since cd is present', RuntimeWarning)
                    cdelt = np.dot(self.wcs.cd, np.diag(self.wcs.cdelt))
            except AttributeError:
                cdelt = np.diag(self.wcs.cdelt)
            try:
                pc = self.wcs.pc
            except AttributeError:
                pc = 1
        pccd = np.dot(cdelt, pc)
        return pccd

    def footprint_contains(self, coord, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determines if a given SkyCoord is contained in the wcs footprint.\n\n        Parameters\n        ----------\n        coord : `~astropy.coordinates.SkyCoord`\n            The coordinate to check if it is within the wcs coordinate.\n        **kwargs :\n           Additional arguments to pass to `~astropy.coordinates.SkyCoord.to_pixel`\n\n        Returns\n        -------\n        response : bool\n           True means the WCS footprint contains the coordinate, False means it does not.\n        '
        return coord.contained_by(self, **kwargs)

def __WCS_unpickle__(cls, dct, fits_data):
    if False:
        print('Hello World!')
    '\n    Unpickles a WCS object from a serialized FITS string.\n    '
    self = cls.__new__(cls)
    buffer = io.BytesIO(fits_data)
    hdulist = fits.open(buffer)
    naxis = dct.pop('naxis', None)
    if naxis:
        hdulist[0].header['naxis'] = naxis
        naxes = dct.pop('_naxis', [])
        for (k, na) in enumerate(naxes):
            hdulist[0].header[f'naxis{k + 1:d}'] = na
    kwargs = dct.pop('_init_kwargs', {})
    self.__dict__.update(dct)
    wcskey = dct.pop('_alt_wcskey', ' ')
    WCS.__init__(self, hdulist[0].header, hdulist, key=wcskey, **kwargs)
    self.pixel_bounds = dct.get('_pixel_bounds', None)
    return self

def find_all_wcs(header, relax=True, keysel=None, fix=True, translate_units='', _do_set=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find all the WCS transformations in the given header.\n\n    Parameters\n    ----------\n    header : str or `~astropy.io.fits.Header` object.\n\n    relax : bool or int, optional\n        Degree of permissiveness:\n\n        - `True` (default): Admit all recognized informal extensions of the\n          WCS standard.\n\n        - `False`: Recognize only FITS keywords defined by the\n          published WCS standard.\n\n        - `int`: a bit field selecting specific extensions to accept.\n          See :ref:`astropy:relaxread` for details.\n\n    keysel : sequence of str, optional\n        A list of flags used to select the keyword types considered by\n        wcslib.  When ``None``, only the standard image header\n        keywords are considered (and the underlying wcspih() C\n        function is called).  To use binary table image array or pixel\n        list keywords, *keysel* must be set.\n\n        Each element in the list should be one of the following strings:\n\n            - 'image': Image header keywords\n\n            - 'binary': Binary table image array keywords\n\n            - 'pixel': Pixel list keywords\n\n        Keywords such as ``EQUIna`` or ``RFRQna`` that are common to\n        binary table image arrays and pixel lists (including\n        ``WCSNna`` and ``TWCSna``) are selected by both 'binary' and\n        'pixel'.\n\n    fix : bool, optional\n        When `True` (default), call `~astropy.wcs.Wcsprm.fix` on\n        the resulting objects to fix any non-standard uses in the\n        header.  `FITSFixedWarning` warnings will be emitted if any\n        changes were made.\n\n    translate_units : str, optional\n        Specify which potentially unsafe translations of non-standard\n        unit strings to perform.  By default, performs none.  See\n        `WCS.fix` for more information about this parameter.  Only\n        effective when ``fix`` is `True`.\n\n    Returns\n    -------\n    wcses : list of `WCS`\n    "
    if isinstance(header, (str, bytes)):
        header_string = header
    elif isinstance(header, fits.Header):
        header_string = header.tostring()
    else:
        raise TypeError('header must be a string or astropy.io.fits.Header object')
    keysel_flags = _parse_keysel(keysel)
    if isinstance(header_string, str):
        header_bytes = header_string.encode('ascii')
    else:
        header_bytes = header_string
    wcsprms = _wcs.find_all_wcs(header_bytes, relax, keysel_flags)
    result = []
    for wcsprm in wcsprms:
        subresult = WCS(fix=False, _do_set=False)
        subresult.wcs = wcsprm
        result.append(subresult)
        if fix:
            subresult.fix(translate_units)
        if _do_set:
            subresult.wcs.set()
    return result

def validate(source):
    if False:
        for i in range(10):
            print('nop')
    '\n    Prints a WCS validation report for the given FITS file.\n\n    Parameters\n    ----------\n    source : str or file-like or `~astropy.io.fits.HDUList`\n        The FITS file to validate.\n\n    Returns\n    -------\n    results : list subclass instance\n        The result is returned as nested lists.  The first level\n        corresponds to the HDUs in the given file.  The next level has\n        an entry for each WCS found in that header.  The special\n        subclass of list will pretty-print the results as a table when\n        printed.\n\n    '

    class _WcsValidateWcsResult(list):

        def __init__(self, key):
            if False:
                print('Hello World!')
            self._key = key

        def __repr__(self):
            if False:
                print('Hello World!')
            result = [f"  WCS key '{self._key or ' '}':"]
            if len(self):
                for entry in self:
                    for (i, line) in enumerate(entry.splitlines()):
                        if i == 0:
                            initial_indent = '    - '
                        else:
                            initial_indent = '      '
                        result.extend(textwrap.wrap(line, initial_indent=initial_indent, subsequent_indent='      '))
            else:
                result.append('    No issues.')
            return '\n'.join(result)

    class _WcsValidateHduResult(list):

        def __init__(self, hdu_index, hdu_name):
            if False:
                return 10
            self._hdu_index = hdu_index
            self._hdu_name = hdu_name
            list.__init__(self)

        def __repr__(self):
            if False:
                return 10
            if len(self):
                if self._hdu_name:
                    hdu_name = f' ({self._hdu_name})'
                else:
                    hdu_name = ''
                result = [f'HDU {self._hdu_index}{hdu_name}:']
                for wcs in self:
                    result.append(repr(wcs))
                return '\n'.join(result)
            return ''

    class _WcsValidateResults(list):

        def __repr__(self):
            if False:
                print('Hello World!')
            result = []
            for hdu in self:
                content = repr(hdu)
                if len(content):
                    result.append(content)
            return '\n\n'.join(result)
    global __warningregistry__
    if isinstance(source, fits.HDUList):
        hdulist = source
        close_file = False
    else:
        hdulist = fits.open(source)
        close_file = True
    results = _WcsValidateResults()
    for (i, hdu) in enumerate(hdulist):
        hdu_results = _WcsValidateHduResult(i, hdu.name)
        results.append(hdu_results)
        with warnings.catch_warnings(record=True) as warning_lines:
            wcses = find_all_wcs(hdu.header, relax=_wcs.WCSHDR_reject, fix=False, _do_set=False)
        for wcs in wcses:
            wcs_results = _WcsValidateWcsResult(wcs.wcs.alt)
            hdu_results.append(wcs_results)
            try:
                del __warningregistry__
            except NameError:
                pass
            with warnings.catch_warnings(record=True) as warning_lines:
                warnings.resetwarnings()
                warnings.simplefilter('always', FITSFixedWarning, append=True)
                try:
                    WCS(hdu.header, hdulist, key=wcs.wcs.alt or ' ', relax=_wcs.WCSHDR_reject, fix=True, _do_set=False)
                except WcsError as e:
                    wcs_results.append(str(e))
                wcs_results.extend([str(x.message) for x in warning_lines])
    if close_file:
        hdulist.close()
    return results