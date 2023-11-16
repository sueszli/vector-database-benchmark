import gzip
import itertools
import os
import re
import shutil
import sys
import warnings
import numpy as np
from astropy.io.fits.file import FILE_MODES, _File
from astropy.io.fits.header import _pad_length
from astropy.io.fits.util import _free_space_check, _get_array_mmap, _is_int, _tmp_name, fileobj_closed, fileobj_mode, ignore_sigint, isfile
from astropy.io.fits.verify import VerifyError, VerifyWarning, _ErrList, _Verify
from astropy.utils import indent
from astropy.utils.compat.optional_deps import HAS_BZ2
from astropy.utils.exceptions import AstropyUserWarning
from .base import ExtensionHDU, _BaseHDU, _NonstandardHDU, _ValidHDU
from .compressed import compressed
from .groups import GroupsHDU
from .image import ImageHDU, PrimaryHDU
if HAS_BZ2:
    import bz2
__all__ = ['HDUList', 'fitsopen']
FITS_SIGNATURE = b'SIMPLE  =                    T'

def fitsopen(name, mode='readonly', memmap=None, save_backup=False, cache=True, lazy_load_hdus=None, ignore_missing_simple=False, *, use_fsspec=None, fsspec_kwargs=None, decompress_in_memory=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Factory function to open a FITS file and return an `HDUList` object.\n\n    Parameters\n    ----------\n    name : str, file-like or `pathlib.Path`\n        File to be opened.\n\n    mode : str, optional\n        Open mode, \'readonly\', \'update\', \'append\', \'denywrite\', or\n        \'ostream\'. Default is \'readonly\'.\n\n        If ``name`` is a file object that is already opened, ``mode`` must\n        match the mode the file was opened with, readonly (rb), update (rb+),\n        append (ab+), ostream (w), denywrite (rb)).\n\n    memmap : bool, optional\n        Is memory mapping to be used? This value is obtained from the\n        configuration item ``astropy.io.fits.Conf.use_memmap``.\n        Default is `True`.\n\n    save_backup : bool, optional\n        If the file was opened in update or append mode, this ensures that\n        a backup of the original file is saved before any changes are flushed.\n        The backup has the same name as the original file with ".bak" appended.\n        If "file.bak" already exists then "file.bak.1" is used, and so on.\n        Default is `False`.\n\n    cache : bool, optional\n        If the file name is a URL, `~astropy.utils.data.download_file` is used\n        to open the file.  This specifies whether or not to save the file\n        locally in Astropy\'s download cache. Default is `True`.\n\n    lazy_load_hdus : bool, optional\n        To avoid reading all the HDUs and headers in a FITS file immediately\n        upon opening.  This is an optimization especially useful for large\n        files, as FITS has no way of determining the number and offsets of all\n        the HDUs in a file without scanning through the file and reading all\n        the headers. Default is `True`.\n\n        To disable lazy loading and read all HDUs immediately (the old\n        behavior) use ``lazy_load_hdus=False``.  This can lead to fewer\n        surprises--for example with lazy loading enabled, ``len(hdul)``\n        can be slow, as it means the entire FITS file needs to be read in\n        order to determine the number of HDUs.  ``lazy_load_hdus=False``\n        ensures that all HDUs have already been loaded after the file has\n        been opened.\n\n        .. versionadded:: 1.3\n\n    uint : bool, optional\n        Interpret signed integer data where ``BZERO`` is the central value and\n        ``BSCALE == 1`` as unsigned integer data.  For example, ``int16`` data\n        with ``BZERO = 32768`` and ``BSCALE = 1`` would be treated as\n        ``uint16`` data. Default is `True` so that the pseudo-unsigned\n        integer convention is assumed.\n\n    ignore_missing_end : bool, optional\n        Do not raise an exception when opening a file that is missing an\n        ``END`` card in the last header. Default is `False`.\n\n    ignore_missing_simple : bool, optional\n        Do not raise an exception when the SIMPLE keyword is missing. Note\n        that io.fits will raise a warning if a SIMPLE card is present but\n        written in a way that does not follow the FITS Standard.\n        Default is `False`.\n\n        .. versionadded:: 4.2\n\n    checksum : bool, str, optional\n        If `True`, verifies that both ``DATASUM`` and ``CHECKSUM`` card values\n        (when present in the HDU header) match the header and data of all HDU\'s\n        in the file.  Updates to a file that already has a checksum will\n        preserve and update the existing checksums unless this argument is\n        given a value of \'remove\', in which case the CHECKSUM and DATASUM\n        values are not checked, and are removed when saving changes to the\n        file. Default is `False`.\n\n    disable_image_compression : bool, optional\n        If `True`, treats compressed image HDU\'s like normal binary table\n        HDU\'s.  Default is `False`.\n\n    do_not_scale_image_data : bool, optional\n        If `True`, image data is not scaled using BSCALE/BZERO values\n        when read.  Default is `False`.\n\n    character_as_bytes : bool, optional\n        Whether to return bytes for string columns, otherwise unicode strings\n        are returned, but this does not respect memory mapping and loads the\n        whole column in memory when accessed. Default is `False`.\n\n    ignore_blank : bool, optional\n        If `True`, the BLANK keyword is ignored if present.\n        Default is `False`.\n\n    scale_back : bool, optional\n        If `True`, when saving changes to a file that contained scaled image\n        data, restore the data to the original type and reapply the original\n        BSCALE/BZERO values. This could lead to loss of accuracy if scaling\n        back to integer values after performing floating point operations on\n        the data. Default is `False`.\n\n    output_verify : str\n        Output verification option.  Must be one of ``"fix"``,\n        ``"silentfix"``, ``"ignore"``, ``"warn"``, or\n        ``"exception"``.  May also be any combination of ``"fix"`` or\n        ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"\n        (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.\n\n    use_fsspec : bool, optional\n        Use `fsspec.open` to open the file? Defaults to `False` unless\n        ``name`` starts with the Amazon S3 storage prefix ``s3://`` or the\n        Google Cloud Storage prefix ``gs://``.  Can also be used for paths\n        with other prefixes (e.g., ``http://``) but in this case you must\n        explicitly pass ``use_fsspec=True``.\n        Use of this feature requires the optional ``fsspec`` package.\n        A ``ModuleNotFoundError`` will be raised if the dependency is missing.\n\n        .. versionadded:: 5.2\n\n    fsspec_kwargs : dict, optional\n        Keyword arguments passed on to `fsspec.open`. This can be used to\n        configure cloud storage credentials and caching behavior.\n        For example, pass ``fsspec_kwargs={"anon": True}`` to enable\n        anonymous access to Amazon S3 open data buckets.\n        See ``fsspec``\'s documentation for available parameters.\n\n        .. versionadded:: 5.2\n\n    decompress_in_memory : bool, optional\n        By default files are decompressed progressively depending on what data\n        is needed.  This is good for memory usage, avoiding decompression of\n        the whole file, but it can be slow. With decompress_in_memory=True it\n        is possible to decompress instead the whole file in memory.\n\n        .. versionadded:: 6.0\n\n    Returns\n    -------\n    hdulist : `HDUList`\n        `HDUList` containing all of the header data units in the file.\n\n    '
    from astropy.io.fits import conf
    if memmap is None:
        memmap = None if conf.use_memmap else False
    else:
        memmap = bool(memmap)
    if lazy_load_hdus is None:
        lazy_load_hdus = conf.lazy_load_hdus
    else:
        lazy_load_hdus = bool(lazy_load_hdus)
    if 'uint' not in kwargs:
        kwargs['uint'] = conf.enable_uint
    if not name:
        raise ValueError(f'Empty filename: {name!r}')
    return HDUList.fromfile(name, mode, memmap, save_backup, cache, lazy_load_hdus, ignore_missing_simple, use_fsspec=use_fsspec, fsspec_kwargs=fsspec_kwargs, decompress_in_memory=decompress_in_memory, **kwargs)

class HDUList(list, _Verify):
    """
    HDU list class.  This is the top-level FITS object.  When a FITS
    file is opened, a `HDUList` object is returned.
    """

    def __init__(self, hdus=[], file=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a `HDUList` object.\n\n        Parameters\n        ----------\n        hdus : BaseHDU or sequence thereof, optional\n            The HDU object(s) to comprise the `HDUList`.  Should be\n            instances of HDU classes like `ImageHDU` or `BinTableHDU`.\n\n        file : file-like, bytes, optional\n            The opened physical file associated with the `HDUList`\n            or a bytes object containing the contents of the FITS\n            file.\n        '
        if isinstance(file, bytes):
            self._data = file
            self._file = None
        else:
            self._file = file
            self._data = None
        self._open_kwargs = {}
        self._in_read_next_hdu = False
        if file is None:
            self._read_all = True
        elif self._file is not None:
            self._read_all = self._file.mode == 'ostream'
        else:
            self._read_all = False
        if hdus is None:
            hdus = []
        if isinstance(hdus, _ValidHDU):
            hdus = [hdus]
        elif not isinstance(hdus, (HDUList, list)):
            raise TypeError('Invalid input for HDUList.')
        for (idx, hdu) in enumerate(hdus):
            if not isinstance(hdu, _BaseHDU):
                raise TypeError(f'Element {idx} in the HDUList input is not an HDU.')
        super().__init__(hdus)
        if file is None:
            self.update_extend()

    def __len__(self):
        if False:
            return 10
        if not self._in_read_next_hdu:
            self.readall()
        return super().__len__()

    def __repr__(self):
        if False:
            print('Hello World!')
        is_fsspec_file = self._file and 'fsspec' in str(self._file._file.__class__.__bases__)
        if not self._read_all and is_fsspec_file:
            return f'{type(self)} (partially read)'
        self.readall()
        return super().__repr__()

    def __iter__(self):
        if False:
            print('Hello World!')
        for idx in itertools.count():
            try:
                yield self[idx]
            except IndexError:
                break

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        '\n        Get an HDU from the `HDUList`, indexed by number or name.\n        '
        if isinstance(key, slice):
            max_idx = key.stop
            if max_idx is None:
                max_idx = len(self)
            max_idx = self._positive_index_of(max_idx)
            number_loaded = super().__len__()
            if max_idx >= number_loaded:
                for i in range(number_loaded, max_idx):
                    if not self._read_next_hdu():
                        break
            try:
                hdus = super().__getitem__(key)
            except IndexError as e:
                if self._read_all:
                    raise e
                else:
                    raise IndexError('HDU not found, possibly because the index is out of range, or because the file was closed before all HDUs were read')
            else:
                return HDUList(hdus)
        try:
            return self._try_while_unread_hdus(super().__getitem__, self._positive_index_of(key))
        except IndexError as e:
            if self._read_all:
                raise e
            else:
                raise IndexError('HDU not found, possibly because the index is out of range, or because the file was closed before all HDUs were read')

    def __contains__(self, item):
        if False:
            while True:
                i = 10
        '\n        Returns `True` if ``item`` is an ``HDU`` _in_ ``self`` or a valid\n        extension specification (e.g., integer extension number, extension\n        name, or a tuple of extension name and an extension version)\n        of a ``HDU`` in ``self``.\n\n        '
        try:
            self._try_while_unread_hdus(self.index_of, item)
        except (KeyError, ValueError):
            return False
        return True

    def __setitem__(self, key, hdu):
        if False:
            print('Hello World!')
        '\n        Set an HDU to the `HDUList`, indexed by number or name.\n        '
        _key = self._positive_index_of(key)
        if isinstance(hdu, (slice, list)):
            if _is_int(_key):
                raise ValueError('An element in the HDUList must be an HDU.')
            for item in hdu:
                if not isinstance(item, _BaseHDU):
                    raise ValueError(f'{item} is not an HDU.')
        elif not isinstance(hdu, _BaseHDU):
            raise ValueError(f'{hdu} is not an HDU.')
        try:
            self._try_while_unread_hdus(super().__setitem__, _key, hdu)
        except IndexError:
            raise IndexError(f'Extension {key} is out of bound or not found.')
        self._resize = True
        self._truncate = False

    def __delitem__(self, key):
        if False:
            return 10
        '\n        Delete an HDU from the `HDUList`, indexed by number or name.\n        '
        if isinstance(key, slice):
            end_index = len(self)
        else:
            key = self._positive_index_of(key)
            end_index = len(self) - 1
        self._try_while_unread_hdus(super().__delitem__, key)
        if key == end_index or (key == -1 and (not self._resize)):
            self._truncate = True
        else:
            self._truncate = False
            self._resize = True

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        output_verify = self._open_kwargs.get('output_verify', 'exception')
        self.close(output_verify=output_verify)

    @classmethod
    def fromfile(cls, fileobj, mode=None, memmap=None, save_backup=False, cache=True, lazy_load_hdus=True, ignore_missing_simple=False, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Creates an `HDUList` instance from a file-like object.\n\n        The actual implementation of ``fitsopen()``, and generally shouldn't\n        be used directly.  Use :func:`open` instead (and see its\n        documentation for details of the parameters accepted by this method).\n        "
        return cls._readfrom(fileobj=fileobj, mode=mode, memmap=memmap, save_backup=save_backup, cache=cache, ignore_missing_simple=ignore_missing_simple, lazy_load_hdus=lazy_load_hdus, **kwargs)

    @classmethod
    def fromstring(cls, data, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Creates an `HDUList` instance from a string or other in-memory data\n        buffer containing an entire FITS file.  Similar to\n        :meth:`HDUList.fromfile`, but does not accept the mode or memmap\n        arguments, as they are only relevant to reading from a file on disk.\n\n        This is useful for interfacing with other libraries such as CFITSIO,\n        and may also be useful for streaming applications.\n\n        Parameters\n        ----------\n        data : str, buffer-like, etc.\n            A string or other memory buffer containing an entire FITS file.\n            Buffer-like objects include :class:`~bytes`, :class:`~bytearray`,\n            :class:`~memoryview`, and :class:`~numpy.ndarray`.\n            It should be noted that if that memory is read-only (such as a\n            Python string) the returned :class:`HDUList`'s data portions will\n            also be read-only.\n        **kwargs : dict\n            Optional keyword arguments.  See\n            :func:`astropy.io.fits.open` for details.\n\n        Returns\n        -------\n        hdul : HDUList\n            An :class:`HDUList` object representing the in-memory FITS file.\n        "
        try:
            np.ndarray((), dtype='ubyte', buffer=data)
        except TypeError:
            raise TypeError(f'The provided object {data} does not contain an underlying memory buffer.  fromstring() requires an object that supports the buffer interface such as bytes, buffer, memoryview, ndarray, etc.  This restriction is to ensure that efficient access to the array/table data is possible.')
        return cls._readfrom(data=data, **kwargs)

    def fileinfo(self, index):
        if False:
            print('Hello World!')
        '\n        Returns a dictionary detailing information about the locations\n        of the indexed HDU within any associated file.  The values are\n        only valid after a read or write of the associated file with\n        no intervening changes to the `HDUList`.\n\n        Parameters\n        ----------\n        index : int\n            Index of HDU for which info is to be returned.\n\n        Returns\n        -------\n        fileinfo : dict or None\n\n            The dictionary details information about the locations of\n            the indexed HDU within an associated file.  Returns `None`\n            when the HDU is not associated with a file.\n\n            Dictionary contents:\n\n            ========== ========================================================\n            Key        Value\n            ========== ========================================================\n            file       File object associated with the HDU\n            filename   Name of associated file object\n            filemode   Mode in which the file was opened (readonly,\n                       update, append, denywrite, ostream)\n            resized    Flag that when `True` indicates that the data has been\n                       resized since the last read/write so the returned values\n                       may not be valid.\n            hdrLoc     Starting byte location of header in file\n            datLoc     Starting byte location of data block in file\n            datSpan    Data size including padding\n            ========== ========================================================\n\n        '
        if self._file is not None:
            output = self[index].fileinfo()
            if not output:
                f = None
                for hdu in self:
                    info = hdu.fileinfo()
                    if info:
                        f = info['file']
                        fm = info['filemode']
                        break
                output = {'file': f, 'filemode': fm, 'hdrLoc': None, 'datLoc': None, 'datSpan': None}
            output['filename'] = self._file.name
            output['resized'] = self._wasresized()
        else:
            output = None
        return output

    def __copy__(self):
        if False:
            return 10
        '\n        Return a shallow copy of an HDUList.\n\n        Returns\n        -------\n        copy : `HDUList`\n            A shallow copy of this `HDUList` object.\n\n        '
        return self[:]
    copy = __copy__

    def __deepcopy__(self, memo=None):
        if False:
            return 10
        return HDUList([hdu.copy() for hdu in self])

    def pop(self, index=-1):
        if False:
            i = 10
            return i + 15
        "Remove an item from the list and return it.\n\n        Parameters\n        ----------\n        index : int, str, tuple of (string, int), optional\n            An integer value of ``index`` indicates the position from which\n            ``pop()`` removes and returns an HDU. A string value or a tuple\n            of ``(string, int)`` functions as a key for identifying the\n            HDU to be removed and returned. If ``key`` is a tuple, it is\n            of the form ``(key, ver)`` where ``ver`` is an ``EXTVER``\n            value that must match the HDU being searched for.\n\n            If the key is ambiguous (e.g. there are multiple 'SCI' extensions)\n            the first match is returned.  For a more precise match use the\n            ``(name, ver)`` pair.\n\n            If even the ``(name, ver)`` pair is ambiguous the numeric index\n            must be used to index the duplicate HDU.\n\n        Returns\n        -------\n        hdu : BaseHDU\n            The HDU object at position indicated by ``index`` or having name\n            and version specified by ``index``.\n        "
        self.readall()
        list_index = self.index_of(index)
        return super().pop(list_index)

    def insert(self, index, hdu):
        if False:
            i = 10
            return i + 15
        '\n        Insert an HDU into the `HDUList` at the given ``index``.\n\n        Parameters\n        ----------\n        index : int\n            Index before which to insert the new HDU.\n\n        hdu : BaseHDU\n            The HDU object to insert\n        '
        if not isinstance(hdu, _BaseHDU):
            raise ValueError(f'{hdu} is not an HDU.')
        num_hdus = len(self)
        if index == 0 or num_hdus == 0:
            if num_hdus != 0:
                if isinstance(self[0], GroupsHDU):
                    raise ValueError("The current Primary HDU is a GroupsHDU.  It can't be made into an extension HDU, so another HDU cannot be inserted before it.")
                hdu1 = ImageHDU(self[0].data, self[0].header)
                super().insert(1, hdu1)
                super().__delitem__(0)
            if not isinstance(hdu, (PrimaryHDU, _NonstandardHDU)):
                if isinstance(hdu, ImageHDU):
                    hdu = PrimaryHDU(hdu.data, hdu.header)
                else:
                    phdu = PrimaryHDU()
                    super().insert(0, phdu)
                    index = 1
        else:
            if isinstance(hdu, GroupsHDU):
                raise ValueError('A GroupsHDU must be inserted as a Primary HDU.')
            if isinstance(hdu, PrimaryHDU):
                hdu = ImageHDU(hdu.data, hdu.header)
        super().insert(index, hdu)
        hdu._new = True
        self._resize = True
        self._truncate = False
        self.update_extend()

    def append(self, hdu):
        if False:
            for i in range(10):
                print('nop')
        '\n        Append a new HDU to the `HDUList`.\n\n        Parameters\n        ----------\n        hdu : BaseHDU\n            HDU to add to the `HDUList`.\n        '
        if not isinstance(hdu, _BaseHDU):
            raise ValueError('HDUList can only append an HDU.')
        if len(self) > 0:
            if isinstance(hdu, GroupsHDU):
                raise ValueError("Can't append a GroupsHDU to a non-empty HDUList")
            if isinstance(hdu, PrimaryHDU):
                hdu = ImageHDU(hdu.data, hdu.header)
        elif not isinstance(hdu, (PrimaryHDU, _NonstandardHDU)):
            if isinstance(hdu, ImageHDU):
                hdu = PrimaryHDU(hdu.data, hdu.header)
            else:
                phdu = PrimaryHDU()
                super().append(phdu)
        super().append(hdu)
        hdu._new = True
        self._resize = True
        self._truncate = False
        self.update_extend()

    def index_of(self, key):
        if False:
            return 10
        "\n        Get the index of an HDU from the `HDUList`.\n\n        Parameters\n        ----------\n        key : int, str, tuple of (string, int) or BaseHDU\n            The key identifying the HDU.  If ``key`` is a tuple, it is of the\n            form ``(name, ver)`` where ``ver`` is an ``EXTVER`` value that must\n            match the HDU being searched for.\n\n            If the key is ambiguous (e.g. there are multiple 'SCI' extensions)\n            the first match is returned.  For a more precise match use the\n            ``(name, ver)`` pair.\n\n            If even the ``(name, ver)`` pair is ambiguous (it shouldn't be\n            but it's not impossible) the numeric index must be used to index\n            the duplicate HDU.\n\n            When ``key`` is an HDU object, this function returns the\n            index of that HDU object in the ``HDUList``.\n\n        Returns\n        -------\n        index : int\n            The index of the HDU in the `HDUList`.\n\n        Raises\n        ------\n        ValueError\n            If ``key`` is an HDU object and it is not found in the ``HDUList``.\n        KeyError\n            If an HDU specified by the ``key`` that is an extension number,\n            extension name, or a tuple of extension name and version is not\n            found in the ``HDUList``.\n\n        "
        if _is_int(key):
            return key
        elif isinstance(key, tuple):
            (_key, _ver) = key
        elif isinstance(key, _BaseHDU):
            return self.index(key)
        else:
            _key = key
            _ver = None
        if not isinstance(_key, str):
            raise KeyError('{} indices must be integers, extension names as strings, or (extname, version) tuples; got {}'.format(self.__class__.__name__, _key))
        _key = _key.strip().upper()
        found = None
        for (idx, hdu) in enumerate(self):
            name = hdu.name
            if isinstance(name, str):
                name = name.strip().upper()
            if (name == _key or (_key == 'PRIMARY' and idx == 0)) and (_ver is None or _ver == hdu.ver):
                found = idx
                break
        if found is None:
            raise KeyError(f'Extension {key!r} not found.')
        else:
            return found

    def _positive_index_of(self, key):
        if False:
            print('Hello World!')
        '\n        Same as index_of, but ensures always returning a positive index\n        or zero.\n\n        (Really this should be called non_negative_index_of but it felt\n        too long.)\n\n        This means that if the key is a negative integer, we have to\n        convert it to the corresponding positive index.  This means\n        knowing the length of the HDUList, which in turn means loading\n        all HDUs.  Therefore using negative indices on HDULists is inherently\n        inefficient.\n        '
        index = self.index_of(key)
        if index >= 0:
            return index
        if abs(index) > len(self):
            raise IndexError(f'Extension {index} is out of bound or not found.')
        return len(self) + index

    def readall(self):
        if False:
            print('Hello World!')
        '\n        Read data of all HDUs into memory.\n        '
        while self._read_next_hdu():
            pass

    @ignore_sigint
    def flush(self, output_verify='fix', verbose=False):
        if False:
            return 10
        '\n        Force a write of the `HDUList` back to the file (for append and\n        update modes only).\n\n        Parameters\n        ----------\n        output_verify : str\n            Output verification option.  Must be one of ``"fix"``,\n            ``"silentfix"``, ``"ignore"``, ``"warn"``, or\n            ``"exception"``.  May also be any combination of ``"fix"`` or\n            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"\n            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.\n\n        verbose : bool\n            When `True`, print verbose messages\n        '
        if self._file.mode not in ('append', 'update', 'ostream'):
            warnings.warn(f"Flush for '{self._file.mode}' mode is not supported.", AstropyUserWarning)
            return
        save_backup = self._open_kwargs.get('save_backup', False)
        if save_backup and self._file.mode in ('append', 'update'):
            filename = self._file.name
            if os.path.exists(filename):
                backup = filename + '.bak'
                idx = 1
                while os.path.exists(backup):
                    backup = filename + '.bak.' + str(idx)
                    idx += 1
                warnings.warn(f'Saving a backup of {filename} to {backup}.', AstropyUserWarning)
                try:
                    shutil.copy(filename, backup)
                except OSError as exc:
                    raise OSError(f'Failed to save backup to destination {filename}') from exc
        self.verify(option=output_verify)
        if self._file.mode in ('append', 'ostream'):
            for hdu in self:
                if verbose:
                    try:
                        extver = str(hdu._header['extver'])
                    except KeyError:
                        extver = ''
                if hdu._new:
                    hdu._prewriteto(checksum=hdu._output_checksum)
                    with _free_space_check(self):
                        hdu._writeto(self._file)
                        if verbose:
                            print('append HDU', hdu.name, extver)
                        hdu._new = False
                    hdu._postwriteto()
        elif self._file.mode == 'update':
            self._flush_update()

    def update_extend(self):
        if False:
            print('Hello World!')
        '\n        Make sure that if the primary header needs the keyword ``EXTEND`` that\n        it has it and it is correct.\n        '
        if not len(self):
            return
        if not isinstance(self[0], PrimaryHDU):
            return
        hdr = self[0].header

        def get_first_ext():
            if False:
                i = 10
                return i + 15
            try:
                return self[1]
            except IndexError:
                return None
        if 'EXTEND' in hdr:
            if not hdr['EXTEND'] and get_first_ext() is not None:
                hdr['EXTEND'] = True
        elif get_first_ext() is not None:
            if hdr['NAXIS'] == 0:
                hdr.set('EXTEND', True, after='NAXIS')
            else:
                n = hdr['NAXIS']
                hdr.set('EXTEND', True, after='NAXIS' + str(n))

    def writeto(self, fileobj, output_verify='exception', overwrite=False, checksum=False):
        if False:
            while True:
                i = 10
        '\n        Write the `HDUList` to a new file.\n\n        Parameters\n        ----------\n        fileobj : str, file-like or `pathlib.Path`\n            File to write to.  If a file object, must be opened in a\n            writeable mode.\n\n        output_verify : str\n            Output verification option.  Must be one of ``"fix"``,\n            ``"silentfix"``, ``"ignore"``, ``"warn"``, or\n            ``"exception"``.  May also be any combination of ``"fix"`` or\n            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"\n            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.\n\n        overwrite : bool, optional\n            If ``True``, overwrite the output file if it exists. Raises an\n            ``OSError`` if ``False`` and the output file exists. Default is\n            ``False``.\n\n        checksum : bool\n            When `True` adds both ``DATASUM`` and ``CHECKSUM`` cards\n            to the headers of all HDU\'s written to the file.\n        '
        if len(self) == 0:
            warnings.warn('There is nothing to write.', AstropyUserWarning)
            return
        self.verify(option=output_verify)
        self.update_extend()
        closed = isinstance(fileobj, str) or fileobj_closed(fileobj)
        mode = FILE_MODES[fileobj_mode(fileobj)] if isfile(fileobj) else 'ostream'
        fileobj = _File(fileobj, mode=mode, overwrite=overwrite)
        hdulist = self.fromfile(fileobj)
        try:
            dirname = os.path.dirname(hdulist._file.name)
        except (AttributeError, TypeError):
            dirname = None
        try:
            with _free_space_check(self, dirname=dirname):
                for hdu in self:
                    hdu._prewriteto(checksum=checksum)
                    hdu._writeto(hdulist._file)
                    hdu._postwriteto()
        finally:
            hdulist.close(output_verify=output_verify, closed=closed)

    def close(self, output_verify='exception', verbose=False, closed=True):
        if False:
            return 10
        '\n        Close the associated FITS file and memmap object, if any.\n\n        Parameters\n        ----------\n        output_verify : str\n            Output verification option.  Must be one of ``"fix"``,\n            ``"silentfix"``, ``"ignore"``, ``"warn"``, or\n            ``"exception"``.  May also be any combination of ``"fix"`` or\n            ``"silentfix"`` with ``"+ignore"``, ``+warn``, or ``+exception"\n            (e.g. ``"fix+warn"``).  See :ref:`astropy:verify` for more info.\n\n        verbose : bool\n            When `True`, print out verbose messages.\n\n        closed : bool\n            When `True`, close the underlying file object.\n        '
        try:
            if self._file and self._file.mode in ('append', 'update') and (not self._file.closed):
                self.flush(output_verify=output_verify, verbose=verbose)
        finally:
            if self._file and closed and hasattr(self._file, 'close'):
                self._file.close()
            for hdu in self:
                hdu._close(closed=closed)

    def info(self, output=None):
        if False:
            while True:
                i = 10
        '\n        Summarize the info of the HDUs in this `HDUList`.\n\n        Note that this function prints its results to the console---it\n        does not return a value.\n\n        Parameters\n        ----------\n        output : file-like or bool, optional\n            A file-like object to write the output to.  If `False`, does not\n            output to a file and instead returns a list of tuples representing\n            the HDU info.  Writes to ``sys.stdout`` by default.\n        '
        if output is None:
            output = sys.stdout
        if self._file is None:
            name = '(No file associated with this HDUList)'
        else:
            name = self._file.name
        results = [f'Filename: {name}', 'No.    Name      Ver    Type      Cards   Dimensions   Format']
        format = '{:3d}  {:10}  {:3} {:11}  {:5d}   {}   {}   {}'
        default = ('', '', '', 0, (), '', '')
        for (idx, hdu) in enumerate(self):
            summary = hdu._summary()
            if len(summary) < len(default):
                summary += default[len(summary):]
            summary = (idx,) + summary
            if output:
                results.append(format.format(*summary))
            else:
                results.append(summary)
        if output:
            output.write('\n'.join(results))
            output.write('\n')
            output.flush()
        else:
            return results[2:]

    def filename(self):
        if False:
            while True:
                i = 10
        '\n        Return the file name associated with the HDUList object if one exists.\n        Otherwise returns None.\n\n        Returns\n        -------\n        filename : str\n            A string containing the file name associated with the HDUList\n            object if an association exists.  Otherwise returns None.\n\n        '
        if self._file is not None:
            if hasattr(self._file, 'name'):
                return self._file.name
        return None

    @classmethod
    def _readfrom(cls, fileobj=None, data=None, mode=None, memmap=None, cache=True, lazy_load_hdus=True, ignore_missing_simple=False, *, use_fsspec=None, fsspec_kwargs=None, decompress_in_memory=False, **kwargs):
        if False:
            return 10
        '\n        Provides the implementations from HDUList.fromfile and\n        HDUList.fromstring, both of which wrap this method, as their\n        implementations are largely the same.\n        '
        if fileobj is not None:
            if not isinstance(fileobj, _File):
                fileobj = _File(fileobj, mode=mode, memmap=memmap, cache=cache, use_fsspec=use_fsspec, fsspec_kwargs=fsspec_kwargs, decompress_in_memory=decompress_in_memory)
            mode = fileobj.mode
            hdulist = cls(file=fileobj)
        else:
            if mode is None:
                mode = 'readonly'
            hdulist = cls(file=data)
        if not ignore_missing_simple and hdulist._file and (hdulist._file.mode != 'ostream') and (hdulist._file.size > 0):
            pos = hdulist._file.tell()
            simple = hdulist._file.read(80)
            match_sig = simple[:29] == FITS_SIGNATURE[:-1] and simple[29:30] in (b'T', b'F')
            if not match_sig:
                match_sig_relaxed = re.match(b'SIMPLE\\s*=\\s*[T|F]', simple)
                if match_sig_relaxed:
                    warnings.warn("Found a SIMPLE card but its format doesn't respect the FITS Standard", VerifyWarning)
                else:
                    if hdulist._file.close_on_error:
                        hdulist._file.close()
                    raise OSError('No SIMPLE card found, this file does not appear to be a valid FITS file. If this is really a FITS file, try with ignore_missing_simple=True')
            hdulist._file.seek(pos)
        hdulist._open_kwargs = kwargs
        if fileobj is not None and fileobj.writeonly:
            return hdulist
        read_one = hdulist._read_next_hdu()
        if not read_one and mode in ('readonly', 'denywrite'):
            if hdulist._file.close_on_error:
                hdulist._file.close()
            raise OSError('Empty or corrupt FITS file')
        if not lazy_load_hdus or kwargs.get('checksum') is True:
            while hdulist._read_next_hdu():
                pass
        hdulist._resize = False
        hdulist._truncate = False
        return hdulist

    def _try_while_unread_hdus(self, func, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Attempt an operation that accesses an HDU by index/name\n        that can fail if not all HDUs have been read yet.  Keep\n        reading HDUs until the operation succeeds or there are no\n        more HDUs to read.\n        '
        while True:
            try:
                return func(*args, **kwargs)
            except Exception:
                if self._read_next_hdu():
                    continue
                else:
                    raise

    def _read_next_hdu(self):
        if False:
            print('Hello World!')
        '\n        Lazily load a single HDU from the fileobj or data string the `HDUList`\n        was opened from, unless no further HDUs are found.\n\n        Returns True if a new HDU was loaded, or False otherwise.\n        '
        if self._read_all:
            return False
        saved_compression_enabled = compressed.COMPRESSION_ENABLED
        (fileobj, data, kwargs) = (self._file, self._data, self._open_kwargs)
        if fileobj is not None and fileobj.closed:
            return False
        try:
            self._in_read_next_hdu = True
            if 'disable_image_compression' in kwargs and kwargs['disable_image_compression']:
                compressed.COMPRESSION_ENABLED = False
            try:
                if fileobj is not None:
                    try:
                        if len(self) > 0:
                            last = self[len(self) - 1]
                            if last._data_offset is not None:
                                offset = last._data_offset + last._data_size
                                fileobj.seek(offset, os.SEEK_SET)
                        hdu = _BaseHDU.readfrom(fileobj, **kwargs)
                    except EOFError:
                        self._read_all = True
                        return False
                    except OSError:
                        if self._file.close_on_error:
                            self._file.close()
                        if fileobj.writeonly:
                            self._read_all = True
                            return False
                        else:
                            raise
                else:
                    if not data:
                        self._read_all = True
                        return False
                    hdu = _BaseHDU.fromstring(data, **kwargs)
                    self._data = data[hdu._data_offset + hdu._data_size:]
                super().append(hdu)
                if len(self) == 1:
                    self.update_extend()
                hdu._new = False
                if 'checksum' in kwargs:
                    hdu._output_checksum = kwargs['checksum']
            except (VerifyError, ValueError) as exc:
                warnings.warn(f'Error validating header for HDU #{len(self)} (note: Astropy uses zero-based indexing).\n{indent(str(exc))}\nThere may be extra bytes after the last HDU or the file is corrupted.', VerifyWarning)
                del exc
                self._read_all = True
                return False
        finally:
            compressed.COMPRESSION_ENABLED = saved_compression_enabled
            self._in_read_next_hdu = False
        return True

    def _verify(self, option='warn'):
        if False:
            print('Hello World!')
        errs = _ErrList([], unit='HDU')
        if len(self) > 0 and (not isinstance(self[0], PrimaryHDU)) and (not isinstance(self[0], _NonstandardHDU)):
            err_text = "HDUList's 0th element is not a primary HDU."
            fix_text = 'Fixed by inserting one as 0th HDU.'

            def fix(self=self):
                if False:
                    i = 10
                    return i + 15
                self.insert(0, PrimaryHDU())
            err = self.run_option(option, err_text=err_text, fix_text=fix_text, fix=fix)
            errs.append(err)
        if len(self) > 1 and ('EXTEND' not in self[0].header or self[0].header['EXTEND'] is not True):
            err_text = 'Primary HDU does not contain an EXTEND keyword equal to T even though there are extension HDUs.'
            fix_text = 'Fixed by inserting or updating the EXTEND keyword.'

            def fix(header=self[0].header):
                if False:
                    for i in range(10):
                        print('nop')
                naxis = header['NAXIS']
                if naxis == 0:
                    after = 'NAXIS'
                else:
                    after = 'NAXIS' + str(naxis)
                header.set('EXTEND', value=True, after=after)
            errs.append(self.run_option(option, err_text=err_text, fix_text=fix_text, fix=fix))
        for (idx, hdu) in enumerate(self):
            if idx > 0 and (not isinstance(hdu, ExtensionHDU)):
                err_text = f"HDUList's element {idx} is not an extension HDU."
                err = self.run_option(option, err_text=err_text, fixable=False)
                errs.append(err)
            else:
                result = hdu._verify(option)
                if result:
                    errs.append(result)
        return errs

    def _flush_update(self):
        if False:
            print('Hello World!')
        'Implements flushing changes to a file in update mode.'
        for hdu in self:
            hdu._prewriteto(checksum=hdu._output_checksum, inplace=True)
        try:
            self._wasresized()
            if self._resize or self._file.compression:
                self._flush_resize()
            else:
                for hdu in self:
                    hdu._writeto(self._file, inplace=True)
            for hdu in self:
                hdu._header._modified = False
        finally:
            for hdu in self:
                hdu._postwriteto()

    def _flush_resize(self):
        if False:
            while True:
                i = 10
        '\n        Implements flushing changes in update mode when parts of one or more HDU\n        need to be resized.\n        '
        old_name = self._file.name
        old_memmap = self._file.memmap
        name = _tmp_name(old_name)
        if not self._file.file_like:
            old_mode = os.stat(old_name).st_mode
            if self._file.compression == 'gzip':
                new_file = gzip.GzipFile(name, mode='ab+')
            elif self._file.compression == 'bzip2':
                if not HAS_BZ2:
                    raise ModuleNotFoundError('This Python installation does not provide the bz2 module.')
                new_file = bz2.BZ2File(name, mode='w')
            else:
                new_file = name
            with self.fromfile(new_file, mode='append') as hdulist:
                for hdu in self:
                    hdu._writeto(hdulist._file, inplace=True, copy=True)
                if sys.platform.startswith('win'):
                    mmaps = [(idx, _get_array_mmap(hdu.data), hdu.data) for (idx, hdu) in enumerate(self) if hdu._has_data]
                hdulist._file.close()
                self._file.close()
            if sys.platform.startswith('win'):
                for (idx, mmap, arr) in mmaps:
                    if mmap is not None:
                        mmap.close()
            os.remove(self._file.name)
            os.rename(name, old_name)
            os.chmod(old_name, old_mode)
            if isinstance(new_file, gzip.GzipFile):
                old_file = gzip.GzipFile(old_name, mode='rb+')
            else:
                old_file = old_name
            ffo = _File(old_file, mode='update', memmap=old_memmap)
            self._file = ffo
            for hdu in self:
                if hdu._has_data and _get_array_mmap(hdu.data) is not None:
                    del hdu.data
                hdu._file = ffo
            if sys.platform.startswith('win'):
                for (idx, mmap, arr) in mmaps:
                    if mmap is not None:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore', category=DeprecationWarning)
                            arr.data = self[idx].data.data
                del mmaps
        else:
            self.writeto(name)
            hdulist = self.fromfile(name)
            ffo = self._file
            ffo.truncate(0)
            ffo.seek(0)
            for hdu in hdulist:
                hdu._writeto(ffo, inplace=True, copy=True)
            hdulist.close()
            os.remove(hdulist._file.name)
        self._resize = False
        self._truncate = False
        for hdu in self:
            hdu._header._modified = False
            hdu._new = False
            hdu._file = ffo

    def _wasresized(self, verbose=False):
        if False:
            return 10
        '\n        Determine if any changes to the HDUList will require a file resize\n        when flushing the file.\n\n        Side effect of setting the objects _resize attribute.\n        '
        if not self._resize:
            for hdu in self:
                nbytes = len(str(hdu._header))
                if nbytes != hdu._data_offset - hdu._header_offset:
                    self._resize = True
                    self._truncate = False
                    if verbose:
                        print('One or more header is resized.')
                    break
                if not hdu._has_data:
                    continue
                nbytes = hdu.size
                nbytes = nbytes + _pad_length(nbytes)
                if nbytes != hdu._data_size:
                    self._resize = True
                    self._truncate = False
                    if verbose:
                        print('One or more data area is resized.')
                    break
            if self._truncate:
                try:
                    self._file.truncate(hdu._data_offset + hdu._data_size)
                except OSError:
                    self._resize = True
                self._truncate = False
        return self._resize