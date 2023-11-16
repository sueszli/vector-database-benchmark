from __future__ import annotations
import json
import os
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any
import numpy as np
from xarray import coding, conventions
from xarray.backends.common import BACKEND_ENTRYPOINTS, AbstractWritableDataStore, BackendArray, BackendEntrypoint, _encode_variable_name, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.core import indexing
from xarray.core.parallelcompat import guess_chunkmanager
from xarray.core.pycompat import integer_types
from xarray.core.utils import FrozenDict, HiddenKeyDict, close_on_error
from xarray.core.variable import Variable
if TYPE_CHECKING:
    from io import BufferedIOBase
    from xarray.backends.common import AbstractDataStore
    from xarray.core.dataset import Dataset
DIMENSION_KEY = '_ARRAY_DIMENSIONS'

def encode_zarr_attr_value(value):
    if False:
        while True:
            i = 10
    '\n    Encode a attribute value as something that can be serialized as json\n\n    Many xarray datasets / variables have numpy arrays and values. This\n    function handles encoding / decoding of such items.\n\n    ndarray -> list\n    scalar array -> scalar\n    other -> other (no change)\n    '
    if isinstance(value, np.ndarray):
        encoded = value.tolist()
    elif isinstance(value, np.generic):
        encoded = value.item()
    else:
        encoded = value
    return encoded

class ZarrArrayWrapper(BackendArray):
    __slots__ = ('datastore', 'dtype', 'shape', 'variable_name', '_array')

    def __init__(self, variable_name, datastore):
        if False:
            return 10
        self.datastore = datastore
        self.variable_name = variable_name
        self._array = self.datastore.zarr_group[self.variable_name]
        self.shape = self._array.shape
        if self._array.filters is not None and any([filt.codec_id == 'vlen-utf8' for filt in self._array.filters]):
            dtype = coding.strings.create_vlen_dtype(str)
        else:
            dtype = self._array.dtype
        self.dtype = dtype

    def get_array(self):
        if False:
            while True:
                i = 10
        return self._array

    def _oindex(self, key):
        if False:
            return 10
        return self.get_array().oindex[key]

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        array = self.get_array()
        if isinstance(key, indexing.BasicIndexer):
            return array[key.tuple]
        elif isinstance(key, indexing.VectorizedIndexer):
            return array.vindex[indexing._arrayize_vectorized_indexer(key, self.shape).tuple]
        else:
            assert isinstance(key, indexing.OuterIndexer)
            return indexing.explicit_indexing_adapter(key, array.shape, indexing.IndexingSupport.VECTORIZED, self._oindex)

def _determine_zarr_chunks(enc_chunks, var_chunks, ndim, name, safe_chunks):
    if False:
        while True:
            i = 10
    '\n    Given encoding chunks (possibly None or []) and variable chunks\n    (possibly None or []).\n    '
    if not var_chunks and (not enc_chunks):
        return None
    if var_chunks and (not enc_chunks):
        if any((len(set(chunks[:-1])) > 1 for chunks in var_chunks)):
            raise ValueError(f'Zarr requires uniform chunk sizes except for final chunk. Variable named {name!r} has incompatible dask chunks: {var_chunks!r}. Consider rechunking using `chunk()`.')
        if any((chunks[0] < chunks[-1] for chunks in var_chunks)):
            raise ValueError(f"Final chunk of Zarr array must be the same size or smaller than the first. Variable named {name!r} has incompatible Dask chunks {var_chunks!r}.Consider either rechunking using `chunk()` or instead deleting or modifying `encoding['chunks']`.")
        return tuple((chunk[0] for chunk in var_chunks))
    if isinstance(enc_chunks, integer_types):
        enc_chunks_tuple = ndim * (enc_chunks,)
    else:
        enc_chunks_tuple = tuple(enc_chunks)
    if len(enc_chunks_tuple) != ndim:
        return _determine_zarr_chunks(None, var_chunks, ndim, name, safe_chunks)
    for x in enc_chunks_tuple:
        if not isinstance(x, int):
            raise TypeError(f"zarr chunk sizes specified in `encoding['chunks']` must be an int or a tuple of ints. Instead found encoding['chunks']={enc_chunks_tuple!r} for variable named {name!r}.")
    if not var_chunks:
        return enc_chunks_tuple
    if var_chunks and enc_chunks_tuple:
        for (zchunk, dchunks) in zip(enc_chunks_tuple, var_chunks):
            for dchunk in dchunks[:-1]:
                if dchunk % zchunk:
                    base_error = f"Specified zarr chunks encoding['chunks']={enc_chunks_tuple!r} for variable named {name!r} would overlap multiple dask chunks {var_chunks!r}. Writing this array in parallel with dask could lead to corrupted data."
                    if safe_chunks:
                        raise NotImplementedError(base_error + " Consider either rechunking using `chunk()`, deleting or modifying `encoding['chunks']`, or specify `safe_chunks=False`.")
        return enc_chunks_tuple
    raise AssertionError('We should never get here. Function logic must be wrong.')

def _get_zarr_dims_and_attrs(zarr_obj, dimension_key, try_nczarr):
    if False:
        while True:
            i = 10
    try:
        dimensions = zarr_obj.attrs[dimension_key]
    except KeyError as e:
        if not try_nczarr:
            raise KeyError(f'Zarr object is missing the attribute `{dimension_key}`, which is required for xarray to determine variable dimensions.') from e
        zarray_path = os.path.join(zarr_obj.path, '.zarray')
        zarray = json.loads(zarr_obj.store[zarray_path])
        try:
            dimensions = [os.path.basename(dim) for dim in zarray['_NCZARR_ARRAY']['dimrefs']]
        except KeyError as e:
            raise KeyError(f'Zarr object is missing the attribute `{dimension_key}` and the NCZarr metadata, which are required for xarray to determine variable dimensions.') from e
    nc_attrs = [attr for attr in zarr_obj.attrs if attr.lower().startswith('_nc')]
    attributes = HiddenKeyDict(zarr_obj.attrs, [dimension_key] + nc_attrs)
    return (dimensions, attributes)

def extract_zarr_variable_encoding(variable, raise_on_invalid=False, name=None, safe_chunks=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Extract zarr encoding dictionary from xarray Variable\n\n    Parameters\n    ----------\n    variable : Variable\n    raise_on_invalid : bool, optional\n\n    Returns\n    -------\n    encoding : dict\n        Zarr encoding for `variable`\n    '
    encoding = variable.encoding.copy()
    safe_to_drop = {'source', 'original_shape'}
    valid_encodings = {'chunks', 'compressor', 'filters', 'cache_metadata', 'write_empty_chunks'}
    for k in safe_to_drop:
        if k in encoding:
            del encoding[k]
    if raise_on_invalid:
        invalid = [k for k in encoding if k not in valid_encodings]
        if invalid:
            raise ValueError(f'unexpected encoding parameters for zarr backend:  {invalid!r}')
    else:
        for k in list(encoding):
            if k not in valid_encodings:
                del encoding[k]
    chunks = _determine_zarr_chunks(encoding.get('chunks'), variable.chunks, variable.ndim, name, safe_chunks)
    encoding['chunks'] = chunks
    return encoding

def encode_zarr_variable(var, needs_copy=True, name=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Converts an Variable into an Variable which follows some\n    of the CF conventions:\n\n        - Nans are masked using _FillValue (or the deprecated missing_value)\n        - Rescaling via: scale_factor and add_offset\n        - datetimes are converted to the CF 'units since time' format\n        - dtype encodings are enforced.\n\n    Parameters\n    ----------\n    var : Variable\n        A variable holding un-encoded data.\n\n    Returns\n    -------\n    out : Variable\n        A variable which has been encoded as described above.\n    "
    var = conventions.encode_cf_variable(var, name=name)
    coder = coding.strings.EncodedStringCoder(allows_unicode=True)
    var = coder.encode(var, name=name)
    var = coding.strings.ensure_fixed_length_bytes(var)
    return var

def _validate_and_transpose_existing_dims(var_name, new_var, existing_var, region, append_dim):
    if False:
        while True:
            i = 10
    if new_var.dims != existing_var.dims:
        if set(existing_var.dims) == set(new_var.dims):
            new_var = new_var.transpose(*existing_var.dims)
        else:
            raise ValueError(f'variable {var_name!r} already exists with different dimension names {existing_var.dims} != {new_var.dims}, but changing variable dimensions is not supported by to_zarr().')
    existing_sizes = {}
    for (dim, size) in existing_var.sizes.items():
        if region is not None and dim in region:
            (start, stop, stride) = region[dim].indices(size)
            assert stride == 1
            size = stop - start
        if dim != append_dim:
            existing_sizes[dim] = size
    new_sizes = {dim: size for (dim, size) in new_var.sizes.items() if dim != append_dim}
    if existing_sizes != new_sizes:
        raise ValueError(f'variable {var_name!r} already exists with different dimension sizes: {existing_sizes} != {new_sizes}. to_zarr() only supports changing dimension sizes when explicitly appending, but append_dim={append_dim!r}. If you are attempting to write to a subset of the existing store without changing dimension sizes, consider using the region argument in to_zarr().')
    return new_var

def _put_attrs(zarr_obj, attrs):
    if False:
        print('Hello World!')
    'Raise a more informative error message for invalid attrs.'
    try:
        zarr_obj.attrs.put(attrs)
    except TypeError as e:
        raise TypeError('Invalid attribute in Dataset.attrs.') from e
    return zarr_obj

class ZarrStore(AbstractWritableDataStore):
    """Store for reading and writing data via zarr"""
    __slots__ = ('zarr_group', '_append_dim', '_consolidate_on_close', '_group', '_mode', '_read_only', '_synchronizer', '_write_region', '_safe_chunks', '_write_empty')

    @classmethod
    def open_group(cls, store, mode='r', synchronizer=None, group=None, consolidated=False, consolidate_on_close=False, chunk_store=None, storage_options=None, append_dim=None, write_region=None, safe_chunks=True, stacklevel=2, zarr_version=None, write_empty: bool | None=None):
        if False:
            for i in range(10):
                print('nop')
        import zarr
        if isinstance(store, os.PathLike):
            store = os.fspath(store)
        if zarr_version is None:
            zarr_version = getattr(store, '_store_version', 2)
        open_kwargs = dict(mode=mode, synchronizer=synchronizer, path=group)
        open_kwargs['storage_options'] = storage_options
        if zarr_version > 2:
            open_kwargs['zarr_version'] = zarr_version
            if consolidated or consolidate_on_close:
                raise ValueError(f'consolidated metadata has not been implemented for zarr version {zarr_version} yet. Set consolidated=False for zarr version {zarr_version}. See also https://github.com/zarr-developers/zarr-specs/issues/136')
            if consolidated is None:
                consolidated = False
        if chunk_store is not None:
            open_kwargs['chunk_store'] = chunk_store
            if consolidated is None:
                consolidated = False
        if consolidated is None:
            try:
                zarr_group = zarr.open_consolidated(store, **open_kwargs)
            except KeyError:
                try:
                    zarr_group = zarr.open_group(store, **open_kwargs)
                    warnings.warn('Failed to open Zarr store with consolidated metadata, but successfully read with non-consolidated metadata. This is typically much slower for opening a dataset. To silence this warning, consider:\n1. Consolidating metadata in this existing store with zarr.consolidate_metadata().\n2. Explicitly setting consolidated=False, to avoid trying to read consolidate metadata, or\n3. Explicitly setting consolidated=True, to raise an error in this case instead of falling back to try reading non-consolidated metadata.', RuntimeWarning, stacklevel=stacklevel)
                except zarr.errors.GroupNotFoundError:
                    raise FileNotFoundError(f"No such file or directory: '{store}'")
        elif consolidated:
            zarr_group = zarr.open_consolidated(store, **open_kwargs)
        else:
            zarr_group = zarr.open_group(store, **open_kwargs)
        return cls(zarr_group, mode, consolidate_on_close, append_dim, write_region, safe_chunks, write_empty)

    def __init__(self, zarr_group, mode=None, consolidate_on_close=False, append_dim=None, write_region=None, safe_chunks=True, write_empty: bool | None=None):
        if False:
            for i in range(10):
                print('nop')
        self.zarr_group = zarr_group
        self._read_only = self.zarr_group.read_only
        self._synchronizer = self.zarr_group.synchronizer
        self._group = self.zarr_group.path
        self._mode = mode
        self._consolidate_on_close = consolidate_on_close
        self._append_dim = append_dim
        self._write_region = write_region
        self._safe_chunks = safe_chunks
        self._write_empty = write_empty

    @property
    def ds(self):
        if False:
            print('Hello World!')
        return self.zarr_group

    def open_store_variable(self, name, zarr_array):
        if False:
            for i in range(10):
                print('nop')
        data = indexing.LazilyIndexedArray(ZarrArrayWrapper(name, self))
        try_nczarr = self._mode == 'r'
        (dimensions, attributes) = _get_zarr_dims_and_attrs(zarr_array, DIMENSION_KEY, try_nczarr)
        attributes = dict(attributes)
        attributes.pop('filters', None)
        encoding = {'chunks': zarr_array.chunks, 'preferred_chunks': dict(zip(dimensions, zarr_array.chunks)), 'compressor': zarr_array.compressor, 'filters': zarr_array.filters}
        if getattr(zarr_array, 'fill_value') is not None:
            attributes['_FillValue'] = zarr_array.fill_value
        return Variable(dimensions, data, attributes, encoding)

    def get_variables(self):
        if False:
            return 10
        return FrozenDict(((k, self.open_store_variable(k, v)) for (k, v) in self.zarr_group.arrays()))

    def get_attrs(self):
        if False:
            while True:
                i = 10
        return {k: v for (k, v) in self.zarr_group.attrs.asdict().items() if not k.lower().startswith('_nc')}

    def get_dimensions(self):
        if False:
            return 10
        try_nczarr = self._mode == 'r'
        dimensions = {}
        for (k, v) in self.zarr_group.arrays():
            (dim_names, _) = _get_zarr_dims_and_attrs(v, DIMENSION_KEY, try_nczarr)
            for (d, s) in zip(dim_names, v.shape):
                if d in dimensions and dimensions[d] != s:
                    raise ValueError(f'found conflicting lengths for dimension {d} ({s} != {dimensions[d]})')
                dimensions[d] = s
        return dimensions

    def set_dimensions(self, variables, unlimited_dims=None):
        if False:
            while True:
                i = 10
        if unlimited_dims is not None:
            raise NotImplementedError("Zarr backend doesn't know how to handle unlimited dimensions")

    def set_attributes(self, attributes):
        if False:
            i = 10
            return i + 15
        _put_attrs(self.zarr_group, attributes)

    def encode_variable(self, variable):
        if False:
            return 10
        variable = encode_zarr_variable(variable)
        return variable

    def encode_attribute(self, a):
        if False:
            return 10
        return encode_zarr_attr_value(a)

    def store(self, variables, attributes, check_encoding_set=frozenset(), writer=None, unlimited_dims=None):
        if False:
            i = 10
            return i + 15
        '\n        Top level method for putting data on this store, this method:\n          - encodes variables/attributes\n          - sets dimensions\n          - sets variables\n\n        Parameters\n        ----------\n        variables : dict-like\n            Dictionary of key/value (variable name / xr.Variable) pairs\n        attributes : dict-like\n            Dictionary of key/value (attribute name / attribute) pairs\n        check_encoding_set : list-like\n            List of variables that should be checked for invalid encoding\n            values\n        writer : ArrayWriter\n        unlimited_dims : list-like\n            List of dimension names that should be treated as unlimited\n            dimensions.\n            dimension on which the zarray will be appended\n            only needed in append mode\n        '
        import zarr
        existing_variable_names = {vn for vn in variables if _encode_variable_name(vn) in self.zarr_group}
        new_variables = set(variables) - existing_variable_names
        variables_without_encoding = {vn: variables[vn] for vn in new_variables}
        (variables_encoded, attributes) = self.encode(variables_without_encoding, attributes)
        if existing_variable_names:
            (existing_vars, _, _) = conventions.decode_cf_variables(self.get_variables(), self.get_attrs())
            vars_with_encoding = {}
            for vn in existing_variable_names:
                vars_with_encoding[vn] = variables[vn].copy(deep=False)
                vars_with_encoding[vn].encoding = existing_vars[vn].encoding
            (vars_with_encoding, _) = self.encode(vars_with_encoding, {})
            variables_encoded.update(vars_with_encoding)
            for var_name in existing_variable_names:
                new_var = variables_encoded[var_name]
                existing_var = existing_vars[var_name]
                new_var = _validate_and_transpose_existing_dims(var_name, new_var, existing_var, self._write_region, self._append_dim)
        if self._mode not in ['r', 'r+']:
            self.set_attributes(attributes)
            self.set_dimensions(variables_encoded, unlimited_dims=unlimited_dims)
        self.set_variables(variables_encoded, check_encoding_set, writer, unlimited_dims=unlimited_dims)
        if self._consolidate_on_close:
            zarr.consolidate_metadata(self.zarr_group.store)

    def sync(self):
        if False:
            while True:
                i = 10
        pass

    def set_variables(self, variables, check_encoding_set, writer, unlimited_dims=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This provides a centralized method to set the variables on the data\n        store.\n\n        Parameters\n        ----------\n        variables : dict-like\n            Dictionary of key/value (variable name / xr.Variable) pairs\n        check_encoding_set : list-like\n            List of variables that should be checked for invalid encoding\n            values\n        writer\n        unlimited_dims : list-like\n            List of dimension names that should be treated as unlimited\n            dimensions.\n        '
        import zarr
        for (vn, v) in variables.items():
            name = _encode_variable_name(vn)
            check = vn in check_encoding_set
            attrs = v.attrs.copy()
            dims = v.dims
            dtype = v.dtype
            shape = v.shape
            fill_value = attrs.pop('_FillValue', None)
            if v.encoding == {'_FillValue': None} and fill_value is None:
                v.encoding = {}
            if name in self.zarr_group:
                if self._write_empty is not None:
                    zarr_array = zarr.open(store=self.zarr_group.chunk_store, path=f'{self.zarr_group.name}/{name}', write_empty_chunks=self._write_empty)
                else:
                    zarr_array = self.zarr_group[name]
            else:
                encoding = extract_zarr_variable_encoding(v, raise_on_invalid=check, name=vn, safe_chunks=self._safe_chunks)
                encoded_attrs = {}
                encoded_attrs[DIMENSION_KEY] = dims
                for (k2, v2) in attrs.items():
                    encoded_attrs[k2] = self.encode_attribute(v2)
                if coding.strings.check_vlen_dtype(dtype) == str:
                    dtype = str
                if self._write_empty is not None:
                    if 'write_empty_chunks' in encoding and encoding['write_empty_chunks'] != self._write_empty:
                        raise ValueError(f"""Differing "write_empty_chunks" values in encoding and parametersGot encoding["write_empty_chunks"] = {encoding['write_empty_chunks']!r} and self._write_empty = {self._write_empty!r}""")
                    else:
                        encoding['write_empty_chunks'] = self._write_empty
                zarr_array = self.zarr_group.create(name, shape=shape, dtype=dtype, fill_value=fill_value, **encoding)
                zarr_array = _put_attrs(zarr_array, encoded_attrs)
            write_region = self._write_region if self._write_region is not None else {}
            write_region = {dim: write_region.get(dim, slice(None)) for dim in dims}
            if self._append_dim is not None and self._append_dim in dims:
                append_axis = dims.index(self._append_dim)
                assert write_region[self._append_dim] == slice(None)
                write_region[self._append_dim] = slice(zarr_array.shape[append_axis], None)
                new_shape = list(zarr_array.shape)
                new_shape[append_axis] += v.shape[append_axis]
                zarr_array.resize(new_shape)
            region = tuple((write_region[dim] for dim in dims))
            writer.add(v.data, zarr_array, region)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        pass

def open_zarr(store, group=None, synchronizer=None, chunks='auto', decode_cf=True, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables=None, consolidated=None, overwrite_encoded_chunks=False, chunk_store=None, storage_options=None, decode_timedelta=None, use_cftime=None, zarr_version=None, chunked_array_type: str | None=None, from_array_kwargs: dict[str, Any] | None=None, **kwargs):
    if False:
        print('Hello World!')
    'Load and decode a dataset from a Zarr store.\n\n    The `store` object should be a valid store for a Zarr group. `store`\n    variables must contain dimension metadata encoded in the\n    `_ARRAY_DIMENSIONS` attribute or must have NCZarr format.\n\n    Parameters\n    ----------\n    store : MutableMapping or str\n        A MutableMapping where a Zarr Group has been stored or a path to a\n        directory in file system where a Zarr DirectoryStore has been stored.\n    synchronizer : object, optional\n        Array synchronizer provided to zarr\n    group : str, optional\n        Group path. (a.k.a. `path` in zarr terminology.)\n    chunks : int or dict or tuple or {None, \'auto\'}, optional\n        Chunk sizes along each dimension, e.g., ``5`` or\n        ``{\'x\': 5, \'y\': 5}``. If `chunks=\'auto\'`, dask chunks are created\n        based on the variable\'s zarr chunks. If `chunks=None`, zarr array\n        data will lazily convert to numpy arrays upon access. This accepts\n        all the chunk specifications as Dask does.\n    overwrite_encoded_chunks : bool, optional\n        Whether to drop the zarr chunks encoded for each variable when a\n        dataset is loaded with specified chunk sizes (default: False)\n    decode_cf : bool, optional\n        Whether to decode these variables, assuming they were saved according\n        to CF conventions.\n    mask_and_scale : bool, optional\n        If True, replace array values equal to `_FillValue` with NA and scale\n        values according to the formula `original_values * scale_factor +\n        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are\n        taken from variable attributes (if they exist).  If the `_FillValue` or\n        `missing_value` attribute contains multiple values a warning will be\n        issued and all array values matching one of the multiple values will\n        be replaced by NA.\n    decode_times : bool, optional\n        If True, decode times encoded in the standard NetCDF datetime format\n        into datetime objects. Otherwise, leave them encoded as numbers.\n    concat_characters : bool, optional\n        If True, concatenate along the last dimension of character arrays to\n        form string arrays. Dimensions will only be concatenated over (and\n        removed) if they have no corresponding variable and if they are only\n        used as the last dimension of character arrays.\n    decode_coords : bool, optional\n        If True, decode the \'coordinates\' attribute to identify coordinates in\n        the resulting dataset.\n    drop_variables : str or iterable, optional\n        A variable or list of variables to exclude from being parsed from the\n        dataset. This may be useful to drop variables with problems or\n        inconsistent values.\n    consolidated : bool, optional\n        Whether to open the store using zarr\'s consolidated metadata\n        capability. Only works for stores that have already been consolidated.\n        By default (`consolidate=None`), attempts to read consolidated metadata,\n        falling back to read non-consolidated metadata if that fails.\n\n        When the experimental ``zarr_version=3``, ``consolidated`` must be\n        either be ``None`` or ``False``.\n    chunk_store : MutableMapping, optional\n        A separate Zarr store only for chunk data.\n    storage_options : dict, optional\n        Any additional parameters for the storage backend (ignored for local\n        paths).\n    decode_timedelta : bool, optional\n        If True, decode variables and coordinates with time units in\n        {\'days\', \'hours\', \'minutes\', \'seconds\', \'milliseconds\', \'microseconds\'}\n        into timedelta objects. If False, leave them encoded as numbers.\n        If None (default), assume the same value of decode_time.\n    use_cftime : bool, optional\n        Only relevant if encoded dates come from a standard calendar\n        (e.g. "gregorian", "proleptic_gregorian", "standard", or not\n        specified).  If None (default), attempt to decode times to\n        ``np.datetime64[ns]`` objects; if this is not possible, decode times to\n        ``cftime.datetime`` objects. If True, always decode times to\n        ``cftime.datetime`` objects, regardless of whether or not they can be\n        represented using ``np.datetime64[ns]`` objects.  If False, always\n        decode times to ``np.datetime64[ns]`` objects; if this is not possible\n        raise an error.\n    zarr_version : int or None, optional\n        The desired zarr spec version to target (currently 2 or 3). The default\n        of None will attempt to determine the zarr version from ``store`` when\n        possible, otherwise defaulting to 2.\n    chunked_array_type: str, optional\n        Which chunked array type to coerce this datasets\' arrays to.\n        Defaults to \'dask\' if installed, else whatever is registered via the `ChunkManagerEntryPoint` system.\n        Experimental API that should not be relied upon.\n    from_array_kwargs: dict, optional\n        Additional keyword arguments passed on to the `ChunkManagerEntrypoint.from_array` method used to create\n        chunked arrays, via whichever chunk manager is specified through the `chunked_array_type` kwarg.\n        Defaults to {\'manager\': \'dask\'}, meaning additional kwargs will be passed eventually to\n        :py:func:`dask.array.from_array`. Experimental API that should not be relied upon.\n\n    Returns\n    -------\n    dataset : Dataset\n        The newly created dataset.\n\n    See Also\n    --------\n    open_dataset\n    open_mfdataset\n\n    References\n    ----------\n    http://zarr.readthedocs.io/\n    '
    from xarray.backends.api import open_dataset
    if from_array_kwargs is None:
        from_array_kwargs = {}
    if chunks == 'auto':
        try:
            guess_chunkmanager(chunked_array_type)
            chunks = {}
        except ValueError:
            chunks = None
    if kwargs:
        raise TypeError('open_zarr() got unexpected keyword arguments ' + ','.join(kwargs.keys()))
    backend_kwargs = {'synchronizer': synchronizer, 'consolidated': consolidated, 'overwrite_encoded_chunks': overwrite_encoded_chunks, 'chunk_store': chunk_store, 'storage_options': storage_options, 'stacklevel': 4, 'zarr_version': zarr_version}
    ds = open_dataset(filename_or_obj=store, group=group, decode_cf=decode_cf, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, engine='zarr', chunks=chunks, drop_variables=drop_variables, chunked_array_type=chunked_array_type, from_array_kwargs=from_array_kwargs, backend_kwargs=backend_kwargs, decode_timedelta=decode_timedelta, use_cftime=use_cftime, zarr_version=zarr_version)
    return ds

class ZarrBackendEntrypoint(BackendEntrypoint):
    """
    Backend for ".zarr" files based on the zarr package.

    For more information about the underlying library, visit:
    https://zarr.readthedocs.io/en/stable

    See Also
    --------
    backends.ZarrStore
    """
    description = 'Open zarr files (.zarr) using zarr in Xarray'
    url = 'https://docs.xarray.dev/en/stable/generated/xarray.backends.ZarrBackendEntrypoint.html'

    def guess_can_open(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore) -> bool:
        if False:
            i = 10
            return i + 15
        if isinstance(filename_or_obj, (str, os.PathLike)):
            (_, ext) = os.path.splitext(filename_or_obj)
            return ext in {'.zarr'}
        return False

    def open_dataset(self, filename_or_obj: str | os.PathLike[Any] | BufferedIOBase | AbstractDataStore, *, mask_and_scale=True, decode_times=True, concat_characters=True, decode_coords=True, drop_variables: str | Iterable[str] | None=None, use_cftime=None, decode_timedelta=None, group=None, mode='r', synchronizer=None, consolidated=None, chunk_store=None, storage_options=None, stacklevel=3, zarr_version=None) -> Dataset:
        if False:
            return 10
        filename_or_obj = _normalize_path(filename_or_obj)
        store = ZarrStore.open_group(filename_or_obj, group=group, mode=mode, synchronizer=synchronizer, consolidated=consolidated, consolidate_on_close=False, chunk_store=chunk_store, storage_options=storage_options, stacklevel=stacklevel + 1, zarr_version=zarr_version)
        store_entrypoint = StoreBackendEntrypoint()
        with close_on_error(store):
            ds = store_entrypoint.open_dataset(store, mask_and_scale=mask_and_scale, decode_times=decode_times, concat_characters=concat_characters, decode_coords=decode_coords, drop_variables=drop_variables, use_cftime=use_cftime, decode_timedelta=decode_timedelta)
        return ds
BACKEND_ENTRYPOINTS['zarr'] = ('zarr', ZarrBackendEntrypoint)