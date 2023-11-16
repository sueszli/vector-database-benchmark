import deeplake
from deeplake.core.distance_type import DistanceType
from deeplake.core.linked_chunk_engine import LinkedChunkEngine
from deeplake.core.storage.lru_cache import LRUCache
from deeplake.util.downsample import apply_partial_downsample
from deeplake.util.invalid_view_op import invalid_view_op
from deeplake.core.version_control.commit_chunk_map import CommitChunkMap
from deeplake.core.version_control.commit_diff import CommitDiff
from deeplake.core.chunk.base_chunk import InputSample
import numpy as np
from typing import Dict, List, Sequence, Union, Optional, Tuple, Any, Callable
from functools import reduce, partial
from deeplake.core import index_maintenance
from deeplake.core.index import Index, IndexEntry, replace_ellipsis_with_slices
from deeplake.core.meta.tensor_meta import TensorMeta, _validate_htype_exists
from deeplake.core.storage import StorageProvider
from deeplake.core.chunk_engine import ChunkEngine
from deeplake.core.compression import _read_timestamps
from deeplake.core.tensor_link import cast_to_type, extend_downsample, get_link_transform, update_downsample
from deeplake.api.info import Info, load_info
from deeplake.util.keys import get_chunk_id_encoder_key, get_chunk_key, get_tensor_commit_chunk_map_key, get_tensor_commit_diff_key, get_tensor_meta_key, get_tensor_tile_encoder_key, get_tensor_vdb_index_key, get_sequence_encoder_key, tensor_exists, get_tensor_info_key, get_sample_id_tensor_key, get_sample_info_tensor_key, get_sample_shape_tensor_key
from deeplake.util.modified import get_modified_indexes
from deeplake.util.class_label import convert_to_text
from deeplake.util.shape_interval import ShapeInterval
from deeplake.util.exceptions import TensorDoesNotExistError, InvalidKeyTypeError, TensorAlreadyExistsError, UnsupportedCompressionError, EmbeddingTensorPopError
from deeplake.util.iteration_warning import check_if_iteration
from deeplake.hooks import dataset_read, dataset_written
from deeplake.util.pretty_print import summary_tensor
from deeplake.constants import FIRST_COMMIT_ID, _NO_LINK_UPDATE, UNSPECIFIED, _INDEX_OPERATION_MAPPING
from deeplake.util.version_control import auto_checkout
from deeplake.util.video import normalize_index
from deeplake.compression import get_compression_type, VIDEO_COMPRESSION, BYTE_COMPRESSION
from deeplake.util.notebook import is_jupyter, video_html, is_colab
from deeplake.util.object_3d.point_cloud import parse_point_cloud_to_dict
from deeplake.util.object_3d.mesh import parse_mesh_to_dict, get_mesh_vertices
from deeplake.util.htype import parse_complex_htype
from deeplake.htype import HTYPE_CONVERSION_LHS, HTYPE_CONSTRAINTS, HTYPE_SUPPORTED_COMPRESSIONS
import warnings
import webbrowser

def create_tensor(key: str, storage: StorageProvider, htype: str, sample_compression: str, chunk_compression: str, version_state: Dict[str, Any], overwrite: bool=False, **kwargs):
    if False:
        print('Hello World!')
    "If a tensor does not exist, create a new one with the provided meta.\n\n    Args:\n        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.\n        storage (StorageProvider): StorageProvider that all tensor data is written to.\n        htype (str): Htype is how the default tensor metadata is defined.\n        sample_compression (str): All samples will be compressed in the provided format. If `None`, samples are uncompressed.\n        chunk_compression (str): All chunks will be compressed in the provided format. If `None`, chunks are uncompressed.\n        version_state (Dict[str, Any]): The version state of the dataset, includes commit_id, commit_node, branch, branch_commit_map and commit_node_map.\n        overwrite (bool): If `True`, any existing data in the tensor's directory will be overwritten.\n        **kwargs: `htype` defaults can be overridden by passing any of the compatible parameters.\n            To see all `htype`s and their correspondent arguments, check out `/deeplake/htypes.py`.\n\n    Raises:\n        TensorAlreadyExistsError: If a tensor defined with `key` already exists and `overwrite` is False.\n    "
    commit_id = version_state['commit_id']
    if not overwrite and tensor_exists(key, storage, commit_id):
        raise TensorAlreadyExistsError(key)
    meta_key = get_tensor_meta_key(key, commit_id)
    meta = TensorMeta(htype=htype, sample_compression=sample_compression, chunk_compression=chunk_compression, **kwargs)
    storage[meta_key] = meta
    if commit_id != FIRST_COMMIT_ID:
        cmap_key = get_tensor_commit_chunk_map_key(key, commit_id)
        cmap = CommitChunkMap()
        storage[cmap_key] = cmap
    diff_key = get_tensor_commit_diff_key(key, commit_id)
    diff = CommitDiff(created=True)
    storage[diff_key] = diff

def delete_tensor(key: str, dataset):
    if False:
        for i in range(10):
            print('nop')
    "Delete tensor from storage.\n\n    Args:\n        key (str): Key for where the chunks, index_meta, and tensor_meta will be located in `storage` relative to it's root.\n        dataset (Dataset): Dataset that the tensor is located in.\n\n    Raises:\n        TensorDoesNotExistError: If no tensor with `key` exists and a `tensor_meta` was not provided.\n    "
    storage = dataset.storage
    version_state = dataset.version_state
    tensor = Tensor(key, dataset)
    chunk_engine: ChunkEngine = tensor.chunk_engine
    enc = chunk_engine.chunk_id_encoder
    index_ids = tensor.meta.get_vdb_index_ids()
    for id in index_ids:
        tensor.delete_vdb_index(id)
    n_chunks = chunk_engine.num_chunks
    chunk_names = [enc.get_name_for_chunk(i) for i in range(n_chunks)]
    chunk_keys = [get_chunk_key(key, chunk_name, version_state['commit_id']) for chunk_name in chunk_names]
    for chunk_key in chunk_keys:
        try:
            del storage[chunk_key]
        except KeyError:
            pass
    commit_id = version_state['commit_id']
    meta_key = get_tensor_meta_key(key, commit_id)
    try:
        del storage[meta_key]
    except KeyError:
        pass
    info_key = get_tensor_info_key(key, commit_id)
    try:
        del storage[info_key]
    except KeyError:
        pass
    diff_key = get_tensor_commit_diff_key(key, commit_id)
    try:
        del storage[diff_key]
    except KeyError:
        pass
    chunk_id_encoder_key = get_chunk_id_encoder_key(key, commit_id)
    try:
        del storage[chunk_id_encoder_key]
    except KeyError:
        pass
    tile_encoder_key = get_tensor_tile_encoder_key(key, commit_id)
    try:
        del storage[tile_encoder_key]
    except KeyError:
        pass
    seq_encoder_key = get_sequence_encoder_key(key, commit_id)
    try:
        del storage[seq_encoder_key]
    except KeyError:
        pass

def _inplace_op(f):
    if False:
        return 10
    op = f.__name__

    def inner(tensor, other):
        if False:
            print('Hello World!')
        tensor._write_initialization()
        tensor.chunk_engine.update(tensor.index, other, op, link_callback=tensor._update_links if tensor.meta.links else None)
        if not tensor.index.is_trivial():
            tensor._skip_next_setitem = True
        return tensor
    return inner

class Tensor:

    def __init__(self, key: str, dataset, index: Optional[Index]=None, is_iteration: bool=False, chunk_engine: Optional[ChunkEngine]=None):
        if False:
            return 10
        'Initializes a new tensor.\n\n        Args:\n            key (str): The internal identifier for this tensor.\n            dataset (Dataset): The dataset that this tensor is located in.\n            index: The Index object restricting the view of this tensor.\n                Can be an int, slice, or (used internally) an Index object.\n            is_iteration (bool): If this tensor is being used as an iterator.\n            chunk_engine (ChunkEngine, optional): The underlying chunk_engine for the tensor.\n\n        Raises:\n            TensorDoesNotExistError: If no tensor with ``key`` exists and a ``tensor_meta`` was not provided.\n\n        Note:\n            This operation does not create a new tensor in the storage provider,\n            and should normally only be performed by Deep Lake internals.\n        '
        self.key = key
        self.dataset = dataset
        self.storage: LRUCache = dataset.storage
        self.index = index or Index()
        self.version_state = dataset.version_state
        self.link_creds = dataset.link_creds
        self.is_iteration = is_iteration
        commit_id = self.version_state['commit_id']
        if not self.is_iteration and (not tensor_exists(self.key, self.storage, commit_id)):
            raise TensorDoesNotExistError(self.key)
        meta_key = get_tensor_meta_key(self.key, commit_id)
        meta = self.storage.get_deeplake_object(meta_key, TensorMeta)
        if chunk_engine is not None:
            self.chunk_engine = chunk_engine
        elif meta.is_link:
            self.chunk_engine = LinkedChunkEngine(self.key, self.storage, self.version_state, link_creds=dataset.link_creds)
        else:
            self.chunk_engine = ChunkEngine(self.key, self.storage, self.version_state)
        if not self.pad_tensor and (not self.is_iteration):
            self.index.validate(self.num_samples)
        self._skip_next_setitem = False
        self._indexing_history: List[int] = []

    @property
    def pad_tensor(self):
        if False:
            i = 10
            return i + 15
        return self.dataset._pad_tensors

    def _write_initialization(self):
        if False:
            return 10
        self.storage.check_readonly()
        if auto_checkout(self.dataset):
            self.chunk_engine = self.version_state['full_tensors'][self.key].chunk_engine

    def _extend(self, samples: Union[np.ndarray, Sequence[InputSample], 'Tensor'], progressbar: bool=False, ignore_errors: bool=False):
        if False:
            return 10
        self._write_initialization()
        [f() for f in list(self.dataset._update_hooks.values())]
        self.chunk_engine.extend(samples, progressbar=progressbar, link_callback=self._extend_links if self.meta.links else None, ignore_errors=ignore_errors)
        dataset_written(self.dataset)
        self.invalidate_libdeeplake_dataset()

    @invalid_view_op
    def extend(self, samples: Union[np.ndarray, Sequence[InputSample], 'Tensor'], progressbar: bool=False, ignore_errors: bool=False):
        if False:
            for i in range(10):
                print('nop')
        'Extends the end of the tensor by appending multiple elements from a sequence. Accepts a sequence, a single batched numpy array,\n        or a sequence of :func:`deeplake.read` outputs, which can be used to load files. See examples down below.\n\n        Example:\n            Numpy input:\n\n            >>> len(tensor)\n            0\n            >>> tensor.extend(np.zeros((100, 28, 28, 1)))\n            >>> len(tensor)\n            100\n\n\n            File input:\n\n            >>> len(tensor)\n            0\n            >>> tensor.extend([\n                    deeplake.read("path/to/image1"),\n                    deeplake.read("path/to/image2"),\n                ])\n            >>> len(tensor)\n            2\n\n\n        Args:\n            samples (np.ndarray, Sequence, Sequence[Sample]): The data to add to the tensor.\n                The length should be equal to the number of samples to add.\n            progressbar (bool): Specifies whether a progressbar should be displayed while extending.\n            ignore_errors (bool): Skip samples that cause errors while extending, if set to ``True``.\n\n        Raises:\n            TensorDtypeMismatchError: Dtype for array must be equal to or castable to this tensor\'s dtype.\n        '
        self._extend(samples, progressbar=progressbar, ignore_errors=ignore_errors)
        if index_maintenance.validate_embedding_tensor(self):
            row_ids = list(range(self.num_samples - len(samples), self.num_samples))
            index_maintenance.index_operation_dataset(self.dataset, dml_type=_INDEX_OPERATION_MAPPING['ADD'], rowids=row_ids)

    @invalid_view_op
    def _extend_with_paths(self, samples: Union[np.ndarray, Sequence[InputSample], 'Tensor'], progressbar: bool=False):
        if False:
            for i in range(10):
                print('nop')
        if not self.is_link:
            raise ValueError('Not supported as the tensor is not a link.')
        self._write_initialization()
        [f() for f in list(self.dataset._update_hooks.values())]
        self.chunk_engine.path_chunk_engine.extend(samples, progressbar=progressbar, link_callback=self._extend_links if self.meta.links else None)
        dataset_written(self.dataset)
        self.invalidate_libdeeplake_dataset()

    @property
    def info(self) -> Info:
        if False:
            while True:
                i = 10
        'Returns the information about the tensor. User can set info of tensor.\n\n        Returns:\n            Info: Information about the tensor.\n\n        Example:\n\n            >>> # update info\n            >>> ds.images.info.update(large=True, gray=False)\n            >>> # get info\n            >>> ds.images.info\n            {\'large\': True, \'gray\': False}\n\n            >>> ds.images.info = {"complete": True}\n            >>> ds.images.info\n            {\'complete\': True}\n\n        '
        commit_id = self.version_state['commit_id']
        chunk_engine = self.chunk_engine
        if chunk_engine._info is None or chunk_engine._info_commit_id != commit_id:
            path = get_tensor_info_key(self.key, commit_id)
            chunk_engine._info = load_info(path, self.dataset, self.key)
            chunk_engine._info_commit_id = commit_id
            self.storage.register_deeplake_object(path, chunk_engine._info)
        return chunk_engine._info

    @info.setter
    def info(self, value):
        if False:
            print('Hello World!')
        if isinstance(value, dict):
            info = self.info
            info.replace_with(value)
        else:
            raise TypeError('Info must be set with type Dict')

    def _append(self, sample: InputSample):
        if False:
            i = 10
            return i + 15
        self._extend([sample], progressbar=False)

    @invalid_view_op
    def append(self, sample: InputSample):
        if False:
            i = 10
            return i + 15
        'Appends a single sample to the end of the tensor. Can be an array, scalar value, or the return value from :func:`deeplake.read`,\n        which can be used to load files. See examples down below.\n\n        Examples:\n            Numpy input:\n\n            >>> len(tensor)\n            0\n            >>> tensor.append(np.zeros((28, 28, 1)))\n            >>> len(tensor)\n            1\n\n            File input:\n\n            >>> len(tensor)\n            0\n            >>> tensor.append(deeplake.read("path/to/file"))\n            >>> len(tensor)\n            1\n\n        Args:\n            sample (InputSample): The data to append to the tensor. :class:`~deeplake.core.sample.Sample` is generated by :func:`deeplake.read`. See the above examples.\n        '
        row_ids = [self.num_samples]
        self._extend([sample], progressbar=False)
        if index_maintenance.validate_embedding_tensor(self):
            index_maintenance.index_operation_dataset(self.dataset, dml_type=_INDEX_OPERATION_MAPPING['ADD'], rowids=row_ids)

    def clear(self):
        if False:
            return 10
        'Deletes all samples from the tensor'
        self.chunk_engine.clear()
        try:
            for t in self._all_tensor_links():
                t.chunk_engine.clear()
        except TensorDoesNotExistError:
            pass
        self.invalidate_libdeeplake_dataset()

    def modified_samples(self, target_id: Optional[str]=None, return_indexes: Optional[bool]=False):
        if False:
            return 10
        'Returns a slice of the tensor with only those elements that were modified/added.\n        By default the modifications are calculated relative to the previous commit made, but this can be changed by providing a ``target id``.\n\n        Args:\n            target_id (str, optional): The commit id or branch name to calculate the modifications relative to. Defaults to ``None``.\n            return_indexes (bool, optional): If ``True``, returns the indexes of the modified elements. Defaults to ``False``.\n\n        Returns:\n            Tensor: A new tensor with only the modified elements if ``return_indexes`` is ``False``.\n            Tuple[Tensor, List[int]]: A new tensor with only the modified elements and the indexes of the modified elements if ``return_indexes`` is ``True``.\n\n        Raises:\n            TensorModifiedError: If a target id is passed which is not an ancestor of the current commit.\n        '
        current_commit_id = self.version_state['commit_id']
        indexes = get_modified_indexes(self.key, current_commit_id, target_id, self.version_state, self.storage)
        tensor = self[indexes]
        if return_indexes:
            return (tensor, indexes)
        return tensor

    @property
    def meta(self):
        if False:
            print('Hello World!')
        'Metadata of the tensor.'
        return self.chunk_engine.tensor_meta

    @property
    def shape(self) -> Tuple[Optional[int], ...]:
        if False:
            i = 10
            return i + 15
        "Get the shape of this tensor. Length is included.\n\n        Example:\n\n            >>> tensor.append(np.zeros((10, 10)))\n            >>> tensor.append(np.zeros((10, 15)))\n            >>> tensor.shape\n            (2, 10, None)\n\n        Returns:\n            tuple: Tuple where each value is either ``None`` (if that axis is dynamic) or an int (if that axis is fixed).\n\n        Note:\n            If you don't want ``None`` in the output shape or want the lower/upper bound shapes,\n            use :attr:`shape_interval` instead.\n        "
        sample_shape_tensor = self._sample_shape_tensor
        sample_shape_provider = self._sample_shape_provider(sample_shape_tensor) if sample_shape_tensor else None
        shape: Tuple[Optional[int], ...]
        shape = self.chunk_engine.shape(self.index, sample_shape_provider=sample_shape_provider, pad_tensor=self.pad_tensor)
        if len(self.index.values) == 1 and (not self.index.values[0].subscriptable()):
            if None not in shape and np.sum(shape) == 0 and self.meta.max_shape:
                shape = (0,) * len(self.meta.max_shape)
        if self.meta.max_shape == [0, 0, 0]:
            shape = ()
        return shape

    def shapes(self):
        if False:
            print('Hello World!')
        'Get the shapes of all the samples in the tensor.\n\n        Returns:\n            np.ndarray: List of shapes of all the samples in the tensor.\n        '
        sample_shape_tensor = self._sample_shape_tensor
        sample_shape_provider = self._sample_shape_provider(sample_shape_tensor) if sample_shape_tensor else None
        return self.chunk_engine.shapes(self.index, sample_shape_provider=sample_shape_provider, pad_tensor=self.pad_tensor)

    @property
    def size(self) -> Optional[int]:
        if False:
            print('Hello World!')
        s = 1
        for x in self.shape:
            if x is None:
                return None
            s *= x
        return s

    @property
    def ndim(self) -> int:
        if False:
            print('Hello World!')
        'Number of dimensions of the tensor.'
        return self.chunk_engine.ndim(self.index)

    @property
    def dtype(self) -> Optional[np.dtype]:
        if False:
            for i in range(10):
                print('nop')
        'Dtype of the tensor.'
        if self.base_htype in ('json', 'list'):
            return np.dtype(str)
        if self.meta.dtype:
            return np.dtype(self.meta.typestr or self.meta.dtype)
        return None

    @property
    def is_sequence(self):
        if False:
            print('Hello World!')
        'Whether this tensor is a sequence tensor.'
        return self.meta.is_sequence

    @property
    def is_link(self):
        if False:
            i = 10
            return i + 15
        'Whether this tensor is a link tensor.'
        return self.meta.is_link

    @property
    def verify(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether linked data will be verified when samples are added. Applicable only to tensors with htype ``link[htype]``.'
        return self.is_link and self.meta.verify

    @property
    def htype(self):
        if False:
            return 10
        'Htype of the tensor.'
        htype = self.meta.htype
        if self.is_sequence:
            htype = f'sequence[{htype}]'
        if self.is_link:
            htype = f'link[{htype}]'
        return htype

    @htype.setter
    def htype(self, value):
        if False:
            return 10
        self._check_compatibility_with_htype(value)
        self.meta.htype = value
        if value == 'class_label':
            self.meta._disable_temp_transform = False
        self.meta.is_dirty = True
        self.dataset.maybe_flush()

    @property
    def hidden(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether this tensor is a hidden tensor.'
        return self.meta.hidden

    @property
    def base_htype(self):
        if False:
            print('Hello World!')
        'Base htype of the tensor.\n\n        Example:\n\n            >>> ds.create_tensor("video_seq", htype="sequence[video]", sample_compression="mp4")\n            >>> ds.video_seq.htype\n            sequence[video]\n            >>> ds.video_seq.base_htype\n            video\n        '
        return self.meta.htype

    @property
    def shape_interval(self) -> ShapeInterval:
        if False:
            i = 10
            return i + 15
        "Returns a :class:`~deeplake.util.shape_interval.ShapeInterval` object that describes this tensor's shape more accurately. Length is included.\n\n        Example:\n\n            >>> tensor.append(np.zeros((10, 10)))\n            >>> tensor.append(np.zeros((10, 15)))\n            >>> tensor.shape_interval\n            ShapeInterval(lower=(2, 10, 10), upper=(2, 10, 15))\n            >>> str(tensor.shape_interval)\n            (2, 10, 10:15)\n\n        Returns:\n            ShapeInterval: Object containing ``lower`` and ``upper`` properties.\n\n        Note:\n            If you are expecting a tuple, use :attr:`shape` instead.\n        "
        sample_shape_tensor = self._sample_shape_tensor
        sample_shape_provider = self._sample_shape_provider(sample_shape_tensor) if sample_shape_tensor else None
        return self.chunk_engine.shape_interval(self.index, sample_shape_provider)

    @property
    def is_dynamic(self) -> bool:
        if False:
            return 10
        'Will return ``True`` if samples in this tensor have shapes that are unequal.'
        return self.shape_interval.is_dynamic

    @property
    def num_samples(self) -> int:
        if False:
            print('Hello World!')
        'Returns the length of the primary axis of the tensor.\n        Ignores any applied indexing and returns the total length.\n        '
        return self.chunk_engine.tensor_length

    def __len__(self):
        if False:
            i = 10
            return i + 15
        'Returns the length of the primary axis of the tensor.\n        Accounts for indexing into the tensor object.\n\n        Examples:\n            >>> len(tensor)\n            0\n            >>> tensor.extend(np.zeros((100, 10, 10)))\n            >>> len(tensor)\n            100\n            >>> len(tensor[5:10])\n            5\n\n        Returns:\n            int: The current length of this tensor.\n        '
        self.chunk_engine.validate_num_samples_is_synchronized()
        return self.index.length(self.num_samples)

    def __getitem__(self, item: Union[int, slice, List[int], Tuple[Union[int, slice, Tuple[int]]], Index], is_iteration: bool=False):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, (int, slice, list, tuple, type(Ellipsis), Index)):
            raise InvalidKeyTypeError(item)
        if isinstance(item, tuple) or item is Ellipsis:
            item = replace_ellipsis_with_slices(item, self.ndim)
        if not self.meta.hidden and (not is_iteration) and isinstance(item, int):
            is_iteration = check_if_iteration(self._indexing_history, item)
            if is_iteration and deeplake.constants.SHOW_ITERATION_WARNING:
                warnings.warn('Indexing by integer in a for loop, like `for i in range(len(ds)): ... ds.tensor[i]` can be quite slow. Use `for i, sample in enumerate(ds)` instead.')
        return Tensor(self.key, self.dataset, index=self.index[item], is_iteration=is_iteration, chunk_engine=self.chunk_engine)

    def _get_bigger_dtype(self, d1, d2):
        if False:
            return 10
        if np.can_cast(d1, d2):
            if np.can_cast(d2, d1):
                return d1
            else:
                return d2
        elif np.can_cast(d2, d1):
            return d2
        else:
            return np.object

    def _infer_np_dtype(self, val: Any) -> np.dtype:
        if False:
            print('Hello World!')
        if hasattr(val, 'dtype'):
            return val.dtype
        elif isinstance(val, int):
            return np.array(0).dtype
        elif isinstance(val, float):
            return np.array(0.0).dtype
        elif isinstance(val, str):
            return np.array('').dtype
        elif isinstance(val, bool):
            return np.dtype(bool)
        elif isinstance(val, Sequence):
            return reduce(self._get_bigger_dtype, map(self._infer_np_dtype, val))
        else:
            raise TypeError(f'Cannot infer numpy dtype for {val}')

    def _update(self, item: Union[int, slice], value: Any):
        if False:
            i = 10
            return i + 15
        self._write_initialization()
        [f() for f in list(self.dataset._update_hooks.values())]
        update_link_callback = self._update_links if self.meta.links else None
        if isinstance(value, Tensor):
            if value._skip_next_setitem:
                value._skip_next_setitem = False
                return
            value = value.numpy(aslist=True)
        item_index = Index(item)
        if deeplake.constants._ENABLE_RANDOM_ASSIGNMENT and isinstance(item, int) and (item >= self.num_samples):
            if self.is_sequence:
                raise NotImplementedError('Random assignment is not supported for sequences yet.')
            num_samples_to_pad = item - self.num_samples
            extend_link_callback = self._extend_links if self.meta.links else None
            self.chunk_engine.pad_and_append(num_samples_to_pad, value, extend_link_callback=extend_link_callback, update_link_callback=update_link_callback)
        else:
            if not item_index.values[0].subscriptable() and (not self.is_sequence):
                value = [value]
            self.chunk_engine.update(self.index[item_index], value, link_callback=update_link_callback)
        dataset_written(self.dataset)
        self.invalidate_libdeeplake_dataset()

    def __setitem__(self, item: Union[int, slice], value: Any):
        if False:
            return 10
        'Update samples with new values.\n\n        Example:\n\n            >>> tensor.append(np.zeros((10, 10)))\n            >>> tensor.shape\n            (1, 10, 10)\n            >>> tensor[0] = np.zeros((3, 3))\n            >>> tensor.shape\n            (1, 3, 3)\n        '
        self._update(item, value)
        if index_maintenance.is_embedding_tensor(self):
            row_index = self.index[Index(item)]
            row_ids = list(row_index.values[0].indices(self.num_samples))
            index_maintenance.index_operation_dataset(self.dataset, dml_type=_INDEX_OPERATION_MAPPING['UPDATE'], rowids=row_ids)

    def __iter__(self):
        if False:
            return 10
        for i in range(len(self)):
            yield self.__getitem__(i, is_iteration=not isinstance(self.index.values[0], list))

    @property
    def is_empty_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        if self.meta.is_link:
            if len(self.meta.max_shape) == 0:
                for chunk in self.chunk_engine.get_chunks_for_sample(0):
                    if len(chunk.data_bytes) != 0:
                        return False
                return True
            return False
        if self.meta.chunk_compression and get_compression_type(self.meta.chunk_compression) != BYTE_COMPRESSION:
            return self.meta.max_shape == [0, 0, 0] or len(self.meta.max_shape) == 0
        return len(self.meta.max_shape) == 0

    def numpy(self, aslist=False, fetch_chunks=False) -> Union[np.ndarray, List[np.ndarray]]:
        if False:
            for i in range(10):
                print('nop')
        'Computes the contents of the tensor in numpy format.\n\n        Args:\n            aslist (bool): If ``True``, a list of np.ndarrays will be returned. Helpful for dynamic tensors.\n                If ``False``, a single np.ndarray will be returned unless the samples are dynamically shaped, in which case\n                an error is raised.\n            fetch_chunks (bool): If ``True``, full chunks will be retrieved from the storage, otherwise only required bytes will be retrieved.\n                This will always be ``True`` even if specified as ``False`` in the following cases:\n\n                - The tensor is ChunkCompressed.\n                - The chunk which is being accessed has more than 128 samples.\n\n        Raises:\n            DynamicTensorNumpyError: If reading a dynamically-shaped array slice without ``aslist=True``.\n            ValueError: If the tensor is a link and the credentials are not populated.\n\n        Returns:\n            A numpy array containing the data represented by this tensor.\n\n        Note:\n            For tensors of htype ``polygon``, aslist is always ``True``.\n        '
        ret = self.chunk_engine.numpy(self.index, aslist=aslist, fetch_chunks=fetch_chunks or self.is_iteration, pad_tensor=self.pad_tensor)
        if self.htype == 'point_cloud':
            if isinstance(ret, list):
                ret = [arr[..., :3] for arr in ret]
            else:
                ret = ret[..., :3]
        if self.htype == 'mesh':
            ret = get_mesh_vertices(self.key, self.index, ret, self.sample_info, aslist)
        dataset_read(self.dataset)
        return ret

    def summary(self):
        if False:
            i = 10
            return i + 15
        'Prints a summary of the tensor.'
        pretty_print = summary_tensor(self)
        print(self)
        print(pretty_print)

    def __str__(self):
        if False:
            return 10
        index_str = f', index={self.index}'
        if self.index.is_trivial():
            index_str = ''
        return f'Tensor(key={repr(self.meta.name or self.key)}{index_str})'
    __repr__ = __str__

    def __array__(self, dtype=None) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        ret = self.numpy()
        if self.base_htype == 'polygon':
            return np.array(ret, dtype=dtype)
        if dtype and ret.dtype != dtype:
            ret = ret.astype(dtype)
        return ret

    @_inplace_op
    def __iadd__(self, other):
        if False:
            return 10
        pass

    @_inplace_op
    def __isub__(self, other):
        if False:
            while True:
                i = 10
        pass

    @_inplace_op
    def __imul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        pass

    @_inplace_op
    def __itruediv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        pass

    @_inplace_op
    def __ifloordiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        pass

    @_inplace_op
    def __imod__(self, other):
        if False:
            print('Hello World!')
        pass

    @_inplace_op
    def __ipow__(self, other):
        if False:
            while True:
                i = 10
        pass

    @_inplace_op
    def __ilshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        pass

    @_inplace_op
    def __irshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        pass

    @_inplace_op
    def __iand__(self, other):
        if False:
            for i in range(10):
                print('nop')
        pass

    @_inplace_op
    def __ixor__(self, other):
        if False:
            while True:
                i = 10
        pass

    @_inplace_op
    def __ior__(self, other):
        if False:
            i = 10
            return i + 15
        pass

    def data(self, aslist: bool=False, fetch_chunks: bool=False) -> Any:
        if False:
            while True:
                i = 10
        'Returns data in the tensor in a format based on the tensor\'s base htype.\n\n        - If tensor has ``text`` base htype\n            - Returns dict with dict["value"] = :meth:`Tensor.text() <text>`\n\n        - If tensor has ``json`` base htype\n            - Returns dict with dict["value"] = :meth:`Tensor.dict() <dict>`\n\n        - If tensor has ``list`` base htype\n            - Returns dict with dict["value"] = :meth:`Tensor.list() <list>`\n\n        - For ``video`` tensors, returns a dict with keys "frames", "timestamps" and "sample_info":\n\n            - Value of dict["frames"] will be same as :meth:`numpy`.\n            - Value of dict["timestamps"] will be same as :attr:`timestamps` corresponding to the frames.\n            - Value of dict["sample_info"] will be same as :attr:`sample_info`.\n\n        - For ``class_label`` tensors, returns a dict with keys "value" and "text".\n\n            - Value of dict["value"] will be same as :meth:`numpy`.\n            - Value of dict["text"] will be list of class labels as strings.\n\n        - For ``image`` or ``dicom`` tensors, returns dict with keys "value" and "sample_info".\n\n            - Value of dict["value"] will be same as :meth:`numpy`.\n            - Value of dict["sample_info"] will be same as :attr:`sample_info`.\n\n        - For all else, returns dict with key "value" with value same as :meth:`numpy`.\n        '
        htype = self.base_htype
        if htype == 'text':
            return {'value': self.text(fetch_chunks=fetch_chunks)}
        if htype == 'json':
            return {'value': self.dict(fetch_chunks=fetch_chunks)}
        if htype == 'list':
            return {'value': self.list(fetch_chunks=fetch_chunks)}
        if self.htype == 'video':
            data = {}
            data['frames'] = self.numpy(aslist=aslist, fetch_chunks=fetch_chunks)
            index = self.index
            if index.values[0].subscriptable():
                root = Tensor(self.key, self.dataset)
                if len(index.values) > 1:
                    data['timestamps'] = np.array([root[i, index.values[1].value].timestamps for i in index.values[0].indices(self.num_samples)])
                else:
                    data['timestamps'] = [root[i].timestamps for i in index.values[0].indices(self.num_samples)]
            else:
                data['timestamps'] = self.timestamps
            if not aslist:
                try:
                    data['timestamps'] = np.array(data['timestamps'])
                except ValueError:
                    data['timestamps'] = np.array(data['timestamps'], dtype=object)
            data['sample_info'] = self.sample_info
            return data
        if htype == 'class_label':
            labels = self.numpy(aslist=aslist, fetch_chunks=fetch_chunks)
            data = {'value': labels}
            class_names = self.info.class_names
            if class_names:
                data['text'] = convert_to_text(labels, class_names)
            return data
        if htype in ('image', 'image.rgb', 'image.gray', 'dicom', 'nifti'):
            return {'value': self.numpy(aslist=aslist, fetch_chunks=fetch_chunks), 'sample_info': self.sample_info or {}}
        elif htype == 'point_cloud':
            full_arr = self.chunk_engine.numpy(self.index, aslist=aslist, pad_tensor=self.pad_tensor, fetch_chunks=fetch_chunks)
            value = parse_point_cloud_to_dict(full_arr, self.ndim, self.sample_info)
            return value
        elif htype == 'mesh':
            full_arr = self.chunk_engine.numpy(self.index, aslist=False, pad_tensor=self.pad_tensor)
            value = parse_mesh_to_dict(full_arr, self.sample_info)
            return value
        else:
            try:
                return {'value': self.chunk_engine.numpy(index=self.index, aslist=aslist, fetch_chunks=fetch_chunks)}
            except NotImplementedError:
                return {'value': self.numpy(aslist=aslist, fetch_chunks=fetch_chunks)}

    def tobytes(self) -> bytes:
        if False:
            print('Hello World!')
        'Returns the bytes of the tensor.\n\n        - Only works for a single sample of tensor.\n        - If the tensor is uncompressed, this returns the bytes of the numpy array.\n        - If the tensor is sample compressed, this returns the compressed bytes of the sample.\n        - If the tensor is chunk compressed, this raises an error.\n\n        Returns:\n            bytes: The bytes of the tensor.\n\n        Raises:\n            ValueError: If the tensor has multiple samples.\n        '
        if self.index.values[0].subscriptable() or len(self.index.values) > 1:
            raise ValueError('tobytes() can be used only on exactly 1 sample.')
        idx = self.index.values[0].value
        if self.pad_tensor and idx >= self.num_samples:
            ret = self.chunk_engine.get_empty_sample().tobytes()
        else:
            ret = self.chunk_engine.read_bytes_for_sample(idx)
        dataset_read(self.dataset)
        return ret

    def _extend_links(self, samples, flat: Optional[bool], progressbar: bool=False):
        if False:
            for i in range(10):
                print('nop')
        has_shape_tensor = False
        updated_tensors = {}
        try:
            for (k, v) in self.meta.links.items():
                if flat is None or v['flatten_sequence'] == flat:
                    tensor = self.version_state['full_tensors'][k]
                    func_name = v['extend']
                    if func_name == 'extend_shape':
                        has_shape_tensor = True
                    func = get_link_transform(func_name)
                    vs = func(samples, factor=tensor.info.downsampling_factor if func == extend_downsample else None, compression=self.meta.sample_compression, htype=self.htype, link_creds=self.link_creds, progressbar=progressbar, tensor_meta=self.meta)
                    dtype = tensor.dtype
                    if dtype:
                        if isinstance(vs, np.ndarray):
                            vs = cast_to_type(vs, dtype)
                        else:
                            vs = [cast_to_type(v, dtype) for v in vs]
                    updated_tensors[k] = tensor.num_samples
                    tensor.extend(vs)
        except Exception as e:
            for (k, num_samples) in updated_tensors.items():
                tensor = self.version_state['full_tensors'][k]
                num_samples_added = tensor.num_samples - num_samples
                for _ in range(num_samples_added):
                    tensor.pop()
            raise

    def _update_links(self, global_sample_index: int, sub_index: Index, new_sample, flat: Optional[bool]):
        if False:
            i = 10
            return i + 15
        has_shape_tensor = False
        for (k, v) in self.meta.links.items():
            if flat is None or v['flatten_sequence'] == flat:
                fname = v.get('update')
                if not fname:
                    continue
                if fname == 'update_shape':
                    has_shape_tensor = True
                func = get_link_transform(fname)
                tensor = self.version_state['full_tensors'][k]
                is_partial = not sub_index.is_trivial()
                val = func(new_sample, old_value=tensor[global_sample_index], factor=tensor.info.downsampling_factor if func == update_downsample else None, compression=self.meta.sample_compression, htype=self.htype, link_creds=self.link_creds, sub_index=sub_index, partial=is_partial, tensor_meta=self.meta)
                if val is not _NO_LINK_UPDATE:
                    if is_partial and func == update_downsample:
                        apply_partial_downsample(tensor, global_sample_index, val)
                    else:
                        val = cast_to_type(val, tensor.dtype)
                        tensor[global_sample_index] = val

    def _check_for_pop(self, index: Optional[int]=None):
        if False:
            while True:
                i = 10
        if index is not None and index != self.num_samples - 1 and (self.meta.htype == 'embedding') and (len(self.get_vdb_indexes()) > 0):
            raise EmbeddingTensorPopError(self.meta.name, index)

    def _pop(self, index: Optional[int]=None):
        if False:
            return 10
        sample_id_tensor = self._sample_id_tensor
        if index is None:
            index = self.num_samples - 1
        sample_id = int(sample_id_tensor[index].numpy()) if sample_id_tensor else None
        self.chunk_engine.pop(index, link_callback=self._pop_links if self.meta.links else None, sample_id=sample_id)
        self.invalidate_libdeeplake_dataset()

    @invalid_view_op
    def pop(self, index: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        'Removes an element at the given index.'
        self._check_for_pop(index)
        self._pop(index)
        if index_maintenance.is_embedding_tensor(self):
            row_ids = [index if index is not None else self.num_samples - 1]
            index_maintenance.index_operation_dataset(self.dataset, dml_type=_INDEX_OPERATION_MAPPING['REMOVE'], rowids=row_ids)

    def _pop_links(self, global_sample_index: int):
        if False:
            while True:
                i = 10
        rev_tensor_names = {v: k for (k, v) in self.dataset.meta.tensor_names.items()}
        if self.meta.is_sequence:
            flat_links: List[str] = []
            links: List[str] = []
            for (link, props) in self.meta.links.items():
                (flat_links if props['flatten_sequence'] else links).append(link)
            if flat_links:
                seq_enc = self.chunk_engine.sequence_encoder
                assert seq_enc is not None
                for link in flat_links:
                    link_tensor = self.dataset[rev_tensor_names.get(link)]
                    for idx in reversed(range(*seq_enc[global_sample_index])):
                        link_tensor.pop(idx)
        else:
            links = list(self.meta.links.keys())
        [self.dataset[rev_tensor_names.get(link)].pop(global_sample_index) for link in links]

    def _all_tensor_links(self):
        if False:
            return 10
        ds = self.dataset
        return [ds.version_state['full_tensors'][ds.version_state['tensor_names'][l]] for l in self.meta.links]

    @property
    def _sample_info_tensor(self):
        if False:
            return 10
        ds = self.dataset
        tensor_name = self.meta.name or self.key
        return ds.version_state['full_tensors'].get(ds.version_state['tensor_names'].get(get_sample_info_tensor_key(tensor_name)))

    @property
    def _sample_shape_tensor(self):
        if False:
            print('Hello World!')
        ds = self.dataset
        tensor_name = self.meta.name or self.key
        return ds.version_state['full_tensors'].get(ds.version_state['tensor_names'].get(get_sample_shape_tensor_key(tensor_name)))

    @property
    def _sample_id_tensor(self):
        if False:
            print('Hello World!')
        tensor_name = self.meta.name or self.key
        return self.dataset._tensors().get(get_sample_id_tensor_key(tensor_name))

    def _sample_shape_provider(self, sample_shape_tensor) -> Callable:
        if False:
            return 10
        if self.is_sequence:

            def get_sample_shape(global_sample_index: int):
                if False:
                    while True:
                        i = 10
                assert self.chunk_engine.sequence_encoder is not None
                seq_pos = slice(*self.chunk_engine.sequence_encoder[global_sample_index])
                idx = Index([IndexEntry(seq_pos)])
                shapes = sample_shape_tensor[idx].numpy(fetch_chunks=True)
                return shapes
        else:

            def get_sample_shape(global_sample_index: int):
                if False:
                    while True:
                        i = 10
                return tuple(sample_shape_tensor[global_sample_index].numpy(fetch_chunks=True).tolist())
        return get_sample_shape

    def _get_sample_info_at_index(self, global_sample_index: int, sample_info_tensor):
        if False:
            i = 10
            return i + 15
        if self.is_sequence:
            assert self.chunk_engine.sequence_encoder is not None
            return [sample_info_tensor[i].data() for i in range(*self.chunk_engine.sequence_encoder[global_sample_index])]
        return sample_info_tensor[global_sample_index].data()['value']

    def _sample_info(self, index: Index):
        if False:
            while True:
                i = 10
        sample_info_tensor = self._sample_info_tensor
        if sample_info_tensor is None:
            return None
        if index.subscriptable_at(0):
            return list(map(partial(self._get_sample_info_at_index, sample_info_tensor=sample_info_tensor), index.values[0].indices(self.num_samples)))
        return self._get_sample_info_at_index(index.values[0].value, sample_info_tensor)

    @property
    def sample_info(self) -> Union[Dict, List[Dict]]:
        if False:
            i = 10
            return i + 15
        "Returns info about particular samples in a tensor. Returns dict in case of single sample, otherwise list of dicts.\n        Data in returned dict would depend on the tensor's htype and the sample itself.\n\n        Example:\n\n            >>> ds.videos[0].sample_info\n            {'duration': 400400, 'fps': 29.97002997002997, 'timebase': 3.3333333333333335e-05, 'shape': [400, 360, 640, 3], 'format': 'mp4', 'filename': '../deeplake/tests/dummy_data/video/samplemp4.mp4', 'modified': False}\n            >>> ds.images[:2].sample_info\n            [{'exif': {'Software': 'Google'}, 'shape': [900, 900, 3], 'format': 'jpeg', 'filename': '../deeplake/tests/dummy_data/images/cat.jpeg', 'modified': False}, {'exif': {}, 'shape': [495, 750, 3], 'format': 'jpeg', 'filename': '../deeplake/tests/dummy_data/images/car.jpg', 'modified': False}]\n        "
        return self._sample_info(self.index)

    def _linked_sample(self):
        if False:
            while True:
                i = 10
        "Returns the linked sample at the given index. This is only applicable for tensors of ``link[]`` htype\n        and can only be used for exactly one sample.\n\n        >>> linked_sample = ds.abc[0]._linked_sample().path\n        'https://picsum.photos/200/300'\n\n        "
        if not self.is_link:
            raise ValueError('Not supported as the tensor is not a link.')
        if self.index.values[0].subscriptable() or len(self.index.values) > 1:
            raise ValueError('_linked_sample can be used only on exatcly 1 sample.')
        return self.chunk_engine.linked_sample(self.index.values[0].value)

    def _get_video_stream_url(self):
        if False:
            return 10
        if self.is_link:
            return self.chunk_engine.get_video_url(self.index.values[0].value)[0]
        from deeplake.visualizer.video_streaming import get_video_stream_url
        return get_video_stream_url(self, self.index.values[0].value)

    def play(self):
        if False:
            for i in range(10):
                print('nop')
        'Play video sample. Plays video in Jupyter notebook or plays in web browser. Video is streamed directly from storage.\n        This method will fail for incompatible htypes.\n\n        Example:\n\n            >>> ds = deeplake.load("./test/my_video_ds")\n            >>> # play second sample\n            >>> ds.videos[2].play()\n\n        Note:\n            Video streaming is not yet supported on colab.\n        '
        if get_compression_type(self.meta.sample_compression) != VIDEO_COMPRESSION and self.htype != 'link[video]':
            raise Exception('Only supported for video tensors.')
        if self.index.values[0].subscriptable():
            raise ValueError('Video streaming requires exactly 1 sample.')
        if len(self.index.values) > 1:
            warnings.warn('Sub indexes to video sample will be ignored while streaming.')
        if is_colab():
            raise NotImplementedError('Video streaming is not supported on colab yet.')
        elif is_jupyter():
            return video_html(src=self._get_video_stream_url(), alt=f'{self.key}[{self.index.values[0].value}]')
        else:
            webbrowser.open(self._get_video_stream_url())

    @property
    def timestamps(self) -> np.ndarray:
        if False:
            print('Hello World!')
        'Returns timestamps (in seconds) for video sample as numpy array.\n\n        Example:\n\n            >>> # Return timestamps for all frames of first video sample\n            >>> ds.videos[0].timestamps.shape\n            (400,)\n            >>> # Return timestamps for 5th to 10th frame of first video sample\n            >>> ds.videos[0, 5:10].timestamps\n            array([0.2002    , 0.23356667, 0.26693332, 0.33366665, 0.4004    ],\n            dtype=float32)\n\n        '
        if get_compression_type(self.meta.sample_compression) != VIDEO_COMPRESSION and self.htype != 'link[video]':
            raise Exception('Only supported for video tensors.')
        index = self.index
        if index.values[0].subscriptable():
            raise ValueError('Only supported for exactly 1 video sample.')
        if self.is_sequence:
            if len(index.values) == 1 or index.values[1].subscriptable():
                raise ValueError('Only supported for exactly 1 video sample.')
            sub_index = index.values[2].value if len(index.values) > 2 else None
        else:
            sub_index = index.values[1].value if len(index.values) > 1 else None
        global_sample_index = next(index.values[0].indices(self.num_samples))
        if self.is_link:
            sample = self.chunk_engine.get_video_url(global_sample_index)[0]
        else:
            sample = self.chunk_engine.get_video_sample(global_sample_index, index, decompress=False)
        nframes = self.shape[0]
        (start, stop, step, reverse) = normalize_index(sub_index, nframes)
        stamps = _read_timestamps(sample, start, stop, step, reverse)
        return stamps

    @property
    def _config(self):
        if False:
            return 10
        'Returns a summary of the configuration of the tensor.'
        tensor_meta = self.meta
        return {'htype': tensor_meta.htype or UNSPECIFIED, 'dtype': tensor_meta.dtype or UNSPECIFIED, 'sample_compression': tensor_meta.sample_compression or UNSPECIFIED, 'chunk_compression': tensor_meta.chunk_compression or UNSPECIFIED, 'hidden': tensor_meta.hidden, 'is_link': tensor_meta.is_link, 'is_sequence': tensor_meta.is_sequence}

    @property
    def sample_indices(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns all the indices pointed to by this tensor in the dataset view.'
        return self.dataset._sample_indices(self.num_samples)

    def _extract_value(self, htype: str, fetch_chunks: bool=False):
        if False:
            while True:
                i = 10
        if self.base_htype != htype:
            raise Exception(f'Only supported for {htype} tensors.')
        if self.ndim == 1:
            return self.numpy(fetch_chunks=fetch_chunks)[0]
        else:
            return [sample[0] for sample in self.numpy(aslist=True)]

    def text(self, fetch_chunks: bool=False):
        if False:
            i = 10
            return i + 15
        "Return text data. Only applicable for tensors with 'text' base htype."
        return self._extract_value('text', fetch_chunks=fetch_chunks)

    def dict(self, fetch_chunks: bool=False):
        if False:
            for i in range(10):
                print('nop')
        "Return json data. Only applicable for tensors with 'json' base htype."
        return self._extract_value('json', fetch_chunks=fetch_chunks)

    def list(self, fetch_chunks: bool=False):
        if False:
            return 10
        "Return list data. Only applicable for tensors with 'list' base htype."
        if self.base_htype != 'list':
            raise Exception('Only supported for list tensors.')
        if self.ndim == 1:
            return list(self.numpy(fetch_chunks=fetch_chunks))
        else:
            return list(map(list, self.numpy(aslist=True, fetch_chunks=fetch_chunks)))

    def path(self, aslist: bool=True, fetch_chunks: bool=False):
        if False:
            return 10
        'Return path data. Only applicable for linked tensors.\n\n        Args:\n            aslist (bool): Returns links in a list if ``True``.\n            fetch_chunks (bool): If ``True``, full chunks will be retrieved from the storage, otherwise only required bytes will be retrieved.\n\n        Returns:\n            Union[np.ndarray, List]: A list or numpy array of links.\n\n        Raises:\n            Exception: If the tensor is not a linked tensor.\n        '
        if not self.is_link:
            raise Exception('Only supported for linked tensors.')
        assert isinstance(self.chunk_engine, LinkedChunkEngine)
        return self.chunk_engine.path(self.index, aslist=aslist, fetch_chunks=fetch_chunks)

    def creds_key(self):
        if False:
            i = 10
            return i + 15
        'Return path data. Only applicable for linked tensors'
        if not self.is_link:
            raise Exception('Only supported for linked tensors.')
        if self.index.values[0].subscriptable() or len(self.index.values) > 1:
            raise ValueError('_linked_sample can be used only on exatcly 1 sample.')
        assert isinstance(self.chunk_engine, LinkedChunkEngine)
        if self.is_sequence:
            indices = range(*self.chunk_engine.sequence_encoder[self.index.values[0].value])
            return [self.chunk_engine.creds_key(i) for i in indices]
        return self.chunk_engine.creds_key(self.index.values[0].value)

    def invalidate_libdeeplake_dataset(self):
        if False:
            print('Hello World!')
        'Invalidates the libdeeplake dataset object.'
        self.dataset.libdeeplake_dataset = None

    def update_vdb_index(self, operation_kind: int, row_ids: List[int]=[]):
        if False:
            while True:
                i = 10
        self.storage.check_readonly()
        if self.meta.htype != 'embedding':
            raise Exception(f'Only supported for embedding tensors.')
        self.invalidate_libdeeplake_dataset()
        self.dataset.flush()
        from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake
        ds = dataset_to_libdeeplake(self.dataset)
        ts = getattr(ds, self.meta.name)
        from deeplake.enterprise.convert_to_libdeeplake import import_indra_api
        api = import_indra_api()
        commit_id = self.version_state['commit_id']
        if operation_kind == _INDEX_OPERATION_MAPPING['ADD']:
            try:
                indexes = api.vdb.add_samples_to_index(ts, add_indices=row_ids)
                for (id, index) in indexes:
                    b = index.serialize()
                    commit_id = self.version_state['commit_id']
                    self.storage[get_tensor_vdb_index_key(self.key, commit_id, id)] = b
                self.storage.flush()
            except:
                raise
        elif operation_kind == _INDEX_OPERATION_MAPPING['REMOVE']:
            try:
                indexes = api.vdb.remove_samples_from_index(ts, remove_indices=row_ids)
                for (id, index) in indexes:
                    b = index.serialize()
                    commit_id = self.version_state['commit_id']
                    self.storage[get_tensor_vdb_index_key(self.key, commit_id, id)] = b
                self.storage.flush()
            except:
                raise
        elif operation_kind == _INDEX_OPERATION_MAPPING['UPDATE']:
            try:
                indexes = api.vdb.update_samples_in_index(ts, update_indices=row_ids)
                for (id, index) in indexes:
                    b = index.serialize()
                    commit_id = self.version_state['commit_id']
                    self.storage[get_tensor_vdb_index_key(self.key, commit_id, id)] = b
                    self.storage.flush()
                self.storage.flush()
            except:
                raise
        else:
            raise AssertionError(f'Invalid operation_kind: {operation_kind}')

    def create_vdb_index(self, id: str, distance: Union[DistanceType, str]=DistanceType.L2_NORM, additional_params: Optional[Dict[str, int]]=None):
        if False:
            return 10
        self.storage.check_readonly()
        if self.meta.htype != 'embedding':
            raise Exception(f'Only supported for embedding tensors.')
        if not self.dataset.libdeeplake_dataset is None:
            ds = self.dataset.libdeeplake_dataset
        else:
            from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake
            ds = dataset_to_libdeeplake(self.dataset)
        ts = getattr(ds, self.meta.name)
        from indra import api
        if type(distance) == DistanceType:
            distance = distance.value
        self.meta.add_vdb_index(id=id, type='hnsw', distance=distance, additional_params=additional_params)
        try:
            if additional_params is None:
                index = api.vdb.generate_index(ts, index_type='hnsw', distance_type=distance)
            else:
                index = api.vdb.generate_index(ts, index_type='hnsw', distance_type=distance, param=additional_params)
            b = index.serialize()
            commit_id = self.version_state['commit_id']
            self.storage[get_tensor_vdb_index_key(self.key, commit_id, id)] = b
            self.invalidate_libdeeplake_dataset()
        except:
            self.meta.remove_vdb_index(id=id)
            raise
        return index

    def delete_vdb_index(self, id: str):
        if False:
            print('Hello World!')
        self.storage.check_readonly()
        if self.meta.htype != 'embedding':
            raise Exception(f'Only supported for embedding tensors.')
        commit_id = self.version_state['commit_id']
        self.unload_vdb_index_cache()
        self.storage.pop(get_tensor_vdb_index_key(self.key, commit_id, id))
        self.meta.remove_vdb_index(id=id)
        self.invalidate_libdeeplake_dataset()
        self.storage.flush()

    def _verify_and_delete_vdb_indexes(self):
        if False:
            while True:
                i = 10
        try:
            is_embedding = self.htype == 'embedding'
            has_vdb_indexes = hasattr(self.meta, 'vdb_indexes')
            try:
                vdb_index_ids_present = len(self.meta.vdb_indexes) > 0
            except AttributeError:
                vdb_index_ids_present = False
            if is_embedding and has_vdb_indexes and vdb_index_ids_present:
                for vdb_index in self.meta.vdb_indexes:
                    id = vdb_index['id']
                    self.delete_vdb_index(id)
        except Exception as e:
            raise Exception(f'An error occurred while deleting VDB indexes: {e}')

    def load_vdb_index(self, id: str, path: Optional[str]=None):
        if False:
            print('Hello World!')
        if self.meta.htype != 'embedding':
            raise Exception(f'Only supported for embedding tensors.')
        if not self.meta.contains_vdb_index(id):
            raise ValueError(f"Tensor meta has no vdb index with name '{id}'.")
        if not self.dataset.libdeeplake_dataset is None:
            ds = self.dataset.libdeeplake_dataset
        else:
            from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake
            ds = dataset_to_libdeeplake(self.dataset)
        ts = getattr(ds, self.meta.name)
        from indra import api
        index_meta = next((x for x in self.meta.vdb_indexes if x['id'] == id))
        commit_id = self.version_state['commit_id']
        b = self.chunk_engine.base_storage[get_tensor_vdb_index_key(self.key, commit_id, id)]
        if path is None:
            return api.vdb.load_index(ts, b, index_type=index_meta['type'], distance_type=index_meta['distance'])
        else:
            return api.vdb.load_index(ts, b, index_type=index_meta['type'], distance_type=index_meta['distance'], path=path)

    def unload_vdb_index_cache(self):
        if False:
            while True:
                i = 10
        if self.meta.htype != 'embedding':
            raise Exception(f'Only supported for embedding tensors.')
        if not self.dataset.libdeeplake_dataset is None:
            ds = self.dataset.libdeeplake_dataset
        else:
            from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake
            ds = dataset_to_libdeeplake(self.dataset)
        ts = getattr(ds, self.meta.name)
        from indra import api
        try:
            api.vdb.unload_index_cache(ts)
        except Exception as e:
            raise Exception(f'An error occurred while cleaning VDB Cache: {e}')

    def get_vdb_indexes(self) -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        if self.meta.htype != 'embedding':
            raise Exception(f'Only supported for embedding tensors.')
        return self.meta.vdb_indexes

    def fetch_vdb_indexes(self) -> List[Dict[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        vdb_indexes = []
        if self.meta.htype == 'embedding':
            if not self.meta.vdb_indexes is None and len(self.meta.vdb_indexes) > 0:
                vdb_indexes.extend(self.meta.vdb_indexes)
        return vdb_indexes

    def _check_compatibility_with_htype(self, htype):
        if False:
            return 10
        'Checks if the tensor is compatible with the given htype.\n        Raises an error if not compatible.\n        '
        (is_sequence, is_link, htype) = parse_complex_htype(htype)
        if is_sequence or is_link:
            raise ValueError(f'Cannot change htype to a sequence or link.')
        _validate_htype_exists(htype)
        if self.htype not in HTYPE_CONVERSION_LHS:
            raise NotImplementedError(f'Changing the htype of a tensor of htype {self.htype} is not supported.')
        if htype not in HTYPE_CONSTRAINTS:
            raise NotImplementedError(f'Changing the htype to {htype} is not supported.')
        compression = self.meta.sample_compression or self.meta.chunk_compression
        if compression:
            supported_compressions = HTYPE_SUPPORTED_COMPRESSIONS.get(htype)
            if supported_compressions and compression not in supported_compressions:
                raise UnsupportedCompressionError(compression, htype)
        constraints = HTYPE_CONSTRAINTS[htype]
        constraints(self.shape, self.dtype)