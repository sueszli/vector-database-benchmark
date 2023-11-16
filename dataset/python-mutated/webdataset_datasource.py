import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
if TYPE_CHECKING:
    import pyarrow

def _base_plus_ext(path: str):
    if False:
        print('Hello World!')
    'Split off all file extensions.\n\n    Returns base, allext.\n\n    Args:\n        path: path with extensions\n\n    Returns:\n        str: path with all extensions removed\n    '
    match = re.match('^((?:.*/|)[^.]+)[.]([^/]*)$', path)
    if not match:
        return (None, None)
    return (match.group(1), match.group(2))

def _valid_sample(sample: Dict[str, Any]):
    if False:
        i = 10
        return i + 15
    'Check whether a sample is valid.\n\n    Args:\n        sample: sample to be checked\n    '
    return sample is not None and isinstance(sample, dict) and (len(list(sample.keys())) > 0) and (not sample.get('__bad__', False))

def _apply_list(f: Union[Callable, List[Callable]], sample: Dict[str, Any], default: Callable=None):
    if False:
        return 10
    'Apply a list of functions to a sample.\n\n    Args:\n        f: function or list of functions\n        sample: sample to be modified\n        default: default function to be applied to all keys.\n            Defaults to None.\n\n    Returns:\n        modified sample\n    '
    if f is None:
        return sample
    if not isinstance(f, list):
        f = [f]
    for g in f:
        if default is not None and (not callable(g)):
            g = partial(default, format=g)
        sample = g(sample)
    return sample

def _check_suffix(suffix: str, suffixes: Union[list, callable]):
    if False:
        print('Hello World!')
    'Check whether a suffix is valid.\n\n    Suffixes can be either None (=accept everything), a callable,\n    or a list of patterns. If the pattern contains */? it is treated\n    as a glob pattern, otherwise it is treated as a literal.\n\n    Args:\n        suffix: suffix to be checked\n        suffixes: list of valid suffixes\n    '
    if suffixes is None:
        return True
    if callable(suffixes):
        return suffixes(suffix)
    for pattern in suffixes:
        if '*' in pattern or '?' in pattern:
            if fnmatch.fnmatch('.' + suffix, pattern):
                return True
        elif suffix == pattern or '.' + suffix == pattern:
            return True
    return False

def _tar_file_iterator(fileobj: Any, fileselect: Optional[Union[bool, callable, list]]=None, filerename: Optional[Union[bool, callable, list]]=None, verbose_open: bool=False, meta: dict=None):
    if False:
        i = 10
        return i + 15
    'Iterate over tar file, yielding filename, content pairs for the given tar stream.\n\n    Args:\n        fileobj: file object\n        fileselect: patterns or function selecting\n            files to be selected\n        meta: metadata to be added to each sample\n    '
    meta = meta or {}
    stream = tarfile.open(fileobj=fileobj, mode='r|*')
    if verbose_open:
        print(f'start {meta}')
    for tarinfo in stream:
        fname = tarinfo.name
        if not tarinfo.isreg() or fname is None:
            continue
        data = stream.extractfile(tarinfo).read()
        fname = _apply_list(filerename, fname)
        assert isinstance(fname, str)
        if not _check_suffix(fname, fileselect):
            continue
        result = dict(fname=fname, data=data)
        yield result
    if verbose_open:
        print(f'done {meta}')

def _group_by_keys(data: List[Dict[str, Any]], keys: callable=_base_plus_ext, suffixes: Optional[Union[list, callable]]=None, meta: dict=None):
    if False:
        for i in range(10):
            print('nop')
    'Return function over iterator that groups key, value pairs into samples.\n\n    Args:\n        data: iterator over key, value pairs\n        keys: function that returns key, suffix for a given key\n        suffixes: list of suffixes to be included in the sample\n        meta: metadata to be added to each sample\n    '
    meta = meta or {}
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        (fname, value) = (filesample['fname'], filesample['data'])
        (prefix, suffix) = keys(fname)
        if prefix is None:
            continue
        if current_sample is None or prefix != current_sample['__key__']:
            if _valid_sample(current_sample):
                current_sample.update(meta)
                yield current_sample
            current_sample = dict(__key__=prefix)
            if '__url__' in filesample:
                current_sample['__url__'] = filesample['__url__']
        if suffix in current_sample:
            raise ValueError(f'{fname}: duplicate file name in tar file ' + f'{suffix} {current_sample.keys()}')
        if suffixes is None or _check_suffix(suffix, suffixes):
            current_sample[suffix] = value
    if _valid_sample(current_sample):
        current_sample.update(meta)
        yield current_sample

def _default_decoder(sample: Dict[str, Any], format: Optional[Union[bool, str]]=True):
    if False:
        for i in range(10):
            print('nop')
    'A default decoder for webdataset.\n\n    This handles common file extensions: .txt, .cls, .cls2,\n        .jpg, .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl.\n    These are the most common extensions used in webdataset.\n    For other extensions, users can provide their own decoder.\n\n    Args:\n        sample: sample, modified in place\n    '
    sample = dict(sample)
    for (key, value) in sample.items():
        extension = key.split('.')[-1]
        if key.startswith('__'):
            continue
        elif extension in ['txt', 'text']:
            sample[key] = value.decode('utf-8')
        elif extension in ['cls', 'cls2']:
            sample[key] = int(value.decode('utf-8'))
        elif extension in ['jpg', 'png', 'ppm', 'pgm', 'pbm', 'pnm']:
            import numpy as np
            import PIL.Image
            if format == 'PIL':
                sample[key] = PIL.Image.open(io.BytesIO(value))
            else:
                sample[key] = np.asarray(PIL.Image.open(io.BytesIO(value)))
        elif extension == 'json':
            import json
            sample[key] = json.loads(value)
        elif extension == 'npy':
            import numpy as np
            sample[key] = np.load(io.BytesIO(value))
        elif extension == 'mp':
            import msgpack
            sample[key] = msgpack.unpackb(value, raw=False)
        elif extension in ['pt', 'pth']:
            import torch
            sample[key] = torch.load(io.BytesIO(value))
        elif extension in ['pickle', 'pkl']:
            import pickle
            sample[key] = pickle.loads(value)
    return sample
extension_to_format = {'jpg': 'jpeg'}

def _default_encoder(sample: Dict[str, Any], format: Optional[Union[str, bool]]=True):
    if False:
        while True:
            i = 10
    'A default encoder for webdataset.\n\n    This handles common file extensions: .txt, .cls, .cls2, .jpg,\n        .png, .json, .npy, .mp, .pt, .pth, .pickle, .pkl\n    These are the most common extensions used in webdataset.\n    For other extensions, users can provide their own encoder.\n\n    Args:\n        sample (Dict[str, Any]): sample\n    '
    sample = dict(sample)
    for (key, value) in sample.items():
        extension = key.split('.')[-1]
        if key.startswith('__'):
            continue
        elif extension in ['txt']:
            sample[key] = value.encode('utf-8')
        elif extension in ['cls', 'cls2']:
            sample[key] = str(value).encode('utf-8')
        elif extension in ['jpg', 'jpeg', 'png', 'ppm', 'pgm', 'pbm', 'pnm']:
            import numpy as np
            import PIL.Image
            if isinstance(value, np.ndarray):
                value = PIL.Image.fromarray(value)
            assert isinstance(value, PIL.Image.Image)
            stream = io.BytesIO()
            value.save(stream, format=extension_to_format.get(extension.lower(), extension))
            sample[key] = stream.getvalue()
        elif extension == 'json':
            import json
            sample[key] = json.dumps(value).encode('utf-8')
        elif extension == 'npy':
            import numpy as np
            stream = io.BytesIO()
            np.save(stream, value)
            sample[key] = stream.getvalue()
        elif extension == 'mp':
            import msgpack
            sample[key] = msgpack.dumps(value)
        elif extension in ['pt', 'pth']:
            import torch
            stream = io.BytesIO()
            torch.save(value, stream)
            sample[key] = stream.getvalue()
        elif extension in ['pickle', 'pkl']:
            import pickle
            stream = io.BytesIO()
            pickle.dump(value, stream)
            sample[key] = stream.getvalue()
    return sample

def _make_iterable(block: BlockAccessor):
    if False:
        print('Hello World!')
    'Make a block iterable.\n\n    This is a placeholder for dealing with more complex blocks.\n\n    Args:\n        block: Ray Dataset block\n\n    Returns:\n        Iterable[Dict[str,Any]]: Iterable of samples\n    '
    return block.iter_rows(public_row_format=False)

@PublicAPI(stability='alpha')
class WebDatasetDatasource(FileBasedDatasource):
    """A Datasource for WebDataset datasets (tar format with naming conventions)."""
    _FILE_EXTENSIONS = ['tar']

    def __init__(self, paths: Union[str, List[str]], decoder: Optional[Union[bool, str, callable, list]]=True, fileselect: Optional[Union[bool, callable, list]]=None, filerename: Optional[Union[bool, callable, list]]=None, suffixes: Optional[Union[bool, callable, list]]=None, verbose_open: bool=False, **file_based_datasource_kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(paths, **file_based_datasource_kwargs)
        self.decoder = decoder
        self.fileselect = fileselect
        self.filerename = filerename
        self.suffixes = suffixes
        self.verbose_open = verbose_open

    def _read_stream(self, stream: 'pyarrow.NativeFile', path: str):
        if False:
            return 10
        'Read and decode samples from a stream.\n\n        Note that fileselect selects files during reading, while suffixes\n        selects files during the grouping step.\n\n        Args:\n            stream: File descriptor to read from.\n            path: Path to the data.\n            decoder: decoder or list of decoders to be applied to samples\n            fileselect: Predicate for skipping files in tar decoder.\n                Defaults to lambda_:False.\n            suffixes: List of suffixes to be extracted. Defaults to None.\n            verbose_open: Print message when opening files. Defaults to False.\n\n        Yields:\n            List[Dict[str, Any]]: List of sample (list of length 1).\n        '
        import pandas as pd
        files = _tar_file_iterator(stream, fileselect=self.fileselect, filerename=self.filerename, verbose_open=self.verbose_open)
        samples = _group_by_keys(files, meta=dict(__url__=path), suffixes=self.suffixes)
        for sample in samples:
            if self.decoder is not None:
                sample = _apply_list(self.decoder, sample, default=_default_decoder)
            yield pd.DataFrame({k: [v] for (k, v) in sample.items()})