"""BioSequenceDataSet loads and saves data to/from bio-sequence objects to
file.
"""
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Dict, List
import fsspec
from Bio import SeqIO
from kedro.io.core import AbstractDataset, get_filepath_str, get_protocol_and_path

class BioSequenceDataSet(AbstractDataset[List, List]):
    """``BioSequenceDataSet`` loads and saves data to a sequence file.

    Example:
    ::

        >>> from kedro.extras.datasets.biosequence import BioSequenceDataSet
        >>> from io import StringIO
        >>> from Bio import SeqIO
        >>>
        >>> data = ">Alpha\\nACCGGATGTA\\n>Beta\\nAGGCTCGGTTA\\n"
        >>> raw_data = []
        >>> for record in SeqIO.parse(StringIO(data), "fasta"):
        >>>     raw_data.append(record)
        >>>
        >>> data_set = BioSequenceDataSet(filepath="ls_orchid.fasta",
        >>>                               load_args={"format": "fasta"},
        >>>                               save_args={"format": "fasta"})
        >>> data_set.save(raw_data)
        >>> sequence_list = data_set.load()
        >>>
        >>> assert raw_data[0].id == sequence_list[0].id
        >>> assert raw_data[0].seq == sequence_list[0].seq

    """
    DEFAULT_LOAD_ARGS = {}
    DEFAULT_SAVE_ARGS = {}

    def __init__(self, filepath: str, load_args: Dict[str, Any]=None, save_args: Dict[str, Any]=None, credentials: Dict[str, Any]=None, fs_args: Dict[str, Any]=None) -> None:
        if False:
            return 10
        '\n        Creates a new instance of ``BioSequenceDataSet`` pointing\n        to a concrete filepath.\n\n        Args:\n            filepath: Filepath in POSIX format to sequence file prefixed with a protocol like\n                `s3://`. If prefix is not provided, `file` protocol (local filesystem) will be used.\n                The prefix should be any protocol supported by ``fsspec``.\n            load_args: Options for parsing sequence files by Biopython ``SeqIO.parse()``.\n            save_args: file format supported by Biopython ``SeqIO.write()``.\n                E.g. `{"format": "fasta"}`.\n            credentials: Credentials required to get access to the underlying filesystem.\n                E.g. for ``GCSFileSystem`` it should look like `{"token": None}`.\n            fs_args: Extra arguments to pass into underlying filesystem class constructor\n                (e.g. `{"project": "my-project"}` for ``GCSFileSystem``), as well as\n                to pass to the filesystem\'s `open` method through nested keys\n                `open_args_load` and `open_args_save`.\n                Here you can find all available arguments for `open`:\n                https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.open\n                All defaults are preserved, except `mode`, which is set to `r` when loading\n                and to `w` when saving.\n\n        Note: Here you can find all supported file formats: https://biopython.org/wiki/SeqIO\n        '
        _fs_args = deepcopy(fs_args) or {}
        _fs_open_args_load = _fs_args.pop('open_args_load', {})
        _fs_open_args_save = _fs_args.pop('open_args_save', {})
        _credentials = deepcopy(credentials) or {}
        (protocol, path) = get_protocol_and_path(filepath)
        self._filepath = PurePosixPath(path)
        self._protocol = protocol
        if protocol == 'file':
            _fs_args.setdefault('auto_mkdir', True)
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        _fs_open_args_load.setdefault('mode', 'r')
        _fs_open_args_save.setdefault('mode', 'w')
        self._fs_open_args_load = _fs_open_args_load
        self._fs_open_args_save = _fs_open_args_save

    def _describe(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return {'filepath': self._filepath, 'protocol': self._protocol, 'load_args': self._load_args, 'save_args': self._save_args}

    def _load(self) -> List:
        if False:
            i = 10
            return i + 15
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path, **self._fs_open_args_load) as fs_file:
            return list(SeqIO.parse(handle=fs_file, **self._load_args))

    def _save(self, data: List) -> None:
        if False:
            for i in range(10):
                print('nop')
        save_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(save_path, **self._fs_open_args_save) as fs_file:
            SeqIO.write(data, handle=fs_file, **self._save_args)

    def _exists(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        load_path = get_filepath_str(self._filepath, self._protocol)
        return self._fs.exists(load_path)

    def _release(self) -> None:
        if False:
            i = 10
            return i + 15
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        if False:
            print('Hello World!')
        'Invalidate underlying filesystem caches.'
        filepath = get_filepath_str(self._filepath, self._protocol)
        self._fs.invalidate_cache(filepath)