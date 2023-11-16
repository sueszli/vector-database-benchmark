import posixpath
from typing import TYPE_CHECKING, Optional
from ray.util.annotations import DeveloperAPI
if TYPE_CHECKING:
    import pyarrow

@DeveloperAPI
class BlockWritePathProvider:
    """Abstract callable that provides concrete output paths when writing
    dataset blocks.

    Current subclasses:
        DefaultBlockWritePathProvider
    """

    def _get_write_path_for_block(self, base_path: str, *, filesystem: Optional['pyarrow.fs.FileSystem']=None, dataset_uuid: Optional[str]=None, task_index: Optional[int]=None, block_index: Optional[int]=None, file_format: Optional[str]=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Resolves and returns the write path for the given dataset block. When\n        implementing this method, care should be taken to ensure that a unique\n        path is provided for every dataset block.\n\n        Args:\n            base_path: The base path to write the dataset block out to. This is\n                expected to be the same for all blocks in the dataset, and may\n                point to either a directory or file prefix.\n            filesystem: The filesystem implementation that will be used to\n                write a file out to the write path returned.\n            dataset_uuid: Unique identifier for the dataset that this block\n                belongs to.\n            block: The block to write.\n            task_index: Ordered index of the write task within its parent\n                dataset.\n            block_index: Ordered index of the block to write within its parent\n                write task.\n            file_format: File format string for the block that can be used as\n                the file extension in the write path returned.\n\n        Returns:\n            The dataset block write path.\n        '
        raise NotImplementedError

    def __call__(self, base_path: str, *, filesystem: Optional['pyarrow.fs.FileSystem']=None, dataset_uuid: Optional[str]=None, task_index: Optional[int]=None, block_index: Optional[int]=None, file_format: Optional[str]=None) -> str:
        if False:
            i = 10
            return i + 15
        return self._get_write_path_for_block(base_path, filesystem=filesystem, dataset_uuid=dataset_uuid, task_index=task_index, block_index=block_index, file_format=file_format)

@DeveloperAPI
class DefaultBlockWritePathProvider(BlockWritePathProvider):
    """Default block write path provider implementation that writes each
    dataset block out to a file of the form:
    {base_path}/{dataset_uuid}_{task_index}_{block_index}.{file_format}
    """

    def _get_write_path_for_block(self, base_path: str, *, filesystem: Optional['pyarrow.fs.FileSystem']=None, dataset_uuid: Optional[str]=None, task_index: Optional[int]=None, block_index: Optional[int]=None, file_format: Optional[str]=None) -> str:
        if False:
            print('Hello World!')
        assert task_index is not None
        if block_index is not None:
            suffix = f'{dataset_uuid}_{task_index:06}_{block_index:06}.{file_format}'
        else:
            suffix = f'{dataset_uuid}_{task_index:06}.{file_format}'
        return posixpath.join(base_path, suffix)