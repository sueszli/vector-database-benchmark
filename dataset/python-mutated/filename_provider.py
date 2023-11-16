from typing import Any, Dict, Optional
from ray.data.block import Block
from ray.util.annotations import PublicAPI

@PublicAPI(stability='alpha')
class FilenameProvider:
    """Generates filenames when you write a :class:`~ray.data.Dataset`.

    Use this class to customize the filenames used when writing a Dataset.

    Some methods write each row to a separate file, while others write each block to a
    separate file. For example, :meth:`ray.data.Dataset.write_images` writes individual
    rows, and :func:`ray.data.Dataset.write_parquet` writes blocks of data. For more
    information about blocks, see :ref:`Data internals <datasets_scheduling>`.

    If you're writing each row to a separate file, implement
    :meth:`~FilenameProvider.get_filename_for_row`. Otherwise, implement
    :meth:`~FilenameProvider.get_filename_for_block`.

    Example:

        This snippet shows you how to encode labels in written files. For example, if
        `"cat"` is a label, you might write a file named `cat_000000_000000_000000.png`.

        .. testcode::

            import ray
            from ray.data.datasource import FilenameProvider

            class ImageFilenameProvider(FilenameProvider):

                def __init__(self, file_format: str):
                    self.file_format = file_format

                def get_filename_for_row(self, row, task_index, block_index, row_index):
                    return (
                        f"{row['label']}_{task_index:06}_{block_index:06}"
                        f"_{row_index:06}.{self.file_format}"
                    )

            ds = ray.data.read_parquet("s3://anonymous@ray-example-data/images.parquet")
            ds.write_images(
                "/tmp/results",
                column="image",
                filename_provider=ImageFilenameProvider("png")
            )
    """

    def get_filename_for_block(self, block: Block, task_index: int, block_index: int) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Generate a filename for a block of data.\n\n        .. note::\n            Filenames must be unique and deterministic for a given task and block index.\n\n            A block consists of multiple rows and corresponds to a single output file.\n            Each task might produce a different number of blocks.\n\n        Args:\n            block: The block that will be written to a file.\n            task_index: The index of the the write task.\n            block_index: The index of the block *within* the write task.\n        '
        raise NotImplementedError

    def get_filename_for_row(self, row: Dict[str, Any], task_index: int, block_index: int, row_index: int) -> str:
        if False:
            print('Hello World!')
        "Generate a filename for a row.\n\n        .. note::\n            Filenames must be unique and deterministic for a given task, block, and row\n            index.\n\n            A block consists of multiple rows, and each row corresponds to a single\n            output file. Each task might produce a different number of blocks, and each\n            block might contain a different number of rows.\n\n        .. tip::\n            If you require a contiguous row index into the global dataset, use\n            :meth:`~Dataset.iter_rows`. This method is single-threaded and isn't\n            recommended for large datasets.\n\n        Args:\n            row: The row that will be written to a file.\n            task_index: The index of the the write task.\n            block_index: The index of the block *within* the write task.\n            row_index: The index of the row *within* the block.\n        "
        raise NotImplementedError

class _DefaultFilenameProvider(FilenameProvider):

    def __init__(self, dataset_uuid: Optional[str]=None, file_format: Optional[str]=None):
        if False:
            while True:
                i = 10
        self._dataset_uuid = dataset_uuid
        self._file_format = file_format

    def get_filename_for_block(self, block: Block, task_index: int, block_index: int) -> str:
        if False:
            return 10
        file_id = f'{task_index:06}_{block_index:06}'
        return self._generate_filename(file_id)

    def get_filename_for_row(self, row: Dict[str, Any], task_index: int, block_index: int, row_index: int) -> str:
        if False:
            i = 10
            return i + 15
        file_id = f'{task_index:06}_{block_index:06}_{row_index:06}'
        return self._generate_filename(file_id)

    def _generate_filename(self, file_id: str) -> str:
        if False:
            while True:
                i = 10
        filename = ''
        if self._dataset_uuid is not None:
            filename += f'{self._dataset_uuid}_'
        filename += file_id
        if self._file_format is not None:
            filename += f'.{self._file_format}'
        return filename