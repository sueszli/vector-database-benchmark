from typing import Any, Iterable, List
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block
from ray.util.annotations import DeveloperAPI

@DeveloperAPI
class Datasink:
    """Interface for defining write-related logic.

    If you want to write data to something that isn't built-in, subclass this class
    and call :meth:`~ray.data.Dataset.write_datasink`.
    """

    def on_write_start(self) -> None:
        if False:
            while True:
                i = 10
        'Callback for when a write job starts.\n\n        Use this method to perform setup for write tasks. For example, creating a\n        staging bucket in S3.\n        '
        pass

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> Any:
        if False:
            i = 10
            return i + 15
        'Write blocks. This is used by a single write task.\n\n        Args:\n            blocks: Generator of data blocks.\n            ctx: ``TaskContext`` for the write task.\n\n        Returns:\n            A user-defined output. Can be anything, and the returned value is passed to\n            :meth:`~Datasink.on_write_complete`.\n        '
        raise NotImplementedError

    def on_write_complete(self, write_results: List[Any]) -> None:
        if False:
            i = 10
            return i + 15
        'Callback for when a write job completes.\n\n        This can be used to "commit" a write output. This method must\n        succeed prior to ``write_datasink()`` returning to the user. If this\n        method fails, then ``on_write_failed()`` is called.\n\n        Args:\n            write_results: The objects returned by every :meth:`~Datasink.write` task.\n        '
        pass

    def on_write_failed(self, error: Exception) -> None:
        if False:
            i = 10
            return i + 15
        'Callback for when a write job fails.\n\n        This is called on a best-effort basis on write failures.\n\n        Args:\n            error: The first error encountered.\n        '
        pass

    def get_name(self) -> str:
        if False:
            print('Hello World!')
        'Return a human-readable name for this datasink.\n\n        This is used as the names of the write tasks.\n        '
        name = type(self).__name__
        datasink_suffix = 'Datasink'
        if name.startswith('_'):
            name = name[1:]
        if name.endswith(datasink_suffix):
            name = name[:-len(datasink_suffix)]
        return name

    @property
    def supports_distributed_writes(self) -> bool:
        if False:
            return 10
        "If ``False``, only launch write tasks on the driver's node."
        return True