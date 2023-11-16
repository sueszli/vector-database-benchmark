from __future__ import annotations
from typing import TYPE_CHECKING
try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol
if TYPE_CHECKING:
    from typing import BinaryIO

class ArtifactStore(Protocol):
    """A protocol defining the interface for an artifact backend.

    An artifact backend is responsible for managing the storage and retrieval
    of artifact data. The backend should provide methods for opening, writing
    and removing artifacts.
    """

    def open_reader(self, artifact_id: str) -> BinaryIO:
        if False:
            while True:
                i = 10
        'Open the artifact identified by the artifact_id.\n\n        This method should return a binary file-like object in read mode, similar to\n        ``open(..., mode="rb")``. If the artifact does not exist, an\n        :exc:`~optuna.artifacts.exceptions.ArtifactNotFound` exception\n        should be raised.\n\n        Args:\n            artifact_id: The identifier of the artifact to open.\n\n        Returns:\n            BinaryIO: A binary file-like object that can be read from.\n        '
        ...

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        if False:
            i = 10
            return i + 15
        'Save the content to the backend.\n\n        Args:\n            artifact_id: The identifier of the artifact to write to.\n            content_body: The content to write to the artifact.\n        '
        ...

    def remove(self, artifact_id: str) -> None:
        if False:
            i = 10
            return i + 15
        'Remove the artifact identified by the artifact_id.\n\n        This method should delete the artifact from the backend. If the artifact does not\n        exist, an :exc:`~optuna.artifacts.exceptions.ArtifactNotFound` exception\n        may be raised.\n\n        Args:\n            artifact_id: The identifier of the artifact to remove.\n        '
        ...