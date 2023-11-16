from __future__ import annotations
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING
from optuna._experimental import experimental_class
from optuna.artifacts.exceptions import ArtifactNotFound
if TYPE_CHECKING:
    from typing import BinaryIO

@experimental_class('3.3.0')
class FileSystemArtifactStore:
    """An artifact store for file systems.

    Args:
        base_path:
            The base path to a directory to store artifacts.

    Example:
        .. code-block:: python

            import os

            import optuna
            from optuna.artifacts import FileSystemArtifactStore
            from optuna.artifacts import upload_artifact


            base_path = "./artifacts"
            os.makedirs(base_path, exist_ok=True)
            artifact_store = FileSystemArtifactStore(base_path=base_path)


            def objective(trial: optuna.Trial) -> float:
                ... = trial.suggest_float("x", -10, 10)
                file_path = generate_example(...)
                upload_artifact(trial, file_path, artifact_store)
                return ...
    """

    def __init__(self, base_path: str | Path) -> None:
        if False:
            while True:
                i = 10
        if isinstance(base_path, str):
            base_path = Path(base_path)
        self._base_path = base_path

    def open_reader(self, artifact_id: str) -> BinaryIO:
        if False:
            print('Hello World!')
        filepath = os.path.join(self._base_path, artifact_id)
        try:
            f = open(filepath, 'rb')
        except FileNotFoundError as e:
            raise ArtifactNotFound('not found') from e
        return f

    def write(self, artifact_id: str, content_body: BinaryIO) -> None:
        if False:
            return 10
        filepath = os.path.join(self._base_path, artifact_id)
        with open(filepath, 'wb') as f:
            shutil.copyfileobj(content_body, f)

    def remove(self, artifact_id: str) -> None:
        if False:
            print('Hello World!')
        filepath = os.path.join(self._base_path, artifact_id)
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            raise ArtifactNotFound('not found') from e
if TYPE_CHECKING:
    from ._protocol import ArtifactStore
    _: ArtifactStore = FileSystemArtifactStore('')