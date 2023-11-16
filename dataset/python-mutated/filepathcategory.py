"""Completion category for filesystem paths.

NOTE: This module deliberately uses os.path rather than pathlib, because of how
it interacts with the completion, which operates on strings. For example, we
need to be able to tell the difference between "~/input" and "~/input/". Also,
if we get "~/input", we want to glob "~/input*" rather than "~/input/*" which
is harder to achieve via pathlib.
"""
import glob
import os
import os.path
from typing import List, Optional, Iterable
from qutebrowser.qt.core import QAbstractListModel, QModelIndex, QObject, Qt, QUrl
from qutebrowser.config import config
from qutebrowser.utils import log

class FilePathCategory(QAbstractListModel):
    """Represent filesystem paths matching a pattern."""

    def __init__(self, name: str, parent: QObject=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._paths: List[str] = []
        self.name = name
        self.columns_to_filter = [0]

    def _contract_user(self, val: str, path: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        "Contract ~/... and ~user/... in results.\n\n        Arguments:\n            val: The user's partially typed path.\n            path: The found path based on the input.\n        "
        if not val.startswith('~'):
            return path
        head = val.split(os.sep)[0]
        return path.replace(os.path.expanduser(head), head, 1)

    def _glob(self, val: str) -> Iterable[str]:
        if False:
            i = 10
            return i + 15
        'Find paths based on the given pattern.'
        if not os.path.isabs(val):
            return []
        try:
            return glob.glob(glob.escape(val) + '*')
        except ValueError as e:
            log.completion.debug(f'Failed to glob: {e}')
            return []

    def _url_to_path(self, val: str) -> str:
        if False:
            while True:
                i = 10
        'Get a path from a file:/// URL.'
        url = QUrl(val)
        assert url.isValid(), url
        assert url.scheme() == 'file', url
        return url.toLocalFile()

    def set_pattern(self, val: str) -> None:
        if False:
            while True:
                i = 10
        "Compute list of suggested paths (called from `CompletionModel`).\n\n        Args:\n            val: The user's partially typed URL/path.\n        "
        if not val:
            self._paths = config.val.completion.favorite_paths or []
        elif val.startswith('file:///'):
            url_path = self._url_to_path(val)
            self._paths = sorted((QUrl.fromLocalFile(path).toString() for path in self._glob(url_path)))
        else:
            try:
                expanded = os.path.expanduser(val)
            except ValueError:
                expanded = val
            paths = self._glob(expanded)
            self._paths = sorted((self._contract_user(val, path) for path in paths))

    def data(self, index: QModelIndex, role: int=Qt.ItemDataRole.DisplayRole) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Implement abstract method in QAbstractListModel.'
        if role == Qt.ItemDataRole.DisplayRole and index.column() == 0:
            return self._paths[index.row()]
        return None

    def rowCount(self, parent: QModelIndex=QModelIndex()) -> int:
        if False:
            i = 10
            return i + 15
        'Implement abstract method in QAbstractListModel.'
        if parent.isValid():
            return 0
        return len(self._paths)