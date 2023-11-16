""" Serve static files from multiple, dynamically defined locations.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import os
from pathlib import Path
from tornado.web import HTTPError, StaticFileHandler
from ...core.types import PathLike
__all__ = ('MultiRootStaticHandler',)

class MultiRootStaticHandler(StaticFileHandler):

    def initialize(self, root: dict[str, PathLike]) -> None:
        if False:
            return 10
        self.root = root
        self.default_filename = None

    @classmethod
    def get_absolute_path(cls, root: dict[str, PathLike], path: str) -> str:
        if False:
            return 10
        try:
            (name, artifact_path) = path.split(os.sep, 1)
        except ValueError:
            raise HTTPError(404)
        artifacts_dir = root.get(name, None)
        if artifacts_dir is not None:
            return super().get_absolute_path(str(artifacts_dir), artifact_path)
        else:
            raise HTTPError(404)

    def validate_absolute_path(self, root: dict[str, PathLike], absolute_path: str) -> str | None:
        if False:
            return 10
        for artifacts_dir in root.values():
            if Path(absolute_path).is_relative_to(artifacts_dir):
                return super().validate_absolute_path(str(artifacts_dir), absolute_path)
        return None