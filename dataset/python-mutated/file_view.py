from __future__ import annotations
import logging
import os
import shutil
from itertools import chain
from os.path import commonpath
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pathlib import Path

def create_session_view(package: Path, temp_path: Path) -> Path:
    if False:
        while True:
            i = 10
    'Allows using the file after you no longer holding a lock to it by moving it into a temp folder.'
    temp_path.mkdir(parents=True, exist_ok=True)
    exists = [i.name for i in temp_path.iterdir()]
    file_id = max(chain((0,), (int(i) for i in exists if str(i).isnumeric())))
    session_dir = temp_path / str(file_id + 1)
    session_dir.mkdir()
    session_package = session_dir / package.name
    links = False
    if hasattr(os, 'link'):
        try:
            os.link(package, session_package)
            links = True
        except (OSError, NotImplementedError):
            pass
    if not links:
        shutil.copyfile(package, session_package)
    operation = 'links' if links else 'copied'
    common = commonpath((session_package, package))
    (rel_session, rel_package) = (session_package.relative_to(common), package.relative_to(common))
    logging.debug('package %s %s to %s (%s)', rel_session, operation, rel_package, common)
    return session_package