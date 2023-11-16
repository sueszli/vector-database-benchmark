"""
Module for registering media files.
"""
from __future__ import annotations
import typing
from ...value_object.read.media_types import MediaType
if typing.TYPE_CHECKING:
    from argparse import Namespace

def get_existing_graphics(args: Namespace) -> set[str]:
    if False:
        while True:
            i = 10
    '\n    List the graphics files that exist in the graphics file paths.\n    '
    filenames = set()
    for filepath in args.srcdir[MediaType.GRAPHICS.value].iterdir():
        filenames.add(filepath.stem)
    return filenames