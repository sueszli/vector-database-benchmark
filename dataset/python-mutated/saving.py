"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from os.path import abspath, expanduser
from typing import Sequence
from jinja2 import Template
from ..core.templates import FILE
from ..core.types import PathLike
from ..models.ui import UIElement
from ..resources import Resources, ResourcesLike
from ..settings import settings
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
DEFAULT_TITLE = 'Bokeh Plot'
__all__ = ('save',)

def save(obj: UIElement | Sequence[UIElement], filename: PathLike | None=None, resources: ResourcesLike | None=None, title: str | None=None, template: Template | str | None=None, state: State | None=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    ' Save an HTML file with the data for the current document.\n\n    Will fall back to the default output state (or an explicitly provided\n    :class:`State` object) for ``filename``, ``resources``, or ``title`` if they\n    are not provided. If the filename is not given and not provided via output state,\n    it is derived from the script name (e.g. ``/foo/myplot.py`` will create\n    ``/foo/myplot.html``)\n\n    Args:\n        obj (UIElement object) : a Layout (Row/Column), Plot or Widget object to display\n\n        filename (PathLike, e.g. str, Path, optional) : filename to save document under (default: None)\n            If None, use the default state configuration.\n\n        resources (Resources or ResourcesMode, optional) : A Resources config to use (default: None)\n            If None, use the default state configuration, if there is one.\n            otherwise use ``resources.INLINE``.\n\n        title (str, optional) : a title for the HTML document (default: None)\n            If None, use the default state title value, if there is one.\n            Otherwise, use "Bokeh Plot"\n\n        template (Template, str, optional) : HTML document template (default: FILE)\n            A Jinja2 Template, see bokeh.core.templates.FILE for the required template\n            parameters\n\n        state (State, optional) :\n            A :class:`State` object. If None, then the current default\n            implicit state is used. (default: None).\n\n    Returns:\n        str: the filename where the HTML file is saved.\n\n    '
    if state is None:
        state = curstate()
    theme = state.document.theme
    (filename, resources, title) = _get_save_args(state, filename, resources, title)
    _save_helper(obj, filename, resources, title, template, theme)
    return abspath(expanduser(filename))

def _get_save_args(state: State, filename: PathLike | None, resources: ResourcesLike | None, title: str | None) -> tuple[PathLike, Resources, str]:
    if False:
        return 10
    '\n\n    '
    (filename, is_default_filename) = _get_save_filename(state, filename)
    resources = _get_save_resources(state, resources, is_default_filename)
    title = _get_save_title(state, title, is_default_filename)
    return (filename, resources, title)

def _get_save_filename(state: State, filename: PathLike | None) -> tuple[PathLike, bool]:
    if False:
        return 10
    if filename is not None:
        return (filename, False)
    if state.file and (not settings.ignore_filename()):
        return (state.file.filename, False)
    return (default_filename('html'), True)

def _get_save_resources(state: State, resources: ResourcesLike | None, suppress_warning: bool) -> Resources:
    if False:
        for i in range(10):
            print('nop')
    if resources is not None:
        if isinstance(resources, Resources):
            return resources
        else:
            return Resources(mode=resources)
    if state.file:
        return state.file.resources
    if not suppress_warning:
        warn('save() called but no resources were supplied and output_file(...) was never called, defaulting to resources.CDN')
    return Resources(mode=settings.resources())

def _get_save_title(state: State, title: str | None, suppress_warning: bool) -> str:
    if False:
        i = 10
        return i + 15
    if title is not None:
        return title
    if state.file:
        return state.file.title
    if not suppress_warning:
        warn("save() called but no title was supplied and output_file(...) was never called, using default title 'Bokeh Plot'")
    return DEFAULT_TITLE

def _save_helper(obj: UIElement | Sequence[UIElement], filename: PathLike, resources: Resources | None, title: str | None, template: Template | str | None, theme: Theme | None=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n\n    '
    from ..embed import file_html
    html = file_html(obj, resources=resources, title=title, template=template or FILE, theme=theme)
    with open(filename, mode='w', encoding='utf-8') as f:
        f.write(html)