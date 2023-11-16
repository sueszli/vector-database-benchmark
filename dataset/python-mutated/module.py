from __future__ import annotations
import filecmp
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, NewType, overload
from urllib.parse import urlparse
from reactpy._warnings import warn
from reactpy.config import REACTPY_DEBUG_MODE, REACTPY_WEB_MODULES_DIR
from reactpy.core.types import ImportSourceDict, VdomDictConstructor
from reactpy.core.vdom import make_vdom_constructor
from reactpy.web.utils import module_name_suffix, resolve_module_exports_from_file, resolve_module_exports_from_url
logger = logging.getLogger(__name__)
SourceType = NewType('SourceType', str)
NAME_SOURCE = SourceType('NAME')
'A named source - usually a Javascript package name'
URL_SOURCE = SourceType('URL')
'A source loaded from a URL, usually a CDN'

def module_from_url(url: str, fallback: Any | None=None, resolve_exports: bool | None=None, resolve_exports_depth: int=5, unmount_before_update: bool=False) -> WebModule:
    if False:
        while True:
            i = 10
    'Load a :class:`WebModule` from a :data:`URL_SOURCE`\n\n    Parameters:\n        url:\n            Where the javascript module will be loaded from which conforms to the\n            interface for :ref:`Custom Javascript Components`\n        fallback:\n            What to temporarily display while the module is being loaded.\n        resolve_imports:\n            Whether to try and find all the named exports of this module.\n        resolve_exports_depth:\n            How deeply to search for those exports.\n        unmount_before_update:\n            Cause the component to be unmounted before each update. This option should\n            only be used if the imported package fails to re-render when props change.\n            Using this option has negative performance consequences since all DOM\n            elements must be changed on each render. See :issue:`461` for more info.\n    '
    return WebModule(source=url, source_type=URL_SOURCE, default_fallback=fallback, file=None, export_names=resolve_module_exports_from_url(url, resolve_exports_depth) if (resolve_exports if resolve_exports is not None else REACTPY_DEBUG_MODE.current) else None, unmount_before_update=unmount_before_update)
_FROM_TEMPLATE_DIR = '__from_template__'

def module_from_template(template: str, package: str, cdn: str='https://esm.sh', fallback: Any | None=None, resolve_exports: bool | None=None, resolve_exports_depth: int=5, unmount_before_update: bool=False) -> WebModule:
    if False:
        print('Hello World!')
    "Create a :class:`WebModule` from a framework template\n\n    This is useful for experimenting with component libraries that do not already\n    support ReactPy's :ref:`Custom Javascript Component` interface.\n\n    .. warning::\n\n        This approach is not recommended for use in a production setting because the\n        framework templates may use unpinned dependencies that could change without\n        warning. It's best to author a module adhering to the\n        :ref:`Custom Javascript Component` interface instead.\n\n    **Templates**\n\n    - ``react``: for modules exporting React components\n\n    Parameters:\n        template:\n            The name of the framework template to use with the given ``package``.\n        package:\n            The name of a package to load. May include a file extension (defaults to\n            ``.js`` if not given)\n        cdn:\n            Where the package should be loaded from. The CDN must distribute ESM modules\n        fallback:\n            What to temporarily display while the module is being loaded.\n        resolve_imports:\n            Whether to try and find all the named exports of this module.\n        resolve_exports_depth:\n            How deeply to search for those exports.\n        unmount_before_update:\n            Cause the component to be unmounted before each update. This option should\n            only be used if the imported package fails to re-render when props change.\n            Using this option has negative performance consequences since all DOM\n            elements must be changed on each render. See :issue:`461` for more info.\n    "
    warn('module_from_template() is deprecated due to instability - use the Javascript Components API instead. This function will be removed in a future release.', DeprecationWarning)
    (template_name, _, template_version) = template.partition('@')
    template_version = '@' + template_version if template_version else ''
    package_name = urlparse(package).path
    cdn = cdn.rstrip('/')
    template_file_name = template_name + module_name_suffix(package_name)
    template_file = Path(__file__).parent / 'templates' / template_file_name
    if not template_file.exists():
        msg = f'No template for {template_file_name!r} exists'
        raise ValueError(msg)
    variables = {'PACKAGE': package, 'CDN': cdn, 'VERSION': template_version}
    content = Template(template_file.read_text()).substitute(variables)
    return module_from_string(_FROM_TEMPLATE_DIR + '/' + package_name, content, fallback, resolve_exports, resolve_exports_depth, unmount_before_update=unmount_before_update)

def module_from_file(name: str, file: str | Path, fallback: Any | None=None, resolve_exports: bool | None=None, resolve_exports_depth: int=5, unmount_before_update: bool=False, symlink: bool=False) -> WebModule:
    if False:
        print('Hello World!')
    'Load a :class:`WebModule` from a given ``file``\n\n    Parameters:\n        name:\n            The name of the package\n        file:\n            The file from which the content of the web module will be created.\n        fallback:\n            What to temporarily display while the module is being loaded.\n        resolve_imports:\n            Whether to try and find all the named exports of this module.\n        resolve_exports_depth:\n            How deeply to search for those exports.\n        unmount_before_update:\n            Cause the component to be unmounted before each update. This option should\n            only be used if the imported package fails to re-render when props change.\n            Using this option has negative performance consequences since all DOM\n            elements must be changed on each render. See :issue:`461` for more info.\n        symlink:\n            Whether the web module should be saved as a symlink to the given ``file``.\n    '
    name += module_name_suffix(name)
    source_file = Path(file).resolve()
    target_file = _web_module_path(name)
    if not source_file.exists():
        msg = f'Source file does not exist: {source_file}'
        raise FileNotFoundError(msg)
    if not target_file.exists():
        _copy_file(target_file, source_file, symlink)
    elif not _equal_files(source_file, target_file):
        logger.info(f'Existing web module {name!r} will be replaced with {target_file.resolve()}')
        target_file.unlink()
        _copy_file(target_file, source_file, symlink)
    return WebModule(source=name, source_type=NAME_SOURCE, default_fallback=fallback, file=target_file, export_names=resolve_module_exports_from_file(source_file, resolve_exports_depth) if (resolve_exports if resolve_exports is not None else REACTPY_DEBUG_MODE.current) else None, unmount_before_update=unmount_before_update)

def _equal_files(f1: Path, f2: Path) -> bool:
    if False:
        while True:
            i = 10
    f1 = f1.resolve()
    f2 = f2.resolve()
    return (f1.is_symlink() or f2.is_symlink()) and f1.resolve() == f2.resolve() or filecmp.cmp(str(f1), str(f2), shallow=False)

def _copy_file(target: Path, source: Path, symlink: bool) -> None:
    if False:
        print('Hello World!')
    target.parent.mkdir(parents=True, exist_ok=True)
    if symlink:
        target.symlink_to(source)
    else:
        shutil.copy(source, target)

def module_from_string(name: str, content: str, fallback: Any | None=None, resolve_exports: bool | None=None, resolve_exports_depth: int=5, unmount_before_update: bool=False) -> WebModule:
    if False:
        while True:
            i = 10
    'Load a :class:`WebModule` whose ``content`` comes from a string.\n\n    Parameters:\n        name:\n            The name of the package\n        content:\n            The contents of the web module\n        fallback:\n            What to temporarily display while the module is being loaded.\n        resolve_imports:\n            Whether to try and find all the named exports of this module.\n        resolve_exports_depth:\n            How deeply to search for those exports.\n        unmount_before_update:\n            Cause the component to be unmounted before each update. This option should\n            only be used if the imported package fails to re-render when props change.\n            Using this option has negative performance consequences since all DOM\n            elements must be changed on each render. See :issue:`461` for more info.\n    '
    name += module_name_suffix(name)
    target_file = _web_module_path(name)
    if target_file.exists() and target_file.read_text() != content:
        logger.info(f'Existing web module {name!r} will be replaced with {target_file.resolve()}')
        target_file.unlink()
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(content)
    return WebModule(source=name, source_type=NAME_SOURCE, default_fallback=fallback, file=target_file, export_names=resolve_module_exports_from_file(target_file, resolve_exports_depth) if (resolve_exports if resolve_exports is not None else REACTPY_DEBUG_MODE.current) else None, unmount_before_update=unmount_before_update)

@dataclass(frozen=True)
class WebModule:
    source: str
    source_type: SourceType
    default_fallback: Any | None
    export_names: set[str] | None
    file: Path | None
    unmount_before_update: bool

@overload
def export(web_module: WebModule, export_names: str, fallback: Any | None=..., allow_children: bool=...) -> VdomDictConstructor:
    if False:
        print('Hello World!')
    ...

@overload
def export(web_module: WebModule, export_names: list[str] | tuple[str, ...], fallback: Any | None=..., allow_children: bool=...) -> list[VdomDictConstructor]:
    if False:
        i = 10
        return i + 15
    ...

def export(web_module: WebModule, export_names: str | list[str] | tuple[str, ...], fallback: Any | None=None, allow_children: bool=True) -> VdomDictConstructor | list[VdomDictConstructor]:
    if False:
        return 10
    'Return one or more VDOM constructors from a :class:`WebModule`\n\n    Parameters:\n        export_names:\n            One or more names to export. If given as a string, a single component\n            will be returned. If a list is given, then a list of components will be\n            returned.\n        fallback:\n            What to temporarily display while the module is being loaded.\n        allow_children:\n            Whether or not these components can have children.\n    '
    if isinstance(export_names, str):
        if web_module.export_names is not None and export_names not in web_module.export_names:
            msg = f'{web_module.source!r} does not export {export_names!r}'
            raise ValueError(msg)
        return _make_export(web_module, export_names, fallback, allow_children)
    else:
        if web_module.export_names is not None:
            missing = sorted(set(export_names).difference(web_module.export_names))
            if missing:
                msg = f'{web_module.source!r} does not export {missing!r}'
                raise ValueError(msg)
        return [_make_export(web_module, name, fallback, allow_children) for name in export_names]

def _make_export(web_module: WebModule, name: str, fallback: Any | None, allow_children: bool) -> VdomDictConstructor:
    if False:
        i = 10
        return i + 15
    return make_vdom_constructor(name, allow_children=allow_children, import_source=ImportSourceDict(source=web_module.source, sourceType=web_module.source_type, fallback=fallback or web_module.default_fallback, unmountBeforeUpdate=web_module.unmount_before_update))

def _web_module_path(name: str) -> Path:
    if False:
        i = 10
        return i + 15
    directory = REACTPY_WEB_MODULES_DIR.current
    path = directory.joinpath(*name.split('/'))
    return path.with_suffix(path.suffix)