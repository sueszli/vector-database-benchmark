import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Generic, TypeVar, Union

def onbuild(build_dir: Union[str, Path], is_source: bool, template_fields: Dict[str, Any], params: Dict[str, Any]):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove the ``versioningit`` build-requirement from Streamlink's source distribution.\n    Also set the static version string in the :mod:`streamlink._version` module when building the sdist/bdist.\n\n    The version string already gets set by ``versioningit`` when building, so the sdist doesn't need to have\n    ``versioningit`` added as a build-requirement. Previously, the generated version string was only applied\n    to the :mod:`streamlink._version` module while ``versioningit`` was still set as a build-requirement.\n\n    This custom onbuild hook gets called via the ``tool.versioningit.onbuild`` config in ``pyproject.toml``,\n    since ``versioningit`` does only support modifying one file via its default onbuild hook configuration.\n    "
    base_dir: Path = Path(build_dir).resolve()
    pkg_dir: Path = base_dir / 'src' if is_source else base_dir
    version: str = template_fields['version']
    cmproxy: Proxy[str]
    if is_source:
        with update_file(base_dir / 'pyproject.toml') as cmproxy:
            cmproxy.set(re.sub('^(\\s*)(\\"versioningit\\b.+?\\",).*$', '\\1# \\2', cmproxy.get(), flags=re.MULTILINE, count=1))
    if is_source:
        with update_file(base_dir / 'setup.py') as cmproxy:
            cmproxy.set(re.sub('^(\\s*)# (version=\\"\\",).*$', f'\\1version="{version}",', cmproxy.get(), flags=re.MULTILINE, count=1))
    with update_file(pkg_dir / 'streamlink' / '_version.py') as cmproxy:
        cmproxy.set(f'__version__ = "{version}"\n')
TProxyItem = TypeVar('TProxyItem')

class Proxy(Generic[TProxyItem]):

    def __init__(self, data: TProxyItem):
        if False:
            for i in range(10):
                print('nop')
        self._data = data

    def get(self) -> TProxyItem:
        if False:
            for i in range(10):
                print('nop')
        return self._data

    def set(self, data: TProxyItem) -> None:
        if False:
            return 10
        self._data = data

@contextmanager
def update_file(file: Path) -> Generator[Proxy[str], None, None]:
    if False:
        i = 10
        return i + 15
    with file.open('r+', encoding='utf-8') as fh:
        proxy = Proxy(fh.read())
        yield proxy
        fh.seek(0)
        fh.write(proxy.get())
        fh.truncate()