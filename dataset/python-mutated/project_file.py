from __future__ import annotations
import hashlib
import json
from typing import Mapping
from tomlkit import TOMLDocument, items
from pdm import termui
from pdm.project.toml_file import TOMLBase
from pdm.utils import deprecation_warning

def _remove_empty_tables(doc: dict) -> None:
    if False:
        while True:
            i = 10
    for (k, v) in list(doc.items()):
        if isinstance(v, dict):
            _remove_empty_tables(v)
            if not v:
                del doc[k]

class PyProject(TOMLBase):
    """The data object representing th pyproject.toml file"""

    def read(self) -> TOMLDocument:
        if False:
            print('Hello World!')
        from pdm.formats import flit, poetry
        data = super().read()
        if 'project' not in data and self._path.exists():
            for converter in (flit, poetry):
                if converter.check_fingerprint(None, self._path):
                    (metadata, settings) = converter.convert(None, self._path, None)
                    data['project'] = metadata
                    if settings:
                        data.setdefault('tool', {}).setdefault('pdm', {}).update(settings)
                    break
        return data

    def write(self, show_message: bool=True) -> None:
        if False:
            return 10
        'Write the TOMLDocument to the file.'
        _remove_empty_tables(self._data)
        super().write()
        if show_message:
            self.ui.echo('Changes are written to [success]pyproject.toml[/].', verbosity=termui.Verbosity.NORMAL)

    @property
    def is_valid(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self._data.get('project'))

    @property
    def metadata(self) -> items.Table:
        if False:
            return 10
        return self._data.setdefault('project', {})

    @property
    def settings(self) -> items.Table:
        if False:
            for i in range(10):
                print('nop')
        return self._data.setdefault('tool', {}).setdefault('pdm', {})

    @property
    def build_system(self) -> dict:
        if False:
            print('Hello World!')
        return self._data.get('build-system', {})

    @property
    def resolution_overrides(self) -> Mapping[str, str]:
        if False:
            for i in range(10):
                print('nop')
        'A compatible getter method for the resolution overrides\n        in the pyproject.toml file.\n        '
        settings = self.settings
        if 'overrides' in settings:
            deprecation_warning("The 'tool.pdm.overrides' table has been renamed to 'tool.pdm.resolution.overrides', please update the setting accordingly.")
            return settings['overrides']
        return settings.get('resolution', {}).get('overrides', {})

    def content_hash(self, algo: str='sha256') -> str:
        if False:
            print('Hello World!')
        'Generate a hash of the sensible content of the pyproject.toml file.\n        When the hash changes, it means the project needs to be relocked.\n        '
        dump_data = {'sources': self.settings.get('source', []), 'dependencies': self.metadata.get('dependencies', []), 'dev-dependencies': self.settings.get('dev-dependencies', {}), 'optional-dependencies': self.metadata.get('optional-dependencies', {}), 'requires-python': self.metadata.get('requires-python', ''), 'overrides': self.resolution_overrides}
        pyproject_content = json.dumps(dump_data, sort_keys=True)
        hasher = hashlib.new(algo)
        hasher.update(pyproject_content.encode('utf-8'))
        return hasher.hexdigest()

    @property
    def plugins(self) -> list[str]:
        if False:
            return 10
        return self.settings.get('plugins', [])