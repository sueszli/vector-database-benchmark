from __future__ import annotations
import logging
from collections import OrderedDict

class PyEnvCfg:

    def __init__(self, content, path) -> None:
        if False:
            return 10
        self.content = content
        self.path = path

    @classmethod
    def from_folder(cls, folder):
        if False:
            while True:
                i = 10
        return cls.from_file(folder / 'pyvenv.cfg')

    @classmethod
    def from_file(cls, path):
        if False:
            i = 10
            return i + 15
        content = cls._read_values(path) if path.exists() else OrderedDict()
        return PyEnvCfg(content, path)

    @staticmethod
    def _read_values(path):
        if False:
            while True:
                i = 10
        content = OrderedDict()
        for line in path.read_text(encoding='utf-8').splitlines():
            equals_at = line.index('=')
            key = line[:equals_at].strip()
            value = line[equals_at + 1:].strip()
            content[key] = value
        return content

    def write(self):
        if False:
            print('Hello World!')
        logging.debug('write %s', self.path)
        text = ''
        for (key, value) in self.content.items():
            line = f'{key} = {value}'
            logging.debug('\t%s', line)
            text += line
            text += '\n'
        self.path.write_text(text, encoding='utf-8')

    def refresh(self):
        if False:
            return 10
        self.content = self._read_values(self.path)
        return self.content

    def __setitem__(self, key, value) -> None:
        if False:
            print('Hello World!')
        self.content[key] = value

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        return self.content[key]

    def __contains__(self, item) -> bool:
        if False:
            i = 10
            return i + 15
        return item in self.content

    def update(self, other):
        if False:
            i = 10
            return i + 15
        self.content.update(other)
        return self

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(path={self.path})'
__all__ = ['PyEnvCfg']