"""Helpers for tests that check configuration
"""
import contextlib
import functools
import os
import tempfile
import textwrap
from typing import Any, Dict, Iterator
import pip._internal.configuration
from pip._internal.utils.misc import ensure_dir
Kind = pip._internal.configuration.Kind
kinds = pip._internal.configuration.kinds

class ConfigurationMixin:

    def setup_method(self) -> None:
        if False:
            i = 10
            return i + 15
        self.configuration = pip._internal.configuration.Configuration(isolated=False)

    def patch_configuration(self, variant: Kind, di: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        old = self.configuration._load_config_files

        @functools.wraps(old)
        def overridden() -> None:
            if False:
                while True:
                    i = 10
            self.configuration._config[variant].update(di)
            self.configuration._parsers[variant].append((None, None))
            old()
        self.configuration._load_config_files = overridden

    @contextlib.contextmanager
    def tmpfile(self, contents: str) -> Iterator[str]:
        if False:
            print('Hello World!')
        (fd, path) = tempfile.mkstemp(prefix='pip_', suffix='_config.ini', text=True)
        os.close(fd)
        contents = textwrap.dedent(contents).lstrip()
        ensure_dir(os.path.dirname(path))
        with open(path, 'w') as f:
            f.write(contents)
        yield path
        os.remove(path)