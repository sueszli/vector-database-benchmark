from __future__ import annotations
import json
import os
import sys
import zipfile
ABS_HERE = os.path.abspath(os.path.dirname(__file__))
NEW_IMPORT_SYSTEM = sys.version_info[0] >= 3

class VersionPlatformSelect:

    def __init__(self) -> None:
        if False:
            return 10
        self.archive = ABS_HERE
        self._zip_file = zipfile.ZipFile(ABS_HERE, 'r')
        self.modules = self._load('modules.json')
        self.distributions = self._load('distributions.json')
        self.__cache = {}

    def _load(self, of_file):
        if False:
            while True:
                i = 10
        version = '.'.join((str(i) for i in sys.version_info[0:2]))
        per_version = json.loads(self.get_data(of_file).decode('utf-8'))
        all_platforms = per_version[version] if version in per_version else per_version['3.9']
        content = all_platforms.get('==any', {})
        not_us = f'!={sys.platform}'
        for (key, value) in all_platforms.items():
            if key.startswith('!=') and key != not_us:
                content.update(value)
        content.update(all_platforms.get(f'=={sys.platform}', {}))
        return content

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        self._zip_file.close()

    def find_mod(self, fullname):
        if False:
            while True:
                i = 10
        if fullname in self.modules:
            return self.modules[fullname]
        return None

    def get_filename(self, fullname):
        if False:
            print('Hello World!')
        zip_path = self.find_mod(fullname)
        return None if zip_path is None else os.path.join(ABS_HERE, zip_path)

    def get_data(self, filename):
        if False:
            return 10
        if filename.startswith(ABS_HERE):
            filename = filename[len(ABS_HERE) + 1:]
            filename = filename.lstrip(os.sep)
        if sys.platform == 'win32':
            filename = '/'.join(filename.split(os.sep))
        with self._zip_file.open(filename) as file_handler:
            return file_handler.read()

    def find_distributions(self, context):
        if False:
            for i in range(10):
                print('nop')
        dist_class = versioned_distribution_class()
        name = context.name
        if name in self.distributions:
            result = dist_class(file_loader=self.get_data, dist_path=self.distributions[name])
            yield result

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return f'{self.__class__.__name__}(path={ABS_HERE})'

    def _register_distutils_finder(self):
        if False:
            return 10
        if 'distlib' not in self.modules:
            return

        class DistlibFinder:

            def __init__(self, path, loader) -> None:
                if False:
                    return 10
                self.path = path
                self.loader = loader

            def find(self, name):
                if False:
                    while True:
                        i = 10

                class Resource:

                    def __init__(self, content) -> None:
                        if False:
                            return 10
                        self.bytes = content
                full_path = os.path.join(self.path, name)
                return Resource(self.loader.get_data(full_path))
        from distlib.resources import register_finder
        register_finder(self, lambda module: DistlibFinder(os.path.dirname(module.__file__), self))
_VER_DISTRIBUTION_CLASS = None

def versioned_distribution_class():
    if False:
        print('Hello World!')
    global _VER_DISTRIBUTION_CLASS
    if _VER_DISTRIBUTION_CLASS is None:
        if sys.version_info >= (3, 8):
            from importlib.metadata import Distribution
        else:
            from importlib_metadata import Distribution

        class VersionedDistribution(Distribution):

            def __init__(self, file_loader, dist_path) -> None:
                if False:
                    i = 10
                    return i + 15
                self.file_loader = file_loader
                self.dist_path = dist_path

            def read_text(self, filename):
                if False:
                    for i in range(10):
                        print('nop')
                return self.file_loader(self.locate_file(filename)).decode('utf-8')

            def locate_file(self, path):
                if False:
                    print('Hello World!')
                return os.path.join(self.dist_path, path)
        _VER_DISTRIBUTION_CLASS = VersionedDistribution
    return _VER_DISTRIBUTION_CLASS
if NEW_IMPORT_SYSTEM:
    from importlib.abc import SourceLoader
    from importlib.util import spec_from_file_location

    class VersionedFindLoad(VersionPlatformSelect, SourceLoader):

        def find_spec(self, fullname, path, target=None):
            if False:
                return 10
            zip_path = self.find_mod(fullname)
            if zip_path is not None:
                return spec_from_file_location(name=fullname, loader=self)
            return None

        def module_repr(self, module):
            if False:
                while True:
                    i = 10
            raise NotImplementedError
else:
    from imp import new_module

    class VersionedFindLoad(VersionPlatformSelect):

        def find_module(self, fullname, path=None):
            if False:
                i = 10
                return i + 15
            return self if self.find_mod(fullname) else None

        def load_module(self, fullname):
            if False:
                print('Hello World!')
            filename = self.get_filename(fullname)
            code = self.get_data(filename)
            mod = sys.modules.setdefault(fullname, new_module(fullname))
            mod.__file__ = filename
            mod.__loader__ = self
            is_package = filename.endswith('__init__.py')
            if is_package:
                mod.__path__ = [os.path.dirname(filename)]
                mod.__package__ = fullname
            else:
                mod.__package__ = fullname.rpartition('.')[0]
            exec(code, mod.__dict__)
            return mod

def run():
    if False:
        while True:
            i = 10
    with VersionedFindLoad() as finder:
        sys.meta_path.insert(0, finder)
        finder._register_distutils_finder()
        from virtualenv.__main__ import run as run_virtualenv
        run_virtualenv()
if __name__ == '__main__':
    run()