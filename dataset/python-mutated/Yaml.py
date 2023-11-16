""" Nuitka yaml utility functions.

Because we want to work with Python2.6 or higher, we play a few tricks with
what library to use for what Python. We have an 2 inline copy of PyYAML, one
that still does 2.6 and one for newer Pythons.

Also we put loading for specific packages in here and a few helpers to work
with these config files.
"""
from __future__ import absolute_import
import os
import pkgutil
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.Options import getUserProvidedYamlFiles
from nuitka.Tracing import general
from .FileOperations import getFileContents
from .Importing import importFromInlineCopy

class PackageConfigYaml(object):
    __slots__ = ('name', 'data')

    def __init__(self, name, data):
        if False:
            return 10
        self.name = name
        assert type(data) is list
        self.data = OrderedDict()
        for item in data:
            module_name = item.pop('module-name')
            if '/' in module_name:
                general.sysexit("Error, invalid module name in '%s' looks like a file path '%s'." % (self.name, module_name))
            if module_name in self.data:
                general.sysexit("Duplicate module-name '%s' encountered." % module_name)
            self.data[module_name] = item

    def __repr__(self):
        if False:
            return 10
        return '<PackageConfigYaml %s>' % self.name

    def get(self, name, section):
        if False:
            i = 10
            return i + 15
        'Return a configs for that section.'
        result = self.data.get(name)
        if result is not None:
            result = result.get(section, ())
        else:
            result = ()
        if type(result) in (dict, OrderedDict):
            result = (result,)
        return result

    def keys(self):
        if False:
            i = 10
            return i + 15
        return self.data.keys()

    def items(self):
        if False:
            i = 10
            return i + 15
        return self.data.items()

    def update(self, other):
        if False:
            for i in range(10):
                print('nop')
        for (key, value) in other.items():
            assert key not in self.data, key
            self.data[key] = value

def getYamlPackage():
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(getYamlPackage, 'yaml'):
        try:
            import yaml
            getYamlPackage.yaml = yaml
        except ImportError:
            getYamlPackage.yaml = importFromInlineCopy('yaml', must_exist=True, delete_module=True)
    return getYamlPackage.yaml

def parseYaml(data):
    if False:
        while True:
            i = 10
    yaml = getYamlPackage()

    class OrderedLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node):
        if False:
            while True:
                i = 10
        loader.flatten_mapping(node)
        return OrderedDict(loader.construct_pairs(node))
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(data, OrderedLoader)
_yaml_cache = {}

def parsePackageYaml(package_name, filename):
    if False:
        i = 10
        return i + 15
    key = (package_name, filename)
    if key not in _yaml_cache:
        data = pkgutil.get_data(package_name, filename)
        if data is None:
            raise IOError('Cannot find %s.%s' % (package_name, filename))
        _yaml_cache[key] = PackageConfigYaml(name=filename, data=parseYaml(data))
    return _yaml_cache[key]
_package_config = None

def getYamlPackageConfiguration():
    if False:
        return 10
    'Get Nuitka package configuration. Merged from multiple sources.'
    global _package_config
    if _package_config is None:
        _package_config = parsePackageYaml('nuitka.plugins.standard', 'standard.nuitka-package.config.yml')
        _package_config.update(parsePackageYaml('nuitka.plugins.standard', 'stdlib2.nuitka-package.config.yml'))
        _package_config.update(parsePackageYaml('nuitka.plugins.standard', 'stdlib3.nuitka-package.config.yml'))
        try:
            _package_config.update(parsePackageYaml('nuitka.plugins.commercial', 'commercial.nuitka-package.config.yml'))
        except IOError:
            pass
        for user_yaml_filename in getUserProvidedYamlFiles():
            _package_config.update(PackageConfigYaml(name=user_yaml_filename, data=parseYaml(getFileContents(user_yaml_filename, mode='rb'))))
    return _package_config

def getYamlPackageConfigurationSchemaFilename():
    if False:
        while True:
            i = 10
    'Get the filename of the schema for Nuitka package configuration.'
    return os.path.join(os.path.dirname(__file__), '..', '..', 'misc', 'nuitka-package-config-schema.json')