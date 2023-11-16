""" Standard plug-in to make enum module work when compiled.

The enum module provides a free function __new__ in class dictionaries to
manual metaclass calls. These become then unbound methods instead of static
methods, due to CPython only checking for plain uncompiled functions.
"""
from nuitka.plugins.PluginBase import NuitkaPluginBase
from nuitka.PythonVersions import python_version

class NuitkaPluginEnumWorkarounds(NuitkaPluginBase):
    """This is to make enum module work when compiled with Nuitka."""
    plugin_name = 'enum-compat'
    plugin_desc = "Required for Python2 and 'enum' package."

    @classmethod
    def isRelevant(cls):
        if False:
            return 10
        return python_version < 768

    @staticmethod
    def isAlwaysEnabled():
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def createPostModuleLoadCode(module):
        if False:
            while True:
                i = 10
        full_name = module.getFullName()
        if full_name == 'enum':
            code = 'from __future__ import absolute_import\nimport enum\ntry:\n    enum.Enum.__new__ = staticmethod(enum.Enum.__new__.__func__)\n    enum.IntEnum.__new__ = staticmethod(enum.IntEnum.__new__.__func__)\nexcept AttributeError:\n    pass\n'
            return (code, 'Monkey patching "enum" for compiled \'__new__\' methods.')