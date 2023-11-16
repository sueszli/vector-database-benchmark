""" Module names are common string type, which deserves special operations.

These are used in Nuitka for module and package names in most places, and
allow to easily make checks on them.

"""
import fnmatch
import os

def checkModuleName(value):
    if False:
        while True:
            i = 10
    return '..' not in str(value) and (not (str(value).endswith('.') or str(value) == '.'))
post_module_load_trigger_name = '-postLoad'
pre_module_load_trigger_name = '-preLoad'
trigger_names = (pre_module_load_trigger_name, post_module_load_trigger_name)

def makeTriggerModuleName(module_name, trigger_name):
    if False:
        print('Hello World!')
    assert trigger_name in trigger_names
    return ModuleName(module_name + trigger_name)
_multi_dist_prefix = 'multidist-'

def makeMultidistModuleName(count, suffix):
    if False:
        i = 10
        return i + 15
    return ModuleName('%s%d-%s' % (_multi_dist_prefix, count, suffix))

class ModuleName(str):

    def __init__(self, value):
        if False:
            return 10
        assert checkModuleName(value), value
        str.__init__(value)

    @staticmethod
    def makeModuleNameInPackage(module_name, package_name):
        if False:
            print('Hello World!')
        'Create a module name in a package.\n\n        Args:\n            - module_name (str or ModuleName) module name to put below the package\n            - package_name (str or ModuleName or None) package to put below\n\n        Returns:\n            Module name "package_name.module_name" or if "package_name" is None\n            then simply "module_name".\n\n        Notes:\n            Prefer this factory function over manually duplicating the pattern\n            behind it.\n\n        '
        if package_name is not None:
            return ModuleName(package_name + '.' + module_name)
        else:
            return ModuleName(module_name)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return "<ModuleName '%s'>" % str(self)

    def asString(self):
        if False:
            i = 10
            return i + 15
        'Get a simply str value.\n\n        Notes:\n            This should only be used to create constant values for code\n            generation, there is no other reason to lower the type of\n            these values otherwise.\n        '
        return str(self)

    def asPath(self):
        if False:
            return 10
        return str(self).replace('.', os.path.sep)

    def getPackageName(self):
        if False:
            while True:
                i = 10
        'Get the package name if any.\n\n        Returns:\n            ModuleName of the containing package or None if already\n            top level.\n        '
        return self.splitModuleBasename()[0]

    def getParentPackageNames(self):
        if False:
            i = 10
            return i + 15
        'Yield parent packages in descending order.'
        parent_packages = []
        parent_package = self.getPackageName()
        while parent_package is not None:
            parent_packages.append(parent_package)
            parent_package = parent_package.getPackageName()
        for parent_package in reversed(parent_packages):
            yield parent_package

    def getRelativePackageName(self, level):
        if False:
            for i in range(10):
                print('nop')
        result = '.'.join(self.asString().split('.')[:-level + 1])
        if result == '':
            return None
        else:
            return ModuleName(result)

    def getTopLevelPackageName(self):
        if False:
            print('Hello World!')
        'Get the top level package name.\n\n        Returns:\n            ModuleName of the top level name.\n        '
        package_name = self.getPackageName()
        if package_name is None:
            return self
        else:
            return package_name.getTopLevelPackageName()

    def getBasename(self):
        if False:
            while True:
                i = 10
        'Get leaf name of the module without package part.\n\n        Returns:\n            ModuleName without package.\n        '
        return self.splitModuleBasename()[1]

    def splitModuleBasename(self):
        if False:
            for i in range(10):
                print('nop')
        'Split a module into package name and module name.'
        if '.' in self:
            package_part = ModuleName(self[:self.rfind('.')])
            module_name = ModuleName(self[self.rfind('.') + 1:])
        else:
            package_part = None
            module_name = self
        return (package_part, module_name)

    def splitPackageName(self):
        if False:
            return 10
        'Split a module into the top level package name and remaining module name.'
        if '.' in self:
            package_part = ModuleName(self[:self.find('.')])
            module_name = ModuleName(self[self.find('.') + 1:])
        else:
            package_part = None
            module_name = self
        return (package_part, module_name)

    def hasNamespace(self, package_name):
        if False:
            for i in range(10):
                print('nop')
        return self == package_name or self.isBelowNamespace(package_name)

    def hasOneOfNamespaces(self, *package_names):
        if False:
            i = 10
            return i + 15
        'Check if a module name is below one of many namespaces.\n\n        Args:\n            - package_names: Star argument that allows also lists and tuples\n\n        Returns:\n            bool - module name is below one of the packages.\n        '
        for package_name in package_names:
            if type(package_name) in (tuple, list, set):
                if self.hasOneOfNamespaces(*package_name):
                    return True
            elif self.hasNamespace(package_name):
                return True
        return False

    def isBelowNamespace(self, package_name):
        if False:
            while True:
                i = 10
        assert type(package_name) in (str, ModuleName), package_name
        return str(self).startswith(package_name + '.')

    def getChildNamed(self, *args):
        if False:
            return 10
        'Get a child package with these names added.'
        return ModuleName('.'.join([self] + list(args)))

    def getSiblingNamed(self, *args):
        if False:
            i = 10
            return i + 15
        'Get a sub-package relative to this child package.'
        return self.getPackageName().getChildNamed(*args)

    def relocateModuleNamespace(self, parent_old, parent_new):
        if False:
            i = 10
            return i + 15
        'Get a module name, where the top level part is translated from old to new.'
        assert self.hasNamespace(parent_old)
        submodule_name_str = str(self)[len(str(parent_old)) + 1:]
        if submodule_name_str:
            return ModuleName(parent_new).getChildNamed(submodule_name_str)
        else:
            return ModuleName(parent_new)

    def matchesToShellPattern(self, pattern):
        if False:
            i = 10
            return i + 15
        'Match a module name to a list of patterns\n\n        Args:\n            pattern:\n                Complies with fnmatch.fnmatch description\n                or also is below the package. So "*.tests" will matches to also\n                "something.tests.MyTest", thereby allowing to match whole\n                packages with one pattern only.\n        Returns:\n            Tuple of two values, where the first value is the result, second value\n            explains why the pattern matched and how.\n        '
        if self == pattern:
            return (True, "is exact match of '%s'" % pattern)
        elif self.isBelowNamespace(pattern):
            return (True, "is package content of '%s'" % pattern)
        elif fnmatch.fnmatch(self.asString(), pattern):
            return (True, "matches pattern '%s'" % pattern)
        elif fnmatch.fnmatch(self.asString(), pattern + '.*'):
            return (True, "is package content of match to pattern '%s'" % pattern)
        else:
            return (False, None)

    def matchesToShellPatterns(self, patterns):
        if False:
            for i in range(10):
                print('nop')
        'Match a module name to a list of patterns\n\n        Args:\n            patterns:\n                List of patterns that comply with fnmatch.fnmatch description\n                or also is below the package. So "*.tests" will matches to also\n                "something.tests.MyTest", thereby allowing to match whole\n                packages with one pattern only.\n        Returns:\n            Tuple of two values, where the first value is the result, second value\n            explains which pattern matched and how.\n        '
        for pattern in patterns:
            (match, reason) = self.matchesToShellPattern(pattern)
            if match:
                return (match, reason)
        return (False, None)

    def isFakeModuleName(self):
        if False:
            while True:
                i = 10
        return str(self).endswith(trigger_names)

    def isMultidistModuleName(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self).startswith(_multi_dist_prefix)
    for _func_name in ('split', 'startswith', 'endswith', '__mod__'):
        code = "def %(func_name)s(*args, **kwargs):\n    from nuitka.Errors import NuitkaCodeDeficit\n    raise NuitkaCodeDeficit('''\nDo not use %(func_name)s on ModuleName objects, use e.g.\n.hasNamespace(),\n.getBasename(),\n.getTopLevelPackageName()\n.hasOneOfNamespaces()\n\nCheck API documentation of nuitka.utils.ModuleNames.ModuleName for more\nvariations.\n''')\n" % {'func_name': _func_name}
        exec(code)