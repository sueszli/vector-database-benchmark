"""Common routines for processing Java. """
import os
import re
import glob
from pathlib import Path
from typing import List
java_parsing = True
default_java_version = '1.4'
scopeStateVersions = ('1.8',)
java_win32_dir_glob = 'C:/Program Files*/*/*jdk*/bin'
java_win32_version_dir_glob = 'C:/Program Files*/*/*jdk*%s*/bin'
java_macos_include_dir_glob = '/System/Library/Frameworks/JavaVM.framework/Headers/'
java_macos_version_include_dir_glob = '/System/Library/Frameworks/JavaVM.framework/Versions/%s*/Headers/'
java_linux_include_dirs_glob = ['/usr/lib/jvm/default-java/include', '/usr/lib/jvm/java-*/include', '/opt/oracle-jdk-bin-*/include', '/opt/openjdk-bin-*/include', '/usr/lib/openjdk-*/include']
java_linux_version_include_dirs_glob = ['/usr/lib/jvm/java-*-sun-%s*/include', '/usr/lib/jvm/java-%s*-openjdk*/include', '/usr/java/jdk%s*/include']
if java_parsing:
    _reToken = re.compile('(\\n|\\\\\\\\|//|\\\\[\\\'"]|[\\\'"{\\};.()]|' + '\\d*\\.\\d*|[A-Za-z_][\\w$.]*|<[A-Za-z_]\\w+>|' + '/\\*|\\*/|\\[\\]|->)')

    class OuterState:
        """The initial state for parsing a Java file for classes,
        interfaces, and anonymous inner classes."""

        def __init__(self, version=default_java_version):
            if False:
                while True:
                    i = 10
            if version not in ('1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7', '1.8', '5', '6', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0'):
                msg = 'Java version %s not supported' % version
                raise NotImplementedError(msg)
            self.version = version
            self.listClasses = []
            self.listOutputs = []
            self.stackBrackets = []
            self.brackets = 0
            self.nextAnon = 1
            self.localClasses = []
            self.stackAnonClassBrackets = []
            self.anonStacksStack = [[0]]
            self.package = None

        def trace(self):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def __getClassState(self):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return self.classState
            except AttributeError:
                ret = ClassState(self)
                self.classState = ret
                return ret

        def __getPackageState(self):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return self.packageState
            except AttributeError:
                ret = PackageState(self)
                self.packageState = ret
                return ret

        def __getAnonClassState(self):
            if False:
                i = 10
                return i + 15
            try:
                return self.anonState
            except AttributeError:
                self.outer_state = self
                ret = SkipState(1, AnonClassState(self))
                self.anonState = ret
                return ret

        def __getSkipState(self):
            if False:
                i = 10
                return i + 15
            try:
                return self.skipState
            except AttributeError:
                ret = SkipState(1, self)
                self.skipState = ret
                return ret

        def _getAnonStack(self):
            if False:
                i = 10
                return i + 15
            return self.anonStacksStack[-1]

        def openBracket(self):
            if False:
                i = 10
                return i + 15
            self.brackets = self.brackets + 1

        def closeBracket(self):
            if False:
                i = 10
                return i + 15
            self.brackets = self.brackets - 1
            if len(self.stackBrackets) and self.brackets == self.stackBrackets[-1]:
                self.listOutputs.append('$'.join(self.listClasses))
                self.localClasses.pop()
                self.listClasses.pop()
                self.anonStacksStack.pop()
                self.stackBrackets.pop()
            if len(self.stackAnonClassBrackets) and self.brackets == self.stackAnonClassBrackets[-1] and (self.version not in scopeStateVersions):
                self._getAnonStack().pop()
                self.stackAnonClassBrackets.pop()

        def parseToken(self, token):
            if False:
                print('Hello World!')
            if token[:2] == '//':
                return IgnoreState('\n', self)
            elif token == '/*':
                return IgnoreState('*/', self)
            elif token == '{':
                self.openBracket()
            elif token == '}':
                self.closeBracket()
            elif token in ['"', "'"]:
                return IgnoreState(token, self)
            elif token == 'new':
                if len(self.listClasses) > 0:
                    return self.__getAnonClassState()
                return self.__getSkipState()
            elif token in ['class', 'interface', 'enum']:
                if len(self.listClasses) == 0:
                    self.nextAnon = 1
                self.stackBrackets.append(self.brackets)
                return self.__getClassState()
            elif token == 'package':
                return self.__getPackageState()
            elif token == '.':
                return self.__getSkipState()
            return self

        def addAnonClass(self):
            if False:
                while True:
                    i = 10
            'Add an anonymous inner class'
            if self.version in ('1.1', '1.2', '1.3', '1.4'):
                clazz = self.listClasses[0]
                self.listOutputs.append('%s$%d' % (clazz, self.nextAnon))
            elif self.version in ('1.5', '1.6', '1.7', '1.8', '5', '6', '9.0', '10.0', '11.0', '12.0', '13.0', '14.0', '15.0', '16.0', '17.0'):
                self.stackAnonClassBrackets.append(self.brackets)
                className = []
                className.extend(self.listClasses)
                self._getAnonStack()[-1] = self._getAnonStack()[-1] + 1
                for anon in self._getAnonStack():
                    className.append(str(anon))
                self.listOutputs.append('$'.join(className))
            self.nextAnon = self.nextAnon + 1
            self._getAnonStack().append(0)

        def setPackage(self, package):
            if False:
                i = 10
                return i + 15
            self.package = package

    class ScopeState:
        """
        A state that parses code within a scope normally,
        within the confines of a scope.
        """

        def __init__(self, old_state):
            if False:
                print('Hello World!')
            self.outer_state = old_state.outer_state
            self.old_state = old_state
            self.brackets = 0

        def __getClassState(self):
            if False:
                return 10
            try:
                return self.classState
            except AttributeError:
                ret = ClassState(self)
                self.classState = ret
                return ret

        def __getAnonClassState(self):
            if False:
                i = 10
                return i + 15
            try:
                return self.anonState
            except AttributeError:
                ret = SkipState(1, AnonClassState(self))
                self.anonState = ret
                return ret

        def __getSkipState(self):
            if False:
                return 10
            try:
                return self.skipState
            except AttributeError:
                ret = SkipState(1, self)
                self.skipState = ret
                return ret

        def openBracket(self):
            if False:
                i = 10
                return i + 15
            self.brackets = self.brackets + 1

        def closeBracket(self):
            if False:
                i = 10
                return i + 15
            self.brackets = self.brackets - 1

        def parseToken(self, token):
            if False:
                while True:
                    i = 10
            if token[:2] == '//':
                return IgnoreState('\n', self)
            elif token == '/*':
                return IgnoreState('*/', self)
            elif token == '{':
                self.openBracket()
            elif token == '}':
                self.closeBracket()
                if self.brackets == 0:
                    self.outer_state._getAnonStack().pop()
                    return self.old_state
            elif token in ['"', "'"]:
                return IgnoreState(token, self)
            elif token == 'new':
                return self.__getAnonClassState()
            elif token == '.':
                return self.__getSkipState()
            return self

    class AnonClassState:
        """A state that looks for anonymous inner classes."""

        def __init__(self, old_state):
            if False:
                for i in range(10):
                    print('nop')
            self.outer_state = old_state.outer_state
            self.old_state = old_state
            self.brace_level = 0

        def parseToken(self, token):
            if False:
                return 10
            if token[:2] == '//':
                return IgnoreState('\n', self)
            elif token == '/*':
                return IgnoreState('*/', self)
            elif token == '\n':
                return self
            elif token[0] == '<' and token[-1] == '>':
                return self
            elif token == '(':
                self.brace_level = self.brace_level + 1
                return self
            if self.brace_level > 0:
                if token == 'new':
                    return SkipState(1, AnonClassState(self))
                elif token in ['"', "'"]:
                    return IgnoreState(token, self)
                elif token == ')':
                    self.brace_level = self.brace_level - 1
                return self
            if token == '{':
                self.outer_state.addAnonClass()
                if self.outer_state.version in scopeStateVersions:
                    return ScopeState(old_state=self.old_state).parseToken(token)
            return self.old_state.parseToken(token)

    class SkipState:
        """A state that will skip a specified number of tokens before
        reverting to the previous state."""

        def __init__(self, tokens_to_skip, old_state):
            if False:
                for i in range(10):
                    print('nop')
            self.tokens_to_skip = tokens_to_skip
            self.old_state = old_state

        def parseToken(self, token):
            if False:
                for i in range(10):
                    print('nop')
            self.tokens_to_skip = self.tokens_to_skip - 1
            if self.tokens_to_skip < 1:
                return self.old_state
            return self

    class ClassState:
        """A state we go into when we hit a class or interface keyword."""

        def __init__(self, outer_state):
            if False:
                i = 10
                return i + 15
            self.outer_state = outer_state

        def parseToken(self, token):
            if False:
                i = 10
                return i + 15
            if token == '\n':
                return self
            if self.outer_state.localClasses and self.outer_state.stackBrackets[-1] > self.outer_state.stackBrackets[-2] + 1:
                locals = self.outer_state.localClasses[-1]
                try:
                    idx = locals[token]
                    locals[token] = locals[token] + 1
                except KeyError:
                    locals[token] = 1
                token = str(locals[token]) + token
            self.outer_state.localClasses.append({})
            self.outer_state.listClasses.append(token)
            self.outer_state.anonStacksStack.append([0])
            return self.outer_state

    class IgnoreState:
        """A state that will ignore all tokens until it gets to a
        specified token."""

        def __init__(self, ignore_until, old_state):
            if False:
                return 10
            self.ignore_until = ignore_until
            self.old_state = old_state

        def parseToken(self, token):
            if False:
                while True:
                    i = 10
            if self.ignore_until == token:
                return self.old_state
            return self

    class PackageState:
        """The state we enter when we encounter the package keyword.
        We assume the next token will be the package name."""

        def __init__(self, outer_state):
            if False:
                for i in range(10):
                    print('nop')
            self.outer_state = outer_state

        def parseToken(self, token):
            if False:
                while True:
                    i = 10
            self.outer_state.setPackage(token)
            return self.outer_state

    def parse_java_file(fn, version=default_java_version):
        if False:
            while True:
                i = 10
        with open(fn, 'r', encoding='utf-8') as f:
            data = f.read()
        return parse_java(data, version)

    def parse_java(contents, version=default_java_version, trace=None):
        if False:
            print('Hello World!')
        'Parse a .java file and return a double of package directory,\n        plus a list of .class files that compiling that .java file will\n        produce'
        package = None
        initial = OuterState(version)
        currstate = initial
        for token in _reToken.findall(contents):
            currstate = currstate.parseToken(token)
            if trace:
                trace(token, currstate)
        if initial.package:
            package = initial.package.replace('.', os.sep)
        return (package, initial.listOutputs)
else:

    def parse_java_file(fn):
        if False:
            return 10
        ' "Parse" a .java file.\n\n        This actually just splits the file name, so the assumption here\n        is that the file name matches the public class name, and that\n        the path to the file is the same as the package name.\n        '
        return os.path.split(fn)

def get_java_install_dirs(platform, version=None) -> List[str]:
    if False:
        i = 10
        return i + 15
    ' Find possible java jdk installation directories.\n\n    Returns a list for use as `default_paths` when looking up actual\n    java binaries with :meth:`SCons.Tool.find_program_path`.\n    The paths are sorted by version, latest first.\n\n    Args:\n        platform: selector for search algorithm.\n        version: if not None, restrict the search to this version.\n\n    Returns:\n        list of default paths for jdk.\n    '
    if platform == 'win32':
        paths = []
        if version:
            paths = glob.glob(java_win32_version_dir_glob % version)
        else:
            paths = glob.glob(java_win32_dir_glob)

        def win32getvnum(java):
            if False:
                for i in range(10):
                    print('nop')
            " Generates a sort key for win32 jdk versions.\n\n            We'll have gotten a path like ...something/*jdk*/bin because\n            that is the pattern we glob for. To generate the sort key,\n            extracts the next-to-last component, then trims it further if\n            it had a complex name, like 'java-1.8.0-openjdk-1.8.0.312-1',\n            to try and put it on a common footing with the more common style,\n            which looks like 'jdk-11.0.2'. \n\n            This is certainly fragile, and if someone has a 9.0 it won't\n            sort right since this will still be alphabetic, BUT 9.0 was\n            not an LTS release and is 30 mos out of support as this note\n            is written so just assume it will be okay.\n            "
            d = Path(java).parts[-2]
            if not d.startswith('jdk'):
                d = 'jdk' + d.rsplit('jdk', 1)[-1]
            return d
        return sorted(paths, key=win32getvnum, reverse=True)
    return []

def get_java_include_paths(env, javac, version) -> List[str]:
    if False:
        return 10
    'Find java include paths for JNI building.\n\n    Cannot be called in isolation - `javac` refers to an already detected\n    compiler. Normally would would call :func:`get_java_install_dirs` first\n    and then do lookups on the paths it returns before calling us.\n\n    Args:\n        env: construction environment, used to extract platform.\n        javac: path to detected javac.\n        version: if not None, restrict the search to this version.\n\n    Returns:\n        list of include directory paths.\n    '
    if not javac:
        return []
    if env['PLATFORM'] == 'win32':
        javac_bin_dir = os.path.dirname(javac)
        java_inc_dir = os.path.normpath(os.path.join(javac_bin_dir, '..', 'include'))
        paths = [java_inc_dir, os.path.join(java_inc_dir, 'win32')]
    elif env['PLATFORM'] == 'darwin':
        if not version:
            paths = [java_macos_include_dir_glob]
        else:
            paths = sorted(glob.glob(java_macos_version_include_dir_glob % version))
    else:
        base_paths = []
        if not version:
            for p in java_linux_include_dirs_glob:
                base_paths.extend(glob.glob(p))
        else:
            for p in java_linux_version_include_dirs_glob:
                base_paths.extend(glob.glob(p % version))
        paths = []
        for p in base_paths:
            paths.extend([p, os.path.join(p, 'linux')])
    return paths