"""Functionality to create lazy evaluation objects.

This includes waiting to import a module until it is actually used.

Most commonly, the 'lazy_import' function is used to import other modules
in an on-demand fashion. Typically use looks like::

    from bzrlib.lazy_import import lazy_import
    lazy_import(globals(), '''
    from bzrlib import (
        errors,
        osutils,
        branch,
        )
    import bzrlib.branch
    ''')

Then 'errors, osutils, branch' and 'bzrlib' will exist as lazy-loaded
objects which will be replaced with a real object on first use.

In general, it is best to only load modules in this way. This is because
it isn't safe to pass these variables to other functions before they
have been replaced. This is especially true for constants, sometimes
true for classes or functions (when used as a factory, or you want
to inherit from them).
"""
from __future__ import absolute_import

class ScopeReplacer(object):
    """A lazy object that will replace itself in the appropriate scope.

    This object sits, ready to create the real object the first time it is
    needed.
    """
    __slots__ = ('_scope', '_factory', '_name', '_real_obj')
    _should_proxy = True

    def __init__(self, scope, factory, name):
        if False:
            for i in range(10):
                print('nop')
        'Create a temporary object in the specified scope.\n        Once used, a real object will be placed in the scope.\n\n        :param scope: The scope the object should appear in\n        :param factory: A callable that will create the real object.\n            It will be passed (self, scope, name)\n        :param name: The variable name in the given scope.\n        '
        object.__setattr__(self, '_scope', scope)
        object.__setattr__(self, '_factory', factory)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_real_obj', None)
        scope[name] = self

    def _resolve(self):
        if False:
            while True:
                i = 10
        'Return the real object for which this is a placeholder'
        name = object.__getattribute__(self, '_name')
        real_obj = object.__getattribute__(self, '_real_obj')
        if real_obj is None:
            factory = object.__getattribute__(self, '_factory')
            scope = object.__getattribute__(self, '_scope')
            obj = factory(self, scope, name)
            if obj is self:
                raise errors.IllegalUseOfScopeReplacer(name, msg="Object tried to replace itself, check it's not using its own scope.")
            real_obj = object.__getattribute__(self, '_real_obj')
            if real_obj is None:
                object.__setattr__(self, '_real_obj', obj)
                scope[name] = obj
                return obj
        if not ScopeReplacer._should_proxy:
            raise errors.IllegalUseOfScopeReplacer(name, msg='Object already replaced, did you assign it to another variable?')
        return real_obj

    def __getattribute__(self, attr):
        if False:
            i = 10
            return i + 15
        obj = object.__getattribute__(self, '_resolve')()
        return getattr(obj, attr)

    def __setattr__(self, attr, value):
        if False:
            for i in range(10):
                print('nop')
        obj = object.__getattribute__(self, '_resolve')()
        return setattr(obj, attr, value)

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        obj = object.__getattribute__(self, '_resolve')()
        return obj(*args, **kwargs)

def disallow_proxying():
    if False:
        for i in range(10):
            print('nop')
    'Disallow lazily imported modules to be used as proxies.\n\n    Calling this function might cause problems with concurrent imports\n    in multithreaded environments, but will help detecting wasteful\n    indirection, so it should be called when executing unit tests.\n\n    Only lazy imports that happen after this call are affected.\n    '
    ScopeReplacer._should_proxy = False

class ImportReplacer(ScopeReplacer):
    """This is designed to replace only a portion of an import list.

    It will replace itself with a module, and then make children
    entries also ImportReplacer objects.

    At present, this only supports 'import foo.bar.baz' syntax.
    """
    __slots__ = ('_import_replacer_children', '_member', '_module_path')

    def __init__(self, scope, name, module_path, member=None, children={}):
        if False:
            for i in range(10):
                print('nop')
        "Upon request import 'module_path' as the name 'module_name'.\n        When imported, prepare children to also be imported.\n\n        :param scope: The scope that objects should be imported into.\n            Typically this is globals()\n        :param name: The variable name. Often this is the same as the\n            module_path. 'bzrlib'\n        :param module_path: A list for the fully specified module path\n            ['bzrlib', 'foo', 'bar']\n        :param member: The member inside the module to import, often this is\n            None, indicating the module is being imported.\n        :param children: Children entries to be imported later.\n            This should be a map of children specifications.\n            ::\n            \n                {'foo':(['bzrlib', 'foo'], None,\n                    {'bar':(['bzrlib', 'foo', 'bar'], None {})})\n                }\n\n        Examples::\n\n            import foo => name='foo' module_path='foo',\n                          member=None, children={}\n            import foo.bar => name='foo' module_path='foo', member=None,\n                              children={'bar':(['foo', 'bar'], None, {}}\n            from foo import bar => name='bar' module_path='foo', member='bar'\n                                   children={}\n            from foo import bar, baz would get translated into 2 import\n            requests. On for 'name=bar' and one for 'name=baz'\n        "
        if member is not None and children:
            raise ValueError('Cannot supply both a member and children')
        object.__setattr__(self, '_import_replacer_children', children)
        object.__setattr__(self, '_member', member)
        object.__setattr__(self, '_module_path', module_path)
        cls = object.__getattribute__(self, '__class__')
        ScopeReplacer.__init__(self, scope=scope, name=name, factory=cls._import)

    def _import(self, scope, name):
        if False:
            for i in range(10):
                print('nop')
        children = object.__getattribute__(self, '_import_replacer_children')
        member = object.__getattribute__(self, '_member')
        module_path = object.__getattribute__(self, '_module_path')
        module_python_path = '.'.join(module_path)
        if member is not None:
            module = __import__(module_python_path, scope, scope, [member], level=0)
            return getattr(module, member)
        else:
            module = __import__(module_python_path, scope, scope, [], level=0)
            for path in module_path[1:]:
                module = getattr(module, path)
        for (child_name, (child_path, child_member, grandchildren)) in children.iteritems():
            cls = object.__getattribute__(self, '__class__')
            cls(module.__dict__, name=child_name, module_path=child_path, member=child_member, children=grandchildren)
        return module

class ImportProcessor(object):
    """Convert text that users input into lazy import requests"""
    __slots__ = ['imports', '_lazy_import_class']

    def __init__(self, lazy_import_class=None):
        if False:
            return 10
        self.imports = {}
        if lazy_import_class is None:
            self._lazy_import_class = ImportReplacer
        else:
            self._lazy_import_class = lazy_import_class

    def lazy_import(self, scope, text):
        if False:
            while True:
                i = 10
        'Convert the given text into a bunch of lazy import objects.\n\n        This takes a text string, which should be similar to normal python\n        import markup.\n        '
        self._build_map(text)
        self._convert_imports(scope)

    def _convert_imports(self, scope):
        if False:
            while True:
                i = 10
        for (name, info) in self.imports.iteritems():
            self._lazy_import_class(scope, name=name, module_path=info[0], member=info[1], children=info[2])

    def _build_map(self, text):
        if False:
            i = 10
            return i + 15
        'Take a string describing imports, and build up the internal map'
        for line in self._canonicalize_import_text(text):
            if line.startswith('import '):
                self._convert_import_str(line)
            elif line.startswith('from '):
                self._convert_from_str(line)
            else:
                raise errors.InvalidImportLine(line, "doesn't start with 'import ' or 'from '")

    def _convert_import_str(self, import_str):
        if False:
            print('Hello World!')
        "This converts a import string into an import map.\n\n        This only understands 'import foo, foo.bar, foo.bar.baz as bing'\n\n        :param import_str: The import string to process\n        "
        if not import_str.startswith('import '):
            raise ValueError('bad import string %r' % (import_str,))
        import_str = import_str[len('import '):]
        for path in import_str.split(','):
            path = path.strip()
            if not path:
                continue
            as_hunks = path.split(' as ')
            if len(as_hunks) == 2:
                name = as_hunks[1].strip()
                module_path = as_hunks[0].strip().split('.')
                if name in self.imports:
                    raise errors.ImportNameCollision(name)
                self.imports[name] = (module_path, None, {})
            else:
                module_path = path.split('.')
                name = module_path[0]
                if name not in self.imports:
                    module_def = ([name], None, {})
                    self.imports[name] = module_def
                else:
                    module_def = self.imports[name]
                cur_path = [name]
                cur = module_def[2]
                for child in module_path[1:]:
                    cur_path.append(child)
                    if child in cur:
                        cur = cur[child][2]
                    else:
                        next = (cur_path[:], None, {})
                        cur[child] = next
                        cur = next[2]

    def _convert_from_str(self, from_str):
        if False:
            i = 10
            return i + 15
        "This converts a 'from foo import bar' string into an import map.\n\n        :param from_str: The import string to process\n        "
        if not from_str.startswith('from '):
            raise ValueError('bad from/import %r' % from_str)
        from_str = from_str[len('from '):]
        (from_module, import_list) = from_str.split(' import ')
        from_module_path = from_module.split('.')
        for path in import_list.split(','):
            path = path.strip()
            if not path:
                continue
            as_hunks = path.split(' as ')
            if len(as_hunks) == 2:
                name = as_hunks[1].strip()
                module = as_hunks[0].strip()
            else:
                name = module = path
            if name in self.imports:
                raise errors.ImportNameCollision(name)
            self.imports[name] = (from_module_path, module, {})

    def _canonicalize_import_text(self, text):
        if False:
            return 10
        'Take a list of imports, and split it into regularized form.\n\n        This is meant to take regular import text, and convert it to\n        the forms that the rest of the converters prefer.\n        '
        out = []
        cur = None
        continuing = False
        for line in text.split('\n'):
            line = line.strip()
            loc = line.find('#')
            if loc != -1:
                line = line[:loc].strip()
            if not line:
                continue
            if cur is not None:
                if line.endswith(')'):
                    out.append(cur + ' ' + line[:-1])
                    cur = None
                else:
                    cur += ' ' + line
            elif '(' in line and ')' not in line:
                cur = line.replace('(', '')
            else:
                out.append(line.replace('(', '').replace(')', ''))
        if cur is not None:
            raise errors.InvalidImportLine(cur, 'Unmatched parenthesis')
        return out

def lazy_import(scope, text, lazy_import_class=None):
    if False:
        while True:
            i = 10
    "Create lazy imports for all of the imports in text.\n\n    This is typically used as something like::\n\n        from bzrlib.lazy_import import lazy_import\n        lazy_import(globals(), '''\n        from bzrlib import (\n            foo,\n            bar,\n            baz,\n            )\n        import bzrlib.branch\n        import bzrlib.transport\n        ''')\n\n    Then 'foo, bar, baz' and 'bzrlib' will exist as lazy-loaded\n    objects which will be replaced with a real object on first use.\n\n    In general, it is best to only load modules in this way. This is\n    because other objects (functions/classes/variables) are frequently\n    used without accessing a member, which means we cannot tell they\n    have been used.\n    "
    proc = ImportProcessor(lazy_import_class=lazy_import_class)
    return proc.lazy_import(scope, text)
lazy_import(globals(), '\nfrom bzrlib import errors\n')