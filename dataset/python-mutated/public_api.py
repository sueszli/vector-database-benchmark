"""Visitor restricting traversal to only the public tensorflow API."""
import re
from tensorflow.python.util import tf_inspect

class PublicAPIVisitor:
    """Visitor to use with `traverse` to visit exactly the public TF API."""

    def __init__(self, visitor):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    `visitor` should be a callable suitable as a visitor for `traverse`. It will\n    be called only for members of the public TensorFlow API.\n\n    Args:\n      visitor: A visitor to call for the public API.\n    '
        self._visitor = visitor
        self._root_name = 'tf'
        self._private_map = {'tf': ['compiler', 'core', 'security', 'dtensor', 'python', 'tsl'], 'tf.flags': ['cpp_flags']}
        self._do_not_descend_map = {'tf': ['examples', 'flags', 'platform', 'pywrap_tensorflow', 'user_ops', 'tools', 'tensorboard'], 'tf.app': ['flags'], 'tf.test': ['mock']}

    @property
    def private_map(self):
        if False:
            while True:
                i = 10
        'A map from parents to symbols that should not be included at all.\n\n    This map can be edited, but it should not be edited once traversal has\n    begun.\n\n    Returns:\n      The map marking symbols to not include.\n    '
        return self._private_map

    @property
    def do_not_descend_map(self):
        if False:
            while True:
                i = 10
        'A map from parents to symbols that should not be descended into.\n\n    This map can be edited, but it should not be edited once traversal has\n    begun.\n\n    Returns:\n      The map marking symbols to not explore.\n    '
        return self._do_not_descend_map

    def set_root_name(self, root_name):
        if False:
            while True:
                i = 10
        "Override the default root name of 'tf'."
        self._root_name = root_name

    def _is_private(self, path, name, obj=None):
        if False:
            for i in range(10):
                print('nop')
        'Return whether a name is private.'
        del obj
        return path in self._private_map and name in self._private_map[path] or (name.startswith('_') and (not re.match('__.*__$', name)) or name in ['__base__', '__class__', '__next_in_mro__'])

    def _do_not_descend(self, path, name):
        if False:
            i = 10
            return i + 15
        'Safely queries if a specific fully qualified name should be excluded.'
        return path in self._do_not_descend_map and name in self._do_not_descend_map[path]

    def __call__(self, path, parent, children):
        if False:
            print('Hello World!')
        'Visitor interface, see `traverse` for details.'
        if tf_inspect.ismodule(parent) and len(path.split('.')) > 10:
            raise RuntimeError('Modules nested too deep:\n%s.%s\n\nThis is likely a problem with an accidental public import.' % (self._root_name, path))
        full_path = '.'.join([self._root_name, path]) if path else self._root_name
        for (name, child) in list(children):
            if self._is_private(full_path, name, child):
                children.remove((name, child))
        self._visitor(path, parent, children)
        for (name, child) in list(children):
            if self._do_not_descend(full_path, name):
                children.remove((name, child))