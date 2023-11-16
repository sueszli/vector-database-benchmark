import logging
from spyder.config.base import _
from spyder.plugins.variableexplorer.widgets.objectexplorer.utils import cut_off_str
logger = logging.getLogger(__name__)
MAX_OBJ_STR_LEN = 50

def name_is_special(method_name):
    if False:
        i = 10
        return i + 15
    'Returns true if the method name starts and ends with two underscores.'
    return method_name.startswith('__') and method_name.endswith('__')

class TreeItem(object):
    """Tree node class that can be used to build trees of objects."""

    def __init__(self, obj, name, obj_path, is_attribute, parent=None):
        if False:
            while True:
                i = 10
        self.parent_item = parent
        self.obj = obj
        self.obj_name = str(name)
        self.obj_path = str(obj_path)
        self.is_attribute = is_attribute
        self.child_items = []
        self.has_children = True
        self.children_fetched = False

    def __str__(self):
        if False:
            i = 10
            return i + 15
        n_children = len(self.child_items)
        if n_children == 0:
            return _('<TreeItem(0x{:x}): {} = {}>').format(id(self.obj), self.obj_path, cut_off_str(self.obj, MAX_OBJ_STR_LEN))
        else:
            return _('<TreeItem(0x{:x}): {} ({:d} children)>').format(id(self.obj), self.obj_path, len(self.child_items))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        n_children = len(self.child_items)
        return _('<TreeItem(0x{:x}): {} ({:d} children)>').format(id(self.obj), self.obj_path, n_children)

    @property
    def is_special_attribute(self):
        if False:
            i = 10
            return i + 15
        '\n        Return true if the items is an attribute and its\n        name begins and end with 2 underscores.\n        '
        return self.is_attribute and name_is_special(self.obj_name)

    @property
    def is_callable_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        'Return true if the items is an attribute and it is callable.'
        return self.is_attribute and self.is_callable

    @property
    def is_callable(self):
        if False:
            while True:
                i = 10
        'Return true if the underlying object is callable.'
        return callable(self.obj)

    def append_child(self, item):
        if False:
            while True:
                i = 10
        item.parent_item = self
        self.child_items.append(item)

    def insert_children(self, idx, items):
        if False:
            i = 10
            return i + 15
        self.child_items[idx:idx] = items
        for item in items:
            item.parent_item = self

    def child(self, row):
        if False:
            return 10
        return self.child_items[row]

    def child_count(self):
        if False:
            i = 10
            return i + 15
        return len(self.child_items)

    def parent(self):
        if False:
            for i in range(10):
                print('nop')
        return self.parent_item

    def row(self):
        if False:
            return 10
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        else:
            return 0

    def pretty_print(self, indent=0):
        if False:
            i = 10
            return i + 15
        logger.debug(indent * '    ' + str(self))
        for child_item in self.child_items:
            child_item.pretty_print(indent + 1)