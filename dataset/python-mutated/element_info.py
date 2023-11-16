"""Interface for classes which should deal with different backend elements"""
from six import integer_types

class ElementInfo(object):
    """Abstract wrapper for an element"""

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Representation of the element info object\n\n        The method prints the following info:\n        * type name as a module name and a class name of the object\n        * title of the control or empty string\n        * class name of the control\n        * unique ID of the control, usually a handle\n        '
        return '<{0}, {1}>'.format(self.__str__(), self.handle)

    def __str__(self):
        if False:
            while True:
                i = 10
        'Pretty print representation of the element info object\n\n        The method prints the following info:\n        * type name as a module name and class name of the object\n        * title of the control or empty string\n        * class name of the control\n        '
        module = self.__class__.__module__
        module = module[module.rfind('.') + 1:]
        type_name = module + '.' + self.__class__.__name__
        return "{0} - '{1}', {2}".format(type_name, self.name, self.class_name)

    def set_cache_strategy(self, cached):
        if False:
            i = 10
            return i + 15
        'Set a cache strategy for frequently used attributes of the element'
        raise NotImplementedError()

    @property
    def handle(self):
        if False:
            print('Hello World!')
        'Return the handle of the element'
        raise NotImplementedError()

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'Return the name of the element'
        raise NotImplementedError()

    @property
    def rich_text(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the text of the element'
        raise NotImplementedError()

    @property
    def control_id(self):
        if False:
            return 10
        'Return the ID of the control'
        raise NotImplementedError()

    @property
    def process_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the ID of process that controls this element'
        raise NotImplementedError()

    @property
    def framework_id(self):
        if False:
            i = 10
            return i + 15
        'Return the framework of the element'
        raise NotImplementedError()

    @property
    def class_name(self):
        if False:
            i = 10
            return i + 15
        'Return the class name of the element'
        raise NotImplementedError()

    @property
    def enabled(self):
        if False:
            return 10
        'Return True if the element is enabled'
        raise NotImplementedError()

    @property
    def visible(self):
        if False:
            return 10
        'Return True if the element is visible'
        raise NotImplementedError()

    @property
    def parent(self):
        if False:
            return 10
        'Return the parent of the element'
        raise NotImplementedError()

    @property
    def top_level_parent(self):
        if False:
            while True:
                i = 10
        '\n        Return the top level window of this element\n\n        The TopLevel parent is different from the parent in that the parent\n        is the element that owns this element - but it may not be a dialog/main\n        window. For example most Comboboxes have an Edit. The ComboBox is the\n        parent of the Edit control.\n\n        This will always return a valid window element (if the control has\n        no top level parent then the control itself is returned - as it is\n        a top level window already!)\n        '
        parent = self.parent
        if parent and parent != self.__class__():
            return parent.top_level_parent
        else:
            return self

    def children(self, **kwargs):
        if False:
            print('Hello World!')
        'Return children of the element'
        raise NotImplementedError()

    def iter_children(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Iterate over children of element'
        raise NotImplementedError()

    def has_depth(self, root, depth):
        if False:
            return 10
        'Return True if element has particular depth level relative to the root'
        if self.control_id != root.control_id:
            if depth > 0:
                parent = self.parent
                return parent.has_depth(root, depth - 1)
            else:
                return False
        else:
            return True

    @staticmethod
    def filter_with_depth(elements, root, depth):
        if False:
            for i in range(10):
                print('nop')
        'Return filtered elements with particular depth level relative to the root'
        if depth is not None:
            if isinstance(depth, integer_types) and depth > 0:
                return [element for element in elements if element.has_depth(root, depth)]
            else:
                raise Exception('Depth must be natural number')
        else:
            return elements

    def get_descendants_with_depth(self, depth=None, **kwargs):
        if False:
            while True:
                i = 10
        'Return a list of all descendant children of the element with the specified depth'
        descendants = []

        def walk_the_tree(root, depth, **kwargs):
            if False:
                while True:
                    i = 10
            if depth == 0:
                return
            for child in root.children(**kwargs):
                descendants.append(child)
                next_depth = None if depth is None else depth - 1
                walk_the_tree(child, next_depth, **kwargs)
        walk_the_tree(self, depth, **kwargs)
        return descendants

    def descendants(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Return descendants of the element'
        raise NotImplementedError()

    def iter_descendants(self, **kwargs):
        if False:
            while True:
                i = 10
        'Iterate over descendants of the element'
        depth = kwargs.pop('depth', None)
        if depth == 0:
            return
        for child in self.iter_children(**kwargs):
            yield child
            if depth is not None:
                kwargs['depth'] = depth - 1
            for c in child.iter_descendants(**kwargs):
                yield c

    @property
    def rectangle(self):
        if False:
            print('Hello World!')
        'Return rectangle of element'
        raise NotImplementedError()

    def dump_window(self):
        if False:
            return 10
        'Dump an element to a set of properties'
        raise NotImplementedError()