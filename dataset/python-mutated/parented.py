import warnings
from abc import ABCMeta, abstractmethod
from nltk.tree.tree import Tree
from nltk.util import slice_bounds

class AbstractParentedTree(Tree, metaclass=ABCMeta):
    """
    An abstract base class for a ``Tree`` that automatically maintains
    pointers to parent nodes.  These parent pointers are updated
    whenever any change is made to a tree's structure.  Two subclasses
    are currently defined:

      - ``ParentedTree`` is used for tree structures where each subtree
        has at most one parent.  This class should be used in cases
        where there is no"sharing" of subtrees.

      - ``MultiParentedTree`` is used for tree structures where a
        subtree may have zero or more parents.  This class should be
        used in cases where subtrees may be shared.

    Subclassing
    ===========
    The ``AbstractParentedTree`` class redefines all operations that
    modify a tree's structure to call two methods, which are used by
    subclasses to update parent information:

      - ``_setparent()`` is called whenever a new child is added.
      - ``_delparent()`` is called whenever a child is removed.
    """

    def __init__(self, node, children=None):
        if False:
            i = 10
            return i + 15
        super().__init__(node, children)
        if children is not None:
            for (i, child) in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i, dry_run=True)
            for (i, child) in enumerate(self):
                if isinstance(child, Tree):
                    self._setparent(child, i)

    @abstractmethod
    def _setparent(self, child, index, dry_run=False):
        if False:
            print('Hello World!')
        "\n        Update the parent pointer of ``child`` to point to ``self``.  This\n        method is only called if the type of ``child`` is ``Tree``;\n        i.e., it is not called when adding a leaf to a tree.  This method\n        is always called before the child is actually added to the\n        child list of ``self``.\n\n        :type child: Tree\n        :type index: int\n        :param index: The index of ``child`` in ``self``.\n        :raise TypeError: If ``child`` is a tree with an impropriate\n            type.  Typically, if ``child`` is a tree, then its type needs\n            to match the type of ``self``.  This prevents mixing of\n            different tree types (single-parented, multi-parented, and\n            non-parented).\n        :param dry_run: If true, the don't actually set the child's\n            parent pointer; just check for any error conditions, and\n            raise an exception if one is found.\n        "

    @abstractmethod
    def _delparent(self, child, index):
        if False:
            i = 10
            return i + 15
        '\n        Update the parent pointer of ``child`` to not point to self.  This\n        method is only called if the type of ``child`` is ``Tree``; i.e., it\n        is not called when removing a leaf from a tree.  This method\n        is always called before the child is actually removed from the\n        child list of ``self``.\n\n        :type child: Tree\n        :type index: int\n        :param index: The index of ``child`` in ``self``.\n        '

    def __delitem__(self, index):
        if False:
            i = 10
            return i + 15
        if isinstance(index, slice):
            (start, stop, step) = slice_bounds(self, index, allow_step=True)
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            super().__delitem__(index)
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError('index out of range')
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            super().__delitem__(index)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be deleted.')
            elif len(index) == 1:
                del self[index[0]]
            else:
                del self[index[0]][index[1:]]
        else:
            raise TypeError('%s indices must be integers, not %s' % (type(self).__name__, type(index).__name__))

    def __setitem__(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, slice):
            (start, stop, step) = slice_bounds(self, index, allow_step=True)
            if not isinstance(value, (list, tuple)):
                value = list(value)
            for (i, child) in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step, dry_run=True)
            for i in range(start, stop, step):
                if isinstance(self[i], Tree):
                    self._delparent(self[i], i)
            for (i, child) in enumerate(value):
                if isinstance(child, Tree):
                    self._setparent(child, start + i * step)
            super().__setitem__(index, value)
        elif isinstance(index, int):
            if index < 0:
                index += len(self)
            if index < 0:
                raise IndexError('index out of range')
            if value is self[index]:
                return
            if isinstance(value, Tree):
                self._setparent(value, index)
            if isinstance(self[index], Tree):
                self._delparent(self[index], index)
            super().__setitem__(index, value)
        elif isinstance(index, (list, tuple)):
            if len(index) == 0:
                raise IndexError('The tree position () may not be assigned to.')
            elif len(index) == 1:
                self[index[0]] = value
            else:
                self[index[0]][index[1:]] = value
        else:
            raise TypeError('%s indices must be integers, not %s' % (type(self).__name__, type(index).__name__))

    def append(self, child):
        if False:
            while True:
                i = 10
        if isinstance(child, Tree):
            self._setparent(child, len(self))
        super().append(child)

    def extend(self, children):
        if False:
            for i in range(10):
                print('nop')
        for child in children:
            if isinstance(child, Tree):
                self._setparent(child, len(self))
            super().append(child)

    def insert(self, index, child):
        if False:
            while True:
                i = 10
        if index < 0:
            index += len(self)
        if index < 0:
            index = 0
        if isinstance(child, Tree):
            self._setparent(child, index)
        super().insert(index, child)

    def pop(self, index=-1):
        if False:
            for i in range(10):
                print('nop')
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError('index out of range')
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        return super().pop(index)

    def remove(self, child):
        if False:
            i = 10
            return i + 15
        index = self.index(child)
        if isinstance(self[index], Tree):
            self._delparent(self[index], index)
        super().remove(child)
    if hasattr(list, '__getslice__'):

        def __getslice__(self, start, stop):
            if False:
                while True:
                    i = 10
            return self.__getitem__(slice(max(0, start), max(0, stop)))

        def __delslice__(self, start, stop):
            if False:
                return 10
            return self.__delitem__(slice(max(0, start), max(0, stop)))

        def __setslice__(self, start, stop, value):
            if False:
                i = 10
                return i + 15
            return self.__setitem__(slice(max(0, start), max(0, stop)), value)

    def __getnewargs__(self):
        if False:
            return 10
        'Method used by the pickle module when un-pickling.\n        This method provides the arguments passed to ``__new__``\n        upon un-pickling. Without this method, ParentedTree instances\n        cannot be pickled and unpickled in Python 3.7+ onwards.\n\n        :return: Tuple of arguments for ``__new__``, i.e. the label\n            and the children of this node.\n        :rtype: Tuple[Any, List[AbstractParentedTree]]\n        '
        return (self._label, list(self))

class ParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    single-parented trees.  The following are methods for querying
    the structure of a parented tree: ``parent``, ``parent_index``,
    ``left_sibling``, ``right_sibling``, ``root``, ``treeposition``.

    Each ``ParentedTree`` may have at most one parent.  In
    particular, subtrees may not be shared.  Any attempt to reuse a
    single ``ParentedTree`` as a child of more than one parent (or
    as multiple children of the same parent) will cause a
    ``ValueError`` exception to be raised.

    ``ParentedTrees`` should never be used in the same tree as ``Trees``
    or ``MultiParentedTrees``.  Mixing tree implementations may result
    in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        if False:
            i = 10
            return i + 15
        self._parent = None
        'The parent of this Tree, or None if it has no parent.'
        super().__init__(node, children)
        if children is None:
            for (i, child) in enumerate(self):
                if isinstance(child, Tree):
                    child._parent = None
                    self._setparent(child, i)

    def _frozen_class(self):
        if False:
            return 10
        from nltk.tree.immutable import ImmutableParentedTree
        return ImmutableParentedTree

    def copy(self, deep=False):
        if False:
            i = 10
            return i + 15
        if not deep:
            warnings.warn(f'{self.__class__.__name__} objects do not support shallow copies. Defaulting to a deep copy.')
        return super().copy(deep=True)

    def parent(self):
        if False:
            for i in range(10):
                print('nop')
        'The parent of this tree, or None if it has no parent.'
        return self._parent

    def parent_index(self):
        if False:
            i = 10
            return i + 15
        '\n        The index of this tree in its parent.  I.e.,\n        ``ptree.parent()[ptree.parent_index()] is ptree``.  Note that\n        ``ptree.parent_index()`` is not necessarily equal to\n        ``ptree.parent.index(ptree)``, since the ``index()`` method\n        returns the first child that is equal to its argument.\n        '
        if self._parent is None:
            return None
        for (i, child) in enumerate(self._parent):
            if child is self:
                return i
        assert False, 'expected to find self in self._parent!'

    def left_sibling(self):
        if False:
            print('Hello World!')
        'The left sibling of this tree, or None if it has none.'
        parent_index = self.parent_index()
        if self._parent and parent_index > 0:
            return self._parent[parent_index - 1]
        return None

    def right_sibling(self):
        if False:
            for i in range(10):
                print('nop')
        'The right sibling of this tree, or None if it has none.'
        parent_index = self.parent_index()
        if self._parent and parent_index < len(self._parent) - 1:
            return self._parent[parent_index + 1]
        return None

    def root(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The root of this tree.  I.e., the unique ancestor of this tree\n        whose parent is None.  If ``ptree.parent()`` is None, then\n        ``ptree`` is its own root.\n        '
        root = self
        while root.parent() is not None:
            root = root.parent()
        return root

    def treeposition(self):
        if False:
            while True:
                i = 10
        '\n        The tree position of this tree, relative to the root of the\n        tree.  I.e., ``ptree.root[ptree.treeposition] is ptree``.\n        '
        if self.parent() is None:
            return ()
        else:
            return self.parent().treeposition() + (self.parent_index(),)

    def _delparent(self, child, index):
        if False:
            i = 10
            return i + 15
        assert isinstance(child, ParentedTree)
        assert self[index] is child
        assert child._parent is self
        child._parent = None

    def _setparent(self, child, index, dry_run=False):
        if False:
            print('Hello World!')
        if not isinstance(child, ParentedTree):
            raise TypeError('Can not insert a non-ParentedTree into a ParentedTree')
        if hasattr(child, '_parent') and child._parent is not None:
            raise ValueError('Can not insert a subtree that already has a parent.')
        if not dry_run:
            child._parent = self

class MultiParentedTree(AbstractParentedTree):
    """
    A ``Tree`` that automatically maintains parent pointers for
    multi-parented trees.  The following are methods for querying the
    structure of a multi-parented tree: ``parents()``, ``parent_indices()``,
    ``left_siblings()``, ``right_siblings()``, ``roots``, ``treepositions``.

    Each ``MultiParentedTree`` may have zero or more parents.  In
    particular, subtrees may be shared.  If a single
    ``MultiParentedTree`` is used as multiple children of the same
    parent, then that parent will appear multiple times in its
    ``parents()`` method.

    ``MultiParentedTrees`` should never be used in the same tree as
    ``Trees`` or ``ParentedTrees``.  Mixing tree implementations may
    result in incorrect parent pointers and in ``TypeError`` exceptions.
    """

    def __init__(self, node, children=None):
        if False:
            i = 10
            return i + 15
        self._parents = []
        "A list of this tree's parents.  This list should not\n           contain duplicates, even if a parent contains this tree\n           multiple times."
        super().__init__(node, children)
        if children is None:
            for (i, child) in enumerate(self):
                if isinstance(child, Tree):
                    child._parents = []
                    self._setparent(child, i)

    def _frozen_class(self):
        if False:
            return 10
        from nltk.tree.immutable import ImmutableMultiParentedTree
        return ImmutableMultiParentedTree

    def parents(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The set of parents of this tree.  If this tree has no parents,\n        then ``parents`` is the empty set.  To check if a tree is used\n        as multiple children of the same parent, use the\n        ``parent_indices()`` method.\n\n        :type: list(MultiParentedTree)\n        '
        return list(self._parents)

    def left_siblings(self):
        if False:
            print('Hello World!')
        '\n        A list of all left siblings of this tree, in any of its parent\n        trees.  A tree may be its own left sibling if it is used as\n        multiple contiguous children of the same parent.  A tree may\n        appear multiple times in this list if it is the left sibling\n        of this tree with respect to multiple parents.\n\n        :type: list(MultiParentedTree)\n        '
        return [parent[index - 1] for (parent, index) in self._get_parent_indices() if index > 0]

    def right_siblings(self):
        if False:
            print('Hello World!')
        '\n        A list of all right siblings of this tree, in any of its parent\n        trees.  A tree may be its own right sibling if it is used as\n        multiple contiguous children of the same parent.  A tree may\n        appear multiple times in this list if it is the right sibling\n        of this tree with respect to multiple parents.\n\n        :type: list(MultiParentedTree)\n        '
        return [parent[index + 1] for (parent, index) in self._get_parent_indices() if index < len(parent) - 1]

    def _get_parent_indices(self):
        if False:
            for i in range(10):
                print('nop')
        return [(parent, index) for parent in self._parents for (index, child) in enumerate(parent) if child is self]

    def roots(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The set of all roots of this tree.  This set is formed by\n        tracing all possible parent paths until trees with no parents\n        are found.\n\n        :type: list(MultiParentedTree)\n        '
        return list(self._get_roots_helper({}).values())

    def _get_roots_helper(self, result):
        if False:
            for i in range(10):
                print('nop')
        if self._parents:
            for parent in self._parents:
                parent._get_roots_helper(result)
        else:
            result[id(self)] = self
        return result

    def parent_indices(self, parent):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of the indices where this tree occurs as a child\n        of ``parent``.  If this child does not occur as a child of\n        ``parent``, then the empty list is returned.  The following is\n        always true::\n\n          for parent_index in ptree.parent_indices(parent):\n              parent[parent_index] is ptree\n        '
        if parent not in self._parents:
            return []
        else:
            return [index for (index, child) in enumerate(parent) if child is self]

    def treepositions(self, root):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of all tree positions that can be used to reach\n        this multi-parented tree starting from ``root``.  I.e., the\n        following is always true::\n\n          for treepos in ptree.treepositions(root):\n              root[treepos] is ptree\n        '
        if self is root:
            return [()]
        else:
            return [treepos + (index,) for parent in self._parents for treepos in parent.treepositions(root) for (index, child) in enumerate(parent) if child is self]

    def _delparent(self, child, index):
        if False:
            print('Hello World!')
        assert isinstance(child, MultiParentedTree)
        assert self[index] is child
        assert len([p for p in child._parents if p is self]) == 1
        for (i, c) in enumerate(self):
            if c is child and i != index:
                break
        else:
            child._parents.remove(self)

    def _setparent(self, child, index, dry_run=False):
        if False:
            print('Hello World!')
        if not isinstance(child, MultiParentedTree):
            raise TypeError('Can not insert a non-MultiParentedTree into a MultiParentedTree')
        if not dry_run:
            for parent in child._parents:
                if parent is self:
                    break
            else:
                child._parents.append(self)
__all__ = ['ParentedTree', 'MultiParentedTree']