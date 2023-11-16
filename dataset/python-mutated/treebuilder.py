"""TreeBuilder helper class.

TreeBuilders are used to build trees of various shapes or properties. This
can be extremely useful in testing for instance.
"""
from __future__ import absolute_import
from bzrlib import errors

class TreeBuilder(object):
    """A TreeBuilder allows the creation of specific content in one tree at a
    time.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        'Construct a TreeBuilder.'
        self._tree = None
        self._root_done = False

    def build(self, recipe):
        if False:
            while True:
                i = 10
        "Build recipe into the current tree.\n\n        :param recipe: A sequence of paths. For each path, the corresponding\n            path in the current tree is created and added. If the path ends in\n            '/' then a directory is added, otherwise a regular file is added.\n        "
        self._ensure_building()
        if not self._root_done:
            self._tree.add('', 'root-id', 'directory')
            self._root_done = True
        for name in recipe:
            if name[-1] == '/':
                self._tree.mkdir(name[:-1])
            else:
                end = '\n'
                content = 'contents of %s%s' % (name.encode('utf-8'), end)
                self._tree.add(name, None, 'file')
                file_id = self._tree.path2id(name)
                self._tree.put_file_bytes_non_atomic(file_id, content)

    def _ensure_building(self):
        if False:
            print('Hello World!')
        'Raise NotBuilding if there is no current tree being built.'
        if self._tree is None:
            raise errors.NotBuilding

    def finish_tree(self):
        if False:
            print('Hello World!')
        'Finish building the current tree.'
        self._ensure_building()
        tree = self._tree
        self._tree = None
        tree.unlock()

    def start_tree(self, tree):
        if False:
            print('Hello World!')
        'Start building on tree.\n\n        :param tree: A tree to start building on. It must provide the\n            MutableTree interface.\n        '
        if self._tree is not None:
            raise errors.AlreadyBuilding
        self._tree = tree
        self._tree.lock_tree_write()