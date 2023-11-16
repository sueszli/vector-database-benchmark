"""Content-filtered view of any tree.
"""
from __future__ import absolute_import
from bzrlib import tree
from bzrlib.filters import ContentFilterContext, filtered_output_bytes

class ContentFilterTree(tree.Tree):
    """A virtual tree that applies content filters to an underlying tree.
    
    Not every operation is supported yet.
    """

    def __init__(self, backing_tree, filter_stack_callback):
        if False:
            i = 10
            return i + 15
        'Construct a new filtered tree view.\n\n        :param filter_stack_callback: A callable taking a path that returns\n            the filter stack that should be used for that path.\n        :param backing_tree: An underlying tree to wrap.\n        '
        self.backing_tree = backing_tree
        self.filter_stack_callback = filter_stack_callback

    def get_file_text(self, file_id, path=None):
        if False:
            print('Hello World!')
        chunks = self.backing_tree.get_file_lines(file_id, path)
        if path is None:
            path = self.backing_tree.id2path(file_id)
        filters = self.filter_stack_callback(path)
        context = ContentFilterContext(path, self, None)
        contents = filtered_output_bytes(chunks, filters, context)
        content = ''.join(contents)
        return content

    def has_filename(self, filename):
        if False:
            return 10
        return self.backing_tree.has_filename

    def is_executable(self, file_id, path=None):
        if False:
            i = 10
            return i + 15
        return self.backing_tree.is_executable(file_id, path)

    def iter_entries_by_dir(self, specific_file_ids=None, yield_parents=None):
        if False:
            print('Hello World!')
        return self.backing_tree.iter_entries_by_dir(specific_file_ids=specific_file_ids, yield_parents=yield_parents)

    def lock_read(self):
        if False:
            i = 10
            return i + 15
        return self.backing_tree.lock_read()

    def unlock(self):
        if False:
            i = 10
            return i + 15
        return self.backing_tree.unlock()