"""Tests util.utils functions."""
import os
from docutils import nodes
from sphinx.util.docutils import SphinxFileOutput, SphinxTranslator, docutils_namespace, new_document, register_node

def test_register_node():
    if False:
        while True:
            i = 10

    class custom_node(nodes.Element):
        pass
    with docutils_namespace():
        register_node(custom_node)
        assert hasattr(nodes.GenericNodeVisitor, 'visit_custom_node')
        assert hasattr(nodes.GenericNodeVisitor, 'depart_custom_node')
        assert hasattr(nodes.SparseNodeVisitor, 'visit_custom_node')
        assert hasattr(nodes.SparseNodeVisitor, 'depart_custom_node')
    assert not hasattr(nodes.GenericNodeVisitor, 'visit_custom_node')
    assert not hasattr(nodes.GenericNodeVisitor, 'depart_custom_node')
    assert not hasattr(nodes.SparseNodeVisitor, 'visit_custom_node')
    assert not hasattr(nodes.SparseNodeVisitor, 'depart_custom_node')

def test_SphinxFileOutput(tmpdir):
    if False:
        print('Hello World!')
    content = 'Hello Sphinx World'
    filename = str(tmpdir / 'test.txt')
    output = SphinxFileOutput(destination_path=filename)
    output.write(content)
    os.utime(filename, (0, 0))
    output.write(content)
    assert os.stat(filename).st_mtime != 0
    filename = str(tmpdir / 'test2.txt')
    output = SphinxFileOutput(destination_path=filename, overwrite_if_changed=True)
    output.write(content)
    os.utime(filename, (0, 0))
    output.write(content)
    assert os.stat(filename).st_mtime == 0
    output.write(content + '; content change')
    assert os.stat(filename).st_mtime != 0

def test_SphinxTranslator(app):
    if False:
        return 10

    class CustomNode(nodes.inline):
        pass

    class MyTranslator(SphinxTranslator):

        def __init__(self, *args):
            if False:
                i = 10
                return i + 15
            self.called = []
            super().__init__(*args)

        def visit_document(self, node):
            if False:
                while True:
                    i = 10
            pass

        def depart_document(self, node):
            if False:
                while True:
                    i = 10
            pass

        def visit_inline(self, node):
            if False:
                i = 10
                return i + 15
            self.called.append('visit_inline')

        def depart_inline(self, node):
            if False:
                return 10
            self.called.append('depart_inline')
    document = new_document('')
    document += CustomNode()
    translator = MyTranslator(document, app.builder)
    document.walkabout(translator)
    assert translator.called == ['visit_inline', 'depart_inline']