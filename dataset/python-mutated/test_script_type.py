import pytest
from .support import PyScriptTest, with_execution_thread

@with_execution_thread(None)
class TestScriptTypePyScript(PyScriptTest):

    def test_display_line_break(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello\\nworld\')\n            </script>\n            ')
        text_content = self.page.locator('script-py').text_content()
        assert 'hello\nworld' == text_content

    def test_amp(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'a &amp; b\')\n            </script>\n            ')
        text_content = self.page.locator('script-py').text_content()
        assert 'a &amp; b' == text_content

    def test_quot(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'a &quot; b\')\n            </script>\n            ')
        text_content = self.page.locator('script-py').text_content()
        assert 'a &quot; b' == text_content

    def test_lt_gt(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'< &lt; &gt; >\')\n            </script>\n            ')
        text_content = self.page.locator('script-py').text_content()
        assert '< &lt; &gt; >' == text_content

    def test_dynamically_add_script_type_py_tag(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <script>\n                function addPyScriptTag() {\n                    let tag = document.createElement(\'script\');\n                    tag.type = \'py\';\n                    tag.textContent = "print(\'hello world\')";\n                    document.body.appendChild(tag);\n                }\n                addPyScriptTag();\n            </script>\n            ')
        self.page.locator('script-py')
        assert self.console.log.lines[-1] == 'hello world'

    def test_script_type_py_src_attribute(self):
        if False:
            print('Hello World!')
        self.writefile('foo.py', "print('hello from foo')")
        self.pyscript_run('\n            <script type="py" src="foo.py"></script>\n            ')
        assert self.console.log.lines[-1] == 'hello from foo'

    def test_script_type_py_worker_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        self.writefile('foo.py', "print('hello from foo')")
        self.pyscript_run('\n            <script type="py" src="foo.py" worker></script>\n            ')
        assert self.console.log.lines[-1] == 'hello from foo'

    @pytest.mark.skip('FIXME: output attribute is not implemented')
    def test_script_type_py_output_attribute(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <div id="first"></div>\n            <script type="py" output="first">\n                print("<p>Hello</p>")\n            </script>\n            ')
        text = self.page.locator('#first').text_content()
        assert '<p>Hello</p>' in text

    @pytest.mark.skip('FIXME: stderr attribute is not implemented')
    def test_script_type_py_stderr_attribute(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <div id="stdout-div"></div>\n            <div id="stderr-div"></div>\n            <script type="py" output="stdout-div" stderr="stderr-div">\n                import sys\n                print("one.", file=sys.stderr)\n                print("two.")\n            </script>\n            ')
        assert self.page.locator('#stdout-div').text_content() == 'one.two.'
        assert self.page.locator('#stderr-div').text_content() == 'one.'