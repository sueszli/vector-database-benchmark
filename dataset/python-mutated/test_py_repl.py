import platform
import pytest
from .support import PyScriptTest, skip_worker
pytest.skip(reason="NEXT: pyscript NEXT doesn't support the REPL yet", allow_module_level=True)

class TestPyRepl(PyScriptTest):

    def _replace(self, py_repl, newcode):
        if False:
            for i in range(10):
                print('nop')
        '\n        Clear the editor and write new code in it.\n        WARNING: this assumes that the textbox has already the focus\n        '
        if 'macOS' in platform.platform():
            self.page.keyboard.press('Meta+A')
        else:
            self.page.keyboard.press('Control+A')
        self.page.keyboard.press('Backspace')
        self.page.keyboard.type(newcode)

    def test_repl_loads(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-repl></py-repl>\n            ')
        py_repl = self.page.query_selector('py-repl .py-repl-box')
        assert py_repl

    def test_execute_preloaded_source(self):
        if False:
            while True:
                i = 10
        "\n        Unfortunately it tests two things at once, but it's impossible to write a\n        smaller test. I think this is the most basic test that we can write.\n\n        We test that:\n            1. the source code that we put in the tag is loaded inside the editor\n            2. clicking the button executes it\n        "
        self.pyscript_run("\n            <py-repl>\n                print('hello from py-repl')\n            </py-repl>\n            ")
        py_repl = self.page.locator('py-repl')
        src = py_repl.locator('div.cm-content').inner_text()
        assert "print('hello from py-repl')" in src
        py_repl.locator('button').click()
        self.page.wait_for_selector('py-terminal')
        assert self.console.log.lines[-1] == 'hello from py-repl'

    def test_execute_code_typed_by_the_user(self):
        if False:
            return 10
        self.pyscript_run('\n            <py-repl></py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.type('print("hello")')
        py_repl.locator('button').click()
        self.page.wait_for_selector('py-terminal')
        assert self.console.log.lines[-1] == 'hello'

    def test_execute_on_shift_enter(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-repl>\n                print("hello world")\n            </py-repl>\n            ')
        self.page.wait_for_selector('py-repl .py-repl-run-button')
        self.page.keyboard.press('Shift+Enter')
        self.page.wait_for_selector('py-terminal')
        assert self.console.log.lines[-1] == 'hello world'
        assert self.page.locator('.cm-line').count() == 1

    @skip_worker('FIXME: display()')
    def test_display(self):
        if False:
            return 10
        self.pyscript_run("\n            <py-repl>\n                display('hello world')\n            </py-repl>\n            ")
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert out_div.inner_text() == 'hello world'

    @skip_worker('TIMEOUT')
    def test_show_last_expression(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that we display() the value of the last expression, as you would\n        expect by a REPL\n        '
        self.pyscript_run('\n            <py-repl>\n                42\n            </py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert out_div.inner_text() == '42'

    @skip_worker('TIMEOUT')
    def test_show_last_expression_with_output(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that we display() the value of the last expression, as you would\n        expect by a REPL\n        '
        self.pyscript_run('\n            <div id="repl-target"></div>\n            <py-repl output="repl-target">\n                42\n            </py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        out_div = py_repl.locator('div.py-repl-output')
        assert out_div.all_inner_texts()[0] == ''
        out_div = self.page.wait_for_selector('#repl-target')
        assert out_div.inner_text() == '42'

    @skip_worker('FIXME: display()')
    def test_run_clears_previous_output(self):
        if False:
            print('Hello World!')
        '\n        Check that we clear the previous output of the cell before executing it\n        again\n        '
        self.pyscript_run("\n            <py-repl>\n                display('hello world')\n            </py-repl>\n            ")
        py_repl = self.page.locator('py-repl')
        self.page.keyboard.press('Shift+Enter')
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert out_div.inner_text() == 'hello world'
        self._replace(py_repl, "display('another output')")
        self.page.keyboard.press('Shift+Enter')
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert out_div.inner_text() == 'another output'

    def test_python_exception(self):
        if False:
            i = 10
            return i + 15
        "\n        See also test01_basic::test_python_exception, since it's very similar\n        "
        self.pyscript_run("\n            <py-repl>\n                raise Exception('this is an error')\n            </py-repl>\n            ")
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        self.page.wait_for_selector('.py-error')
        tb_lines = self.console.error.lines[-1].splitlines()
        assert tb_lines[0] == '[pyexec] Python exception:'
        assert tb_lines[1] == 'Traceback (most recent call last):'
        assert tb_lines[-1] == 'Exception: this is an error'
        err_pre = py_repl.locator('div.py-repl-output > pre.py-error')
        tb_lines = err_pre.inner_text().splitlines()
        assert tb_lines[0] == 'Traceback (most recent call last):'
        assert tb_lines[-1] == 'Exception: this is an error'
        self.check_py_errors('this is an error')

    @skip_worker('FIXME: display()')
    def test_multiple_repls(self):
        if False:
            return 10
        '\n        Multiple repls showing in the correct order in the page\n        '
        self.pyscript_run('\n            <py-repl data-testid=="first"> display("first") </py-repl>\n            <py-repl data-testid=="second"> display("second") </py-repl>\n            ')
        first_py_repl = self.page.get_by_text('first')
        first_py_repl.click()
        self.page.keyboard.press('Shift+Enter')
        self.page.wait_for_selector('#py-internal-0-repl-output')
        assert self.page.inner_text('#py-internal-0-repl-output') == 'first'
        second_py_repl = self.page.get_by_text('second')
        second_py_repl.click()
        self.page.keyboard.press('Shift+Enter')
        self.page.wait_for_selector('#py-internal-1-repl-output')
        assert self.page.inner_text('#py-internal-1-repl-output') == 'second'

    @skip_worker('FIXME: display()')
    def test_python_exception_after_previous_output(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run("\n            <py-repl>\n                display('hello world')\n            </py-repl>\n            ")
        py_repl = self.page.locator('py-repl')
        self.page.keyboard.press('Shift+Enter')
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert out_div.inner_text() == 'hello world'
        self._replace(py_repl, '0/0')
        self.page.keyboard.press('Shift+Enter')
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert 'hello world' not in out_div.inner_text()
        assert 'ZeroDivisionError' in out_div.inner_text()
        self.check_py_errors('ZeroDivisionError')

    @skip_worker('FIXME: js.document')
    def test_hide_previous_error_after_successful_run(self):
        if False:
            print('Hello World!')
        "\n        this tests the fact that a new error div should be created once there's an\n        error but also that it should disappear automatically once the error\n        is fixed\n        "
        self.pyscript_run("\n            <py-repl>\n                raise Exception('this is an error')\n            </py-repl>\n            ")
        py_repl = self.page.locator('py-repl')
        self.page.keyboard.press('Shift+Enter')
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert 'this is an error' in out_div.inner_text()
        self._replace(py_repl, "display('hello')")
        self.page.keyboard.press('Shift+Enter')
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert out_div.inner_text() == 'hello'
        self.check_py_errors('this is an error')

    def test_output_attribute_does_not_exist(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If we try to use an attribute which doesn't exist, we display an error\n        instead\n        "
        self.pyscript_run('\n            <py-repl output="I-dont-exist">\n                print(\'I will not be executed\')\n            </py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        banner = self.page.wait_for_selector('.py-warning')
        banner_content = banner.inner_text()
        expected = 'output = "I-dont-exist" does not match the id of any element on the page.'
        assert banner_content == expected

    @skip_worker('TIMEOUT')
    def test_auto_generate(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <py-repl auto-generate="true">\n            </py-repl>\n            ')
        py_repls = self.page.locator('py-repl')
        outputs = py_repls.locator('div.py-repl-output')
        assert py_repls.count() == 1
        assert outputs.count() == 1
        self.page.keyboard.type("'hello'")
        self.page.keyboard.press('Shift+Enter')
        self.page.locator('py-repl[exec-id="1"]').wait_for()
        assert py_repls.count() == 2
        assert outputs.count() == 2
        self.page.keyboard.type("'world'")
        self.page.keyboard.press('Shift+Enter')
        self.page.locator('py-repl[exec-id="2"]').wait_for()
        assert py_repls.count() == 3
        assert outputs.count() == 3
        out_texts = [el.inner_text() for el in self.iter_locator(outputs)]
        assert out_texts == ['hello', 'world', '']

    @skip_worker('FIXME: display()')
    def test_multiple_repls_mixed_display_order(self):
        if False:
            print('Hello World!')
        "\n        Displaying several outputs that don't obey the order in which the original\n        repl displays were created using the auto_generate attr\n        "
        self.pyscript_run('\n            <py-repl auto-generate="true" data-testid=="first"> display("root first") </py-repl>\n            <py-repl auto-generate="true" data-testid=="second"> display("root second") </py-repl>\n            ')
        second_py_repl = self.page.get_by_text('root second')
        second_py_repl.click()
        self.page.keyboard.press('Shift+Enter')
        self.page.wait_for_selector('#py-internal-1-repl-output')
        self.page.keyboard.type("display('second children')")
        self.page.keyboard.press('Shift+Enter')
        self.page.wait_for_selector('#py-internal-1-1-repl-output')
        first_py_repl = self.page.get_by_text('root first')
        first_py_repl.click()
        self.page.keyboard.press('Shift+Enter')
        self.page.wait_for_selector('#py-internal-0-repl-output')
        self.page.keyboard.type("display('first children')")
        self.page.keyboard.press('Shift+Enter')
        self.page.wait_for_selector('#py-internal-0-1-repl-output')
        assert self.page.inner_text('#py-internal-1-1-repl-output') == 'second children'
        assert self.page.inner_text('#py-internal-0-1-repl-output') == 'first children'

    @skip_worker('FIXME: display()')
    def test_repl_output_attribute(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <div id="repl-target"></div>\n            <py-repl output="repl-target">\n                print(\'print from py-repl\')\n                display(\'display from py-repl\')\n            </py-repl>\n\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        target = self.page.wait_for_selector('#repl-target')
        assert 'print from py-repl' in target.inner_text()
        out_div = self.page.wait_for_selector('#py-internal-0-repl-output')
        assert out_div.inner_text() == 'display from py-repl'
        self.assert_no_banners()

    @skip_worker('FIXME: js.document')
    def test_repl_output_display_async(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <div id="repl-target"></div>\n            <script type="py">\n                import asyncio\n                import js\n\n                async def print_it():\n                    await asyncio.sleep(1)\n                    print(\'print from py-repl\')\n\n\n                async def display_it():\n                    display(\'display from py-repl\')\n                    await asyncio.sleep(2)\n\n                async def done():\n                    await asyncio.sleep(3)\n                    js.console.log("DONE")\n            </script>\n\n            <py-repl output="repl-target">\n                asyncio.ensure_future(print_it());\n                asyncio.ensure_future(display_it());\n                asyncio.ensure_future(done());\n            </py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        self.wait_for_console('DONE')
        assert self.page.locator('#repl-target').text_content() == ''
        self.assert_no_banners()

    @skip_worker('FIXME: js.document')
    def test_repl_stdio_dynamic_tags(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <div id="first"></div>\n            <div id="second"></div>\n            <py-repl output="first">\n                import js\n\n                print("first.")\n\n                # Using string, since no clean way to write to the\n                # code contents of the CodeMirror in a PyRepl\n                newTag = \'<py-repl id="second-repl" output="second">print("second.")</py-repl>\'\n                js.document.body.innerHTML += newTag\n            </py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        assert self.page.wait_for_selector('#first').inner_text() == 'first.\n'
        second_repl = self.page.locator('py-repl#second-repl')
        second_repl.locator('button').click()
        assert self.page.wait_for_selector('#second').inner_text() == 'second.\n'

    def test_repl_output_id_errors(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <py-repl output="not-on-page">\n                print("bad.")\n                print("bad.")\n            </py-repl>\n\n            <py-repl output="not-on-page">\n                print("bad.")\n            </py-repl>\n            ')
        py_repls = self.page.query_selector_all('py-repl')
        for repl in py_repls:
            repl.query_selector_all('button')[0].click()
        banner = self.page.wait_for_selector('.py-warning')
        banner_content = banner.inner_text()
        expected = 'output = "not-on-page" does not match the id of any element on the page.'
        assert banner_content == expected

    def test_repl_stderr_id_errors(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <py-repl stderr="not-on-page">\n                import sys\n                print("bad.", file=sys.stderr)\n                print("bad.", file=sys.stderr)\n            </py-repl>\n\n            <py-repl stderr="not-on-page">\n                print("bad.", file=sys.stderr)\n            </py-repl>\n            ')
        py_repls = self.page.query_selector_all('py-repl')
        for repl in py_repls:
            repl.query_selector_all('button')[0].click()
        banner = self.page.wait_for_selector('.py-warning')
        banner_content = banner.inner_text()
        expected = 'stderr = "not-on-page" does not match the id of any element on the page.'
        assert banner_content == expected

    def test_repl_output_stderr(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <div id="stdout-div"></div>\n            <div id="stderr-div"></div>\n            <py-repl output="stdout-div" stderr="stderr-div">\n                import sys\n                print("one.", file=sys.stderr)\n                print("two.")\n            </py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        assert self.page.wait_for_selector('#stdout-div').inner_text() == 'one.\ntwo.\n'
        assert self.page.wait_for_selector('#stderr-div').inner_text() == 'one.\n'
        self.assert_no_banners()

    @skip_worker('TIMEOUT')
    def test_repl_output_attribute_change(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <div id="first"></div>\n            <div id="second"></div>\n            <!-- There is no tag with id "third" -->\n            <py-repl id="repl-tag" output="first">\n                print("one.")\n\n                # Change the \'output\' attribute of this tag\n                import js\n                this_tag = js.document.getElementById("repl-tag")\n\n                this_tag.setAttribute("output", "second")\n                print("two.")\n\n                this_tag.setAttribute("output", "third")\n                print("three.")\n            </script>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        assert self.page.wait_for_selector('#first').inner_text() == 'one.\n'
        assert self.page.wait_for_selector('#second').inner_text() == 'two.\n'
        expected_alert_banner_msg = 'output = "third" does not match the id of any element on the page.'
        alert_banner = self.page.wait_for_selector('.alert-banner')
        assert expected_alert_banner_msg in alert_banner.inner_text()

    @skip_worker('TIMEOUT')
    def test_repl_output_element_id_change(self):
        if False:
            return 10
        self.pyscript_run('\n            <div id="first"></div>\n            <div id="second"></div>\n            <!-- There is no tag with id "third" -->\n            <py-repl id="pyscript-tag" output="first">\n                print("one.")\n\n                # Change the ID of the targeted DIV to something else\n                import js\n                target_tag = js.document.getElementById("first")\n\n                # should fail and show banner\n                target_tag.setAttribute("id", "second")\n                print("two.")\n\n                # But changing both the \'output\' attribute and the id of the target\n                # should work\n                target_tag.setAttribute("id", "third")\n                js.document.getElementById("pyscript-tag").setAttribute("output", "third")\n                print("three.")\n            </py-repl>\n            ')
        py_repl = self.page.locator('py-repl')
        py_repl.locator('button').click()
        assert self.page.wait_for_selector('#third').inner_text() == 'one.\nthree.\n'
        expected_alert_banner_msg = 'output = "first" does not match the id of any element on the page.'
        alert_banner = self.page.wait_for_selector('.alert-banner')
        assert expected_alert_banner_msg in alert_banner.inner_text()

    def test_repl_load_content_from_src(self):
        if False:
            while True:
                i = 10
        self.writefile('loadReplSrc1.py', "print('1')")
        self.pyscript_run('\n            <py-repl id="py-repl1" output="replOutput1" src="./loadReplSrc1.py"></py-repl>\n            <div id="replOutput1"></div>\n            ')
        successMsg = '[py-repl] loading code from ./loadReplSrc1.py to repl...success'
        assert self.console.info.lines[-1] == successMsg
        py_repl = self.page.locator('py-repl')
        code = py_repl.locator('div.cm-content').inner_text()
        assert "print('1')" in code

    @skip_worker('TIMEOUT')
    def test_repl_src_change(self):
        if False:
            return 10
        self.writefile('loadReplSrc2.py', '2')
        self.writefile('loadReplSrc3.py', "print('3')")
        self.pyscript_run('\n            <py-repl id="py-repl2" output="replOutput2" src="./loadReplSrc2.py"></py-repl>\n            <div id="replOutput2"></div>\n\n            <py-repl id="py-repl3" output="replOutput3">\n                import js\n                target_tag = js.document.getElementById("py-repl2")\n                target_tag.setAttribute("src", "./loadReplSrc3.py")\n            </py-repl>\n            <div id="replOutput3"></div>\n            ')
        successMsg1 = '[py-repl] loading code from ./loadReplSrc2.py to repl...success'
        assert self.console.info.lines[-1] == successMsg1
        py_repl3 = self.page.locator('py-repl#py-repl3')
        py_repl3.locator('button').click()
        py_repl2 = self.page.locator('py-repl#py-repl2')
        py_repl2.locator('button').click()
        self.page.wait_for_selector('py-terminal')
        assert self.console.log.lines[-1] == '3'
        successMsg2 = '[py-repl] loading code from ./loadReplSrc3.py to repl...success'
        assert self.console.info.lines[-1] == successMsg2

    def test_repl_src_path_that_do_not_exist(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <py-repl id="py-repl4" output="replOutput4" src="./loadReplSrc4.py"></py-repl>\n            <div id="replOutput4"></div>\n            ')
        errorMsg = '(PY0404): Fetching from URL ./loadReplSrc4.py failed with error 404 (Not Found). Are your filename and path correct?'
        assert self.console.error.lines[-1] == errorMsg