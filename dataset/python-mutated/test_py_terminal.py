import time
import pytest
from playwright.sync_api import expect
from .support import PageErrors, PyScriptTest, only_worker, skip_worker

class TestPyTerminal(PyScriptTest):

    def test_multiple_terminals(self):
        if False:
            i = 10
            return i + 15
        '\n        Multiple terminals are not currently supported\n        '
        self.pyscript_run('\n            <script type="py" terminal></script>\n            <script type="py" terminal></script>\n            ', wait_for_pyscript=False, check_js_errors=False)
        assert self.assert_banner_message('You can use at most 1 terminal')
        with pytest.raises(PageErrors, match='You can use at most 1 terminal'):
            self.check_js_errors()

    @only_worker
    def test_py_terminal_os_write(self):
        if False:
            while True:
                i = 10
        '\n        An `os.write("text")` should land in the terminal\n        '
        self.pyscript_run('\n            <script type="py" terminal>\n                import os\n                os.write(1, str.encode("hello\\n"))\n                os.write(2, str.encode("world\\n"))\n            </script>\n            ', wait_for_pyscript=False)
        self.page.get_by_text('hello\n').wait_for()
        self.page.get_by_text('world\n').wait_for()

    def test_py_terminal(self):
        if False:
            i = 10
            return i + 15
        '\n        1. <py-terminal> should redirect stdout and stderr to the DOM\n\n        2. they also go to the console as usual\n        '
        self.pyscript_run('\n            <script type="py" terminal>\n                import sys\n                print(\'hello world\')\n                print(\'this goes to stderr\', file=sys.stderr)\n                print(\'this goes to stdout\')\n            </script>\n            ', wait_for_pyscript=False)
        self.page.get_by_text('hello world').wait_for()
        term = self.page.locator('py-terminal')
        term_lines = term.inner_text().splitlines()
        assert term_lines[0:3] == ['hello world', 'this goes to stderr', 'this goes to stdout']

    @skip_worker("Workers don't have events + two different workers don't share the same I/O")
    def test_button_action(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                def greetings(event):\n                    print(\'hello world\')\n            </script>\n            <script type="py" terminal></script>\n\n            <button id="my-button" py-click="greetings">Click me</button>\n            ')
        term = self.page.locator('py-terminal')
        self.page.locator('button').click()
        last_line = self.page.get_by_text('hello world')
        last_line.wait_for()
        assert term.inner_text().rstrip() == 'hello world'

    def test_xterm_function(self):
        if False:
            i = 10
            return i + 15
        "Test a few basic behaviors of the xtermjs terminal.\n\n        This test isn't meant to capture all of the behaviors of an xtermjs terminal;\n        rather, it confirms with a few basic formatting sequences that (1) the xtermjs\n        terminal is functioning/loaded correctly and (2) that output toward that terminal\n        isn't being escaped in a way that prevents it reacting to escape seqeunces. The\n        main goal is preventing regressions.\n        "
        self.pyscript_run('\n            <script type="py" terminal>\n                print("\x1b[33mYellow\x1b[0m")\n                print("\x1b[4mUnderline\x1b[24m")\n                print("\x1b[1mBold\x1b[22m")\n                print("\x1b[3mItalic\x1b[23m")\n                print("done")\n            </script>\n            ', wait_for_pyscript=False)
        last_line = self.page.get_by_text('done')
        last_line.wait_for()
        time.sleep(1)
        rows = self.page.locator('.xterm-rows')
        first_line = rows.locator('div').nth(0)
        first_char = first_line.locator('span').nth(0)
        color = first_char.evaluate("(element) => getComputedStyle(element).getPropertyValue('color')")
        assert color == 'rgb(196, 160, 0)'
        second_line = rows.locator('div').nth(1)
        first_char = second_line.locator('span').nth(0)
        text_decoration = first_char.evaluate("(element) => getComputedStyle(element).getPropertyValue('text-decoration')")
        assert 'underline' in text_decoration
        baseline_font_weight = first_char.evaluate("(element) => getComputedStyle(element).getPropertyValue('font-weight')")
        third_line = rows.locator('div').nth(2)
        first_char = third_line.locator('span').nth(0)
        font_weight = first_char.evaluate("(element) => getComputedStyle(element).getPropertyValue('font-weight')")
        assert int(font_weight) > int(baseline_font_weight)
        fourth_line = rows.locator('div').nth(3)
        first_char = fourth_line.locator('span').nth(0)
        font_style = first_char.evaluate("(element) => getComputedStyle(element).getPropertyValue('font-style')")
        assert font_style == 'italic'