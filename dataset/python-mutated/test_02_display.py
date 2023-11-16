import base64
import html
import io
import os
import re
import numpy as np
import pytest
from PIL import Image
from .support import PageErrors, PyScriptTest, filter_inner_text, filter_page_content, only_main, skip_worker, wait_for_render
DISPLAY_OUTPUT_ID_PATTERN = 'script-py[id^="py-"]'

class TestDisplay(PyScriptTest):

    def test_simple_display(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py">\n                print(\'ciao\')\n                from pyscript import display\n                display("hello world")\n            </script>\n            ', timeout=20000)
        node_list = self.page.query_selector_all(DISPLAY_OUTPUT_ID_PATTERN)
        pattern = '<div>hello world</div>'
        assert node_list[0].inner_html() == pattern
        assert len(node_list) == 1

    def test_consecutive_display(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello 1\')\n            </script>\n            <p>hello 2</p>\n            <script type="py">\n                from pyscript import display\n                display(\'hello 3\')\n            </script>\n            ')
        inner_text = self.page.inner_text('body')
        lines = inner_text.splitlines()
        lines = [line for line in filter_page_content(lines)]
        assert lines == ['hello 1', 'hello 2', 'hello 3']

    def test_target_parameter(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello world\', target="mydiv")\n            </script>\n            <div id="mydiv"></div>\n            ')
        mydiv = self.page.locator('#mydiv')
        assert mydiv.inner_text() == 'hello world'

    def test_target_parameter_with_sharp(self):
        if False:
            return 10
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello world\', target="#mydiv")\n            </script>\n            <div id="mydiv"></div>\n            ')
        mydiv = self.page.locator('#mydiv')
        assert mydiv.inner_text() == 'hello world'

    def test_non_existing_id_target_raises_value_error(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello world\', target="non-existing")\n            </script>\n            ')
        error_msg = f'Invalid selector with id=non-existing. Cannot be found in the page.'
        self.check_py_errors(f'ValueError: {error_msg}')

    def test_empty_string_target_raises_value_error(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello world\', target="")\n            </script>\n            ')
        self.check_py_errors(f'ValueError: Cannot have an empty target')

    def test_non_string_target_values_raise_typerror(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display("hello False", target=False)\n            </script>\n            ')
        error_msg = f'target must be str or None, not bool'
        self.check_py_errors(f'TypeError: {error_msg}')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display("hello False", target=123)\n            </script>\n            ')
        error_msg = f'target must be str or None, not int'
        self.check_py_errors(f'TypeError: {error_msg}')

    @skip_worker('NEXT: display(target=...) does not work')
    def test_tag_target_attribute(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py" target="hello">\n                from pyscript import display\n                display(\'hello\')\n                display("goodbye world", target="goodbye")\n                display(\'world\')\n            </script>\n            <div id="hello"></div>\n            <div id="goodbye"></div>\n            ')
        hello = self.page.locator('#hello')
        assert hello.inner_text() == 'hello\nworld'
        goodbye = self.page.locator('#goodbye')
        assert goodbye.inner_text() == 'goodbye world'

    @skip_worker('NEXT: display target does not work properly')
    def test_target_script_py(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <div>ONE</div>\n            <script type="py" id="two">\n                # just a placeholder\n            </script>\n            <div>THREE</div>\n\n            <script type="py">\n                from pyscript import display\n                display(\'TWO\', target="two")\n            </script>\n            ')
        text = self.page.inner_text('body')
        assert text == 'ONE\nTWO\nTHREE'

    @skip_worker('NEXT: display target does not work properly')
    def test_consecutive_display_target(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py" id="first">\n                from pyscript import display\n                display(\'hello 1\')\n            </script>\n                <p>hello in between 1 and 2</p>\n            <script type="py" id="second">\n                from pyscript import display\n                display(\'hello 2\', target="second")\n            </script>\n            <script type="py" id="third">\n                from pyscript import display\n                display(\'hello 3\')\n            </script>\n            ')
        inner_text = self.page.inner_text('body')
        lines = inner_text.splitlines()
        lines = [line for line in filter_page_content(lines)]
        assert lines == ['hello 1', 'hello in between 1 and 2', 'hello 2', 'hello 3']

    def test_multiple_display_calls_same_tag(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello\')\n                display(\'world\')\n            </script>\n        ')
        tag = self.page.locator('script-py')
        lines = tag.inner_text().splitlines()
        assert lines == ['hello', 'world']

    @only_main
    def test_implicit_target_from_a_different_tag(self):
        if False:
            return 10
        self.pyscript_run('\n                <script type="py">\n                    from pyscript import display\n                    def say_hello():\n                        display(\'hello\')\n                </script>\n\n                <script type="py">\n                    from pyscript import display\n                    say_hello()\n                </script>\n            ')
        elems = self.page.locator('script-py')
        py0 = elems.nth(0)
        py1 = elems.nth(1)
        assert py0.inner_text() == ''
        assert py1.inner_text() == 'hello'

    @skip_worker("NEXT: py-click doesn't work")
    def test_no_explicit_target(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n                <script type="py">\n                    from pyscript import display\n                    def display_hello(error):\n                        display(\'hello world\')\n                </script>\n                <button id="my-button" py-click="display_hello">Click me</button>\n            ')
        self.page.locator('button').click()
        text = self.page.locator('script-py').text_content()
        assert 'hello world' in text

    @skip_worker('NEXT: display target does not work properly')
    def test_explicit_target_pyscript_tag(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                def display_hello():\n                    display(\'hello\', target=\'second-pyscript-tag\')\n            </script>\n            <script type="py" id="second-pyscript-tag">\n                display_hello()\n            </script>\n            ')
        text = self.page.locator('script-py').nth(1).inner_text()
        assert text == 'hello'

    @skip_worker('NEXT: display target does not work properly')
    def test_explicit_target_on_button_tag(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                def display_hello(error):\n                    display(\'hello\', target=\'my-button\')\n            </script>\n            <button id="my-button" py-click="display_hello">Click me</button>\n        ')
        self.page.locator('text=Click me').click()
        text = self.page.locator('id=my-button').inner_text()
        assert 'hello' in text

    def test_append_true(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'AAA\', append=True)\n                display(\'BBB\', append=True)\n            </script>\n        ')
        output = self.page.locator('script-py')
        assert output.inner_text() == 'AAA\nBBB'

    def test_append_false(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'AAA\', append=False)\n                display(\'BBB\', append=False)\n            </script>\n        ')
        output = self.page.locator('script-py')
        assert output.inner_text() == 'BBB'

    def test_display_multiple_values(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                hello = \'hello\'\n                world = \'world\'\n                display(hello, world)\n            </script>\n            ')
        output = self.page.locator('script-py')
        assert output.inner_text() == 'hello\nworld'

    def test_display_multiple_append_false(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display(\'hello\', append=False)\n                display(\'world\', append=False)\n            </script>\n        ')
        output = self.page.locator('script-py')
        assert output.inner_text() == 'world'

    def test_display_multiple_append_false_with_target(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <div id="circle-div"></div>\n            <script type="py">\n                from pyscript import display\n                class Circle:\n                    r = 0\n                    def _repr_svg_(self):\n                        return (\n                            f\'<svg height="{self.r*2}" width="{self.r*2}">\'\n                            f\'<circle cx="{self.r}" cy="{self.r}" r="{self.r}" fill="red" /></svg>\'\n                        )\n\n                circle = Circle()\n\n                circle.r += 5\n                # display(circle, target="circle-div", append=False)\n                circle.r += 5\n                display(circle, target="circle-div", append=False)\n            </script>\n        ')
        innerhtml = self.page.locator('id=circle-div').inner_html()
        assert innerhtml == '<svg height="20" width="20"><circle cx="10" cy="10" r="10" fill="red"></circle></svg>'
        assert self.console.error.lines == []

    def test_display_list_dict_tuple(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                l = [\'A\', 1, \'!\']\n                d = {\'B\': 2, \'List\': l}\n                t = (\'C\', 3, \'!\')\n                display(l, d, t)\n            </script>\n            ')
        inner_text = self.page.inner_text('html')
        filtered_inner_text = filter_inner_text(inner_text)
        print(filtered_inner_text)
        assert filtered_inner_text == "['A', 1, '!']\n{'B': 2, 'List': ['A', 1, '!']}\n('C', 3, '!')"

    def test_display_should_escape(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                display("<p>hello world</p>")\n            </script>\n            ')
        out = self.page.locator('script-py > div')
        assert out.inner_html() == html.escape('<p>hello world</p>')
        assert out.inner_text() == '<p>hello world</p>'

    def test_display_HTML(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display, HTML\n                display(HTML("<p>hello world</p>"))\n            </script>\n            ')
        out = self.page.locator('script-py > div')
        assert out.inner_html() == '<p>hello world</p>'
        assert out.inner_text() == 'hello world'

    @skip_worker('NEXT: matplotlib-pyodide backend does not work')
    def test_image_display(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n                <py-config> packages = ["matplotlib"] </py-config>\n                <script type="py">\n                    from pyscript import display\n                    import matplotlib.pyplot as plt\n                    xpoints = [3, 6, 9]\n                    ypoints = [1, 2, 3]\n                    plt.plot(xpoints, ypoints)\n                    display(plt)\n                </script>\n            ', timeout=30 * 1000)
        wait_for_render(self.page, '*', '<img src=[\'"]data:image')
        test = self.page.wait_for_selector('img')
        img_src = test.get_attribute('src').replace('data:image/png;charset=utf-8;base64,', '')
        img_data = np.asarray(Image.open(io.BytesIO(base64.b64decode(img_src))))
        with Image.open(os.path.join(os.path.dirname(__file__), 'test_assets', 'line_plot.png')) as image:
            ref_data = np.asarray(image)
        deviation = np.mean(np.abs(img_data - ref_data))
        assert deviation == 0.0
        self.assert_no_banners()

    def test_empty_HTML_and_console_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                import js\n                print(\'print from python\')\n                js.console.log(\'print from js\')\n                js.console.error(\'error from js\');\n            </script>\n        ')
        inner_html = self.page.content()
        assert re.search('', inner_html)
        console_text = self.console.all.lines
        assert 'print from python' in console_text
        assert 'print from js' in console_text
        assert 'error from js' in console_text

    def test_text_HTML_and_console_output(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n            <script type="py">\n                from pyscript import display\n                import js\n                display(\'this goes to the DOM\')\n                print(\'print from python\')\n                js.console.log(\'print from js\')\n                js.console.error(\'error from js\');\n            </script>\n        ')
        inner_text = self.page.inner_text('script-py')
        assert inner_text == 'this goes to the DOM'
        assert self.console.log.lines[-2:] == ['print from python', 'print from js']
        print(self.console.error.lines)
        assert self.console.error.lines[-1] == 'error from js'

    def test_console_line_break(self):
        if False:
            return 10
        self.pyscript_run('\n            <script type="py">\n            print(\'1print\\n2print\')\n            print(\'1console\\n2console\')\n            </script>\n        ')
        console_text = self.console.all.lines
        assert console_text.index('1print') == console_text.index('2print') - 1
        assert console_text.index('1console') == console_text.index('2console') - 1

    @skip_worker('NEXT: display target does not work properly')
    def test_image_renders_correctly(self):
        if False:
            print('Hello World!')
        '\n        This is just a sanity check to make sure that images are rendered\n        in a reasonable way.\n        '
        self.pyscript_run('\n            <py-config>\n                packages = ["pillow"]\n            </py-config>\n\n            <div id="img-target" />\n            <script type="py">\n                from pyscript import display\n                from PIL import Image\n                img = Image.new("RGB", (4, 4), color=(0, 0, 0))\n                display(img, target=\'img-target\', append=False)\n            </script>\n            ')
        img_src = self.page.locator('img').get_attribute('src')
        assert img_src.startswith('data:image/png;charset=utf-8;base64')