import pytest
from .support import PyScriptTest, skip_worker

class TestEventHandler(PyScriptTest):

    def test_when_decorator_with_event(self):
        if False:
            for i in range(10):
                print('nop')
        'When the decorated function takes a single parameter,\n        it should be passed the event object\n        '
        self.pyscript_run('\n            <button id="foo_id">foo_button</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector="#foo_id")\n                def foo(evt):\n                    print(f"clicked {evt.target.id}")\n            </script>\n        ')
        self.page.locator('text=foo_button').click()
        self.wait_for_console('clicked foo_id')
        self.assert_no_banners()

    def test_when_decorator_without_event(self):
        if False:
            for i in range(10):
                print('nop')
        "When the decorated function takes no parameters (not including 'self'),\n        it should be called without the event object\n        "
        self.pyscript_run('\n            <button id="foo_id">foo_button</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector="#foo_id")\n                def foo():\n                    print("The button was clicked")\n            </script>\n        ')
        self.page.locator('text=foo_button').click()
        self.wait_for_console('The button was clicked')
        self.assert_no_banners()

    def test_multiple_when_decorators_with_event(self):
        if False:
            print('Hello World!')
        self.pyscript_run('\n            <button id="foo_id">foo_button</button>\n            <button id="bar_id">bar_button</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector="#foo_id")\n                def foo_click(evt):\n                    print(f"foo_click! id={evt.target.id}")\n                @when("click", selector="#bar_id")\n                def bar_click(evt):\n                    print(f"bar_click! id={evt.target.id}")\n            </script>\n        ')
        self.page.locator('text=foo_button').click()
        self.wait_for_console('foo_click! id=foo_id')
        self.page.locator('text=bar_button').click()
        self.wait_for_console('bar_click! id=bar_id')
        self.assert_no_banners()

    def test_two_when_decorators(self):
        if False:
            return 10
        'When decorating a function twice, both should function'
        self.pyscript_run('\n            <button id="foo_id">foo_button</button>\n            <button class="bar_class">bar_button</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector="#foo_id")\n                @when("mouseover", selector=".bar_class")\n                def foo(evt):\n                    print(f"got event: {evt.type}")\n            </script>\n        ')
        self.page.locator('text=bar_button').hover()
        self.wait_for_console('got event: mouseover')
        self.page.locator('text=foo_button').click()
        self.wait_for_console('got event: click')
        self.assert_no_banners()

    def test_two_when_decorators_same_element(self):
        if False:
            while True:
                i = 10
        'When decorating a function twice *on the same DOM element*, both should function'
        self.pyscript_run('\n            <button id="foo_id">foo_button</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector="#foo_id")\n                @when("mouseover", selector="#foo_id")\n                def foo(evt):\n                    print(f"got event: {evt.type}")\n            </script>\n        ')
        self.page.locator('text=foo_button').hover()
        self.wait_for_console('got event: mouseover')
        self.page.locator('text=foo_button').click()
        self.wait_for_console('got event: click')
        self.assert_no_banners()

    def test_when_decorator_multiple_elements(self):
        if False:
            while True:
                i = 10
        "The @when decorator's selector should successfully select multiple\n        DOM elements\n        "
        self.pyscript_run('\n            <button class="bar_class">button1</button>\n            <button class="bar_class">button2</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector=".bar_class")\n                def foo(evt):\n                    print(f"{evt.target.innerText} was clicked")\n            </script>\n        ')
        self.page.locator('text=button1').click()
        self.page.locator('text=button2').click()
        self.wait_for_console('button2 was clicked')
        assert 'button1 was clicked' in self.console.log.lines
        assert 'button2 was clicked' in self.console.log.lines
        self.assert_no_banners()

    def test_when_decorator_duplicate_selectors(self):
        if False:
            i = 10
            return i + 15
        ' '
        self.pyscript_run('\n            <button id="foo_id">foo_button</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector="#foo_id")\n                @when("click", selector="#foo_id")\n                def foo(evt):\n                    foo.n += 1\n                    print(f"click {foo.n} on {evt.target.id}")\n                foo.n = 0\n            </script>\n        ')
        self.page.locator('text=foo_button').click()
        self.wait_for_console('click 1 on foo_id')
        self.wait_for_console('click 2 on foo_id')
        self.assert_no_banners()

    @skip_worker('NEXT: error banner not shown')
    def test_when_decorator_invalid_selector(self):
        if False:
            return 10
        'When the selector parameter of @when is invalid, it should show an error'
        self.pyscript_run('\n            <button id="foo_id">foo_button</button>\n            <script type="py">\n                from pyscript import when\n                @when("click", selector="#.bad")\n                def foo(evt):\n                    ...\n            </script>\n        ')
        self.page.locator('text=foo_button').click()
        msg = "Failed to execute 'querySelectorAll' on 'Document': '#.bad' is not a valid selector."
        error = self.page.wait_for_selector('.py-error')
        banner_text = error.inner_text()
        if msg not in banner_text:
            raise AssertionError(f"Expected message '{msg}' does not match banner text '{banner_text}'")
        assert msg in self.console.error.lines[-1]
        self.check_py_errors(msg)