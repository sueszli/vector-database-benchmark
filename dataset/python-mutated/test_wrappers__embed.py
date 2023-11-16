from __future__ import annotations
import pytest
pytest
import bokeh.embed.wrappers as bew

class Test_wrap_in_onload:

    def test_render(self) -> None:
        if False:
            i = 10
            return i + 15
        assert bew.wrap_in_onload('code\nmorecode') == '(function() {\n  const fn = function() {\n    code\n    morecode\n  };\n  if (document.readyState != "loading") fn();\n  else document.addEventListener("DOMContentLoaded", fn);\n})();'

class Test_wrap_in_safely:

    def test_render(self) -> None:
        if False:
            print('Hello World!')
        assert bew.wrap_in_safely('code\nmorecode') == 'Bokeh.safely(function() {\n  code\n  morecode\n});'

class Test_wrap_in_script_tag:

    def test_render(self) -> None:
        if False:
            i = 10
            return i + 15
        assert bew.wrap_in_script_tag('code\nmorecode') == '\n<script type="text/javascript">\n  code\n  morecode\n</script>'

def test__ONLOAD() -> None:
    if False:
        i = 10
        return i + 15
    assert bew._ONLOAD == '(function() {\n  const fn = function() {\n%(code)s\n  };\n  if (document.readyState != "loading") fn();\n  else document.addEventListener("DOMContentLoaded", fn);\n})();'

def test__SAFELY() -> None:
    if False:
        return 10
    assert bew._SAFELY == 'Bokeh.safely(function() {\n%(code)s\n});'