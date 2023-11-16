from __future__ import annotations
import pytest
pytest
from html import escape
from bokeh.models import PreText
from tests.support.plugins.project import BokehModelPage
from tests.support.util.selenium import find_element_for
pytest_plugins = ('tests.support.plugins.project',)
text = '\nYour <a href="https://en.wikipedia.org/wiki/HTML">HTML</a>-supported text is initialized with the <b>text</b> argument.  The\nremaining div arguments are <b>width</b> and <b>height</b>. For this example, those values\nare <i>200</i> and <i>100</i> respectively.'

@pytest.mark.selenium
class Test_PreText:

    def test_displays_div_as_text(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            print('Hello World!')
        para = PreText(text=text)
        page = bokeh_model_page(para)
        el = find_element_for(page.driver, para, 'div pre')
        assert el.get_attribute('innerHTML') == escape(text, quote=None)
        assert page.has_no_console_errors()

    def test_set_styles(self, bokeh_model_page: BokehModelPage) -> None:
        if False:
            return 10
        para = PreText(text=text, styles={'font-size': '26px'})
        page = bokeh_model_page(para)
        el = find_element_for(page.driver, para)
        assert 'font-size: 26px;' in el.get_attribute('style')
        assert page.has_no_console_errors()