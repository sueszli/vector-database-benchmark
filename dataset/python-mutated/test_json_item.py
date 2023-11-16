from __future__ import annotations
import pytest
pytest
import json
from jinja2 import Template
from selenium.webdriver.common.by import By
from bokeh.embed import json_item
from bokeh.models import Plot
from bokeh.resources import INLINE
pytest_plugins = ('tests.support.plugins.project', 'tests.support.plugins.selenium')
PAGE = Template('\n<!DOCTYPE html>\n<html lang="en">\n<head>\n  {{ resources }}\n</head>\n\n<body>\n  <div id="_target"></div>\n  <script>\n    Bokeh.embed.embed_item({{ item }}, "_target");\n  </script>\n</body>\n')

@pytest.mark.selenium
class Test_json_item:

    def test_bkroot_added_to_target(self, driver, test_file_path_and_url, has_no_console_errors) -> None:
        if False:
            i = 10
            return i + 15
        p = Plot(css_classes=['this-plot'])
        html = PAGE.render(item=json.dumps(json_item(p)), resources=INLINE.render())
        (path, url) = test_file_path_and_url
        with open(path, 'w') as f:
            f.write(html)
        driver.get(url)
        div = driver.find_elements(By.CLASS_NAME, 'this-plot')
        assert has_no_console_errors(driver)
        assert len(div) == 1