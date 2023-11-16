from __future__ import annotations
import pytest
pytest
from subprocess import run
from sys import executable as python
from tests.support.util.project import ls_modules, verify_clean_imports
SELENIUM_ALLOWED = ('tests.support', 'bokeh.io.webdriver')
MODULES = ls_modules(skip_prefixes=SELENIUM_ALLOWED)

def test_no_selenium_common_combined() -> None:
    if False:
        while True:
            i = 10
    ' Basic usage of Bokeh should not result in any Selenium code being\n    imported. This test ensures that importing basic modules does not bring in\n    Tornado.\n\n    '
    proc = run([python, '-c', verify_clean_imports('selenium', MODULES)])
    assert proc.returncode == 0, 'Selenium imported in common modules'