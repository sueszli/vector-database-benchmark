"""Unit tests for the charts middleware class"""
from pandasai.middlewares import ChartsMiddleware
from unittest.mock import Mock
import pytest

class TestChartsMiddleware:
    """Unit tests for the charts middleware class"""

    @pytest.fixture
    def middleware(self):
        if False:
            while True:
                i = 10
        return ChartsMiddleware()

    def test_add_close_all(self, middleware):
        if False:
            print('Hello World!')
        "Test adding plt.close('all') to the code"
        code = 'plt.show()'
        assert middleware(code=code) == "plt.show(block=False)\nplt.close('all')"

    def test_add_close_all_if_in_console(self, middleware):
        if False:
            while True:
                i = 10
        '\n        Test should not add block=False if running in console\n        '
        middleware._is_running_in_console = Mock(return_value=True)
        code = 'plt.show()'
        assert middleware(code=code) == "plt.show()\nplt.close('all')"

    def test_not_add_close_all_if_already_there(self, middleware):
        if False:
            print('Hello World!')
        "Test that plt.close('all') is not added if it is already there"
        code = "plt.show()\nplt.close('all')"
        assert middleware(code=code) == "plt.show(block=False)\nplt.close('all')"

    def test_no_add_close_all_if_not_show(self, middleware):
        if False:
            return 10
        "Test that plt.close('all') is not added if plt.show() is not there"
        code = 'plt.plot()'
        assert middleware(code=code) == 'plt.plot()'