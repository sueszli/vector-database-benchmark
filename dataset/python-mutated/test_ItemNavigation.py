from unittest import mock
import pytest
from ulauncher.ui.ItemNavigation import ItemNavigation

class TestItemNavigation:

    @pytest.fixture
    def items(self):
        if False:
            while True:
                i = 10
        return [mock.MagicMock() for _ in range(5)]

    @pytest.fixture
    def nav(self, items):
        if False:
            print('Hello World!')
        return ItemNavigation(items)

    @pytest.fixture(autouse=True)
    def query_history(self, mocker):
        if False:
            i = 10
            return i + 15
        return mocker.patch('ulauncher.ui.ItemNavigation.query_history')

    def test_select_is_called(self, nav, items):
        if False:
            i = 10
            return i + 15
        nav.select(1)
        assert nav.index == 1
        items[1].select.assert_called_once_with()

    def test_select_and_deselect_is_called(self, nav, items):
        if False:
            return 10
        nav.select(1)
        nav.select(5)
        items[1].deselect.assert_called_once_with()
        items[0].select.assert_called_once_with()
        assert nav.index == 0, 'First element is not selected'

    def test_go_up_from_start(self, nav, items):
        if False:
            print('Hello World!')
        nav.go_up()
        items[4].select.assert_called_once_with()

    def test_go_up_from_1st(self, nav, items):
        if False:
            for i in range(10):
                print('nop')
        nav.select(1)
        nav.go_up()
        items[0].select.assert_called_once_with()

    def test_go_up_from_last(self, nav, items):
        if False:
            i = 10
            return i + 15
        nav.select(4)
        nav.go_up()
        items[3].select.assert_called_once_with()

    def test_go_down_from_second(self, nav, items):
        if False:
            while True:
                i = 10
        nav.select(2)
        nav.go_down()
        items[3].select.assert_called_once_with()

    def test_go_down_from_last(self, nav, items):
        if False:
            while True:
                i = 10
        nav.select(4)
        nav.go_down()
        items[0].select.assert_called_once_with()

    def test_enter_no_index(self, nav, items):
        if False:
            print('Hello World!')
        nav.select(2)
        selected_result = items[2].result
        assert nav.activate('test') is selected_result.on_activation.return_value

    def test_enter__alternative(self, nav, items):
        if False:
            print('Hello World!')
        nav.select(2)
        selected_result = items[2].result
        nav.activate('test', True)
        selected_result.on_activation.assert_called_with('test', True)