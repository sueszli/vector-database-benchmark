import pytest
from ulauncher.modes.calc.CalcResult import CalcResult

class TestCalcResult:

    @pytest.fixture
    def CopyToClipboardAction(self, mocker):
        if False:
            while True:
                i = 10
        return mocker.patch('ulauncher.modes.calc.CalcResult.CopyToClipboardAction')

    def test_get_name(self):
        if False:
            i = 10
            return i + 15
        assert CalcResult(52).name == '52'
        assert CalcResult('42').name == '42'
        assert CalcResult(error='message').name == 'Error!'

    def test_get_description(self):
        if False:
            for i in range(10):
                print('nop')
        assert CalcResult(52).description == 'Enter to copy to the clipboard'
        assert CalcResult(error='message').get_description('q') == 'message'

    def test_on_activation(self, CopyToClipboardAction):
        if False:
            i = 10
            return i + 15
        item = CalcResult(52)
        assert item.on_activation('q') == CopyToClipboardAction.return_value
        CopyToClipboardAction.assert_called_with('52')

    def test_on_activation__error__returns_true(self, CopyToClipboardAction):
        if False:
            return 10
        item = CalcResult(error='message')
        assert item.on_activation('q') is True
        assert not CopyToClipboardAction.called