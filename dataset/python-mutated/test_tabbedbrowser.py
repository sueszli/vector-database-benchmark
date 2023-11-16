import pytest
from qutebrowser.mainwindow import tabbedbrowser

class TestTabDeque:

    @pytest.mark.parametrize('size', [-1, 5])
    def test_size_handling(self, size, config_stub):
        if False:
            while True:
                i = 10
        config_stub.val.tabs.focus_stack_size = size
        dq = tabbedbrowser.TabDeque()
        dq.update_size()