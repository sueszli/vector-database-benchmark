from __future__ import annotations
from ansible.utils.display import Display
from unittest.mock import MagicMock

def test_display_with_fake_cowsay_binary(capsys, mocker):
    if False:
        while True:
            i = 10
    display = Display()
    mocker.patch('ansible.constants.ANSIBLE_COW_PATH', './cowsay.sh')
    mock_popen = MagicMock()
    mock_popen.return_value.returncode = 1
    mocker.patch('subprocess.Popen', mock_popen)
    assert not hasattr(display, 'cows_available')
    assert display.b_cowsay is None