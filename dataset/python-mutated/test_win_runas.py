import pytest
from salt.utils import win_runas

@pytest.mark.parametrize('input_value, expected', [('test_user', ('test_user', '.')), ('domain\\test_user', ('test_user', 'domain')), ('domain.com\\test_user', ('test_user', 'domain.com')), ('test_user@domain', ('test_user', 'domain')), ('test_user@domain.com', ('test_user', 'domain.com'))])
def test_split_username(input_value, expected):
    if False:
        while True:
            i = 10
    '\n    Test that the username is parsed properly from various domain/username\n    combinations\n    '
    result = win_runas.split_username(input_value)
    assert result == expected