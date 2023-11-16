import pytest
from salt.runners import manage

def test_deprecation_58638():
    if False:
        i = 10
        return i + 15
    pytest.raises(TypeError, manage.list_state, show_ipv4='data')
    try:
        manage.list_state(show_ipv4='data')
    except TypeError as no_show_ipv4:
        assert str(no_show_ipv4) == "list_state() got an unexpected keyword argument 'show_ipv4'"