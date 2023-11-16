import pytest
from sacred.host_info import get_host_info, host_info_getter, host_info_gatherers

def test_get_host_info(monkeypatch: pytest.MonkeyPatch):
    if False:
        for i in range(10):
            print('nop')
    with monkeypatch.context() as cntx:
        cntx.setattr('sacred.settings.SETTINGS.HOST_INFO.INCLUDE_CPU_INFO', True)
        host_info = get_host_info()
    assert isinstance(host_info['hostname'], str)
    assert isinstance(host_info['cpu'], str)
    assert host_info['cpu'] != 'Unknown'
    assert isinstance(host_info['os'], (tuple, list))
    assert isinstance(host_info['python_version'], str)

def test_host_info_decorator():
    if False:
        print('Hello World!')
    try:
        assert 'greeting' not in host_info_gatherers

        @host_info_getter
        def greeting():
            if False:
                print('Hello World!')
            return 'hello'
        assert 'greeting' in host_info_gatherers
        assert host_info_gatherers['greeting'] == greeting
        assert get_host_info()['greeting'] == 'hello'
    finally:
        del host_info_gatherers['greeting']

def test_host_info_decorator_with_name():
    if False:
        for i in range(10):
            print('nop')
    try:
        assert 'foo' not in host_info_gatherers

        @host_info_getter(name='foo')
        def greeting():
            if False:
                print('Hello World!')
            return 'hello'
        assert 'foo' in host_info_gatherers
        assert 'greeting' not in host_info_gatherers
        assert host_info_gatherers['foo'] == greeting
        assert get_host_info()['foo'] == 'hello'
    finally:
        del host_info_gatherers['foo']

def test_host_info_decorator_depreciation_warning():
    if False:
        i = 10
        return i + 15
    try:
        assert 'foo' not in host_info_gatherers
        with pytest.warns(DeprecationWarning):

            @host_info_getter(name='foo')
            def greeting():
                if False:
                    for i in range(10):
                        print('nop')
                return 'hello'
    finally:
        del host_info_gatherers['foo']