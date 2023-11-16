import pytest
__all__ = ['cog_mgr', 'default_dir']

@pytest.fixture()
def cog_mgr(red):
    if False:
        for i in range(10):
            print('nop')
    return red._cog_mgr

@pytest.fixture()
def default_dir(red):
    if False:
        print('Hello World!')
    return red._main_dir