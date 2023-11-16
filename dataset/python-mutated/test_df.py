import sys
from importlib import reload
from types import ModuleType
import pytest
from libqtile.widget import df
from test.widgets.conftest import FakeBar

class FakeOS(ModuleType):

    class statvfs:

        def __init__(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            pass

        @property
        def f_frsize(self):
            if False:
                for i in range(10):
                    print('nop')
            return 4096

        @property
        def f_blocks(self):
            if False:
                return 10
            return 60000000

        @property
        def f_bfree(self):
            if False:
                while True:
                    i = 10
            return 15000000

        @property
        def f_bavail(self):
            if False:
                while True:
                    i = 10
            return 10000000

@pytest.fixture()
def patched_df(monkeypatch):
    if False:
        while True:
            i = 10
    monkeypatch.setitem(sys.modules, 'os', FakeOS('os'))
    reload(df)

@pytest.mark.usefixtures('patched_df')
def test_df_no_warning(fake_qtile, fake_window):
    if False:
        print('Hello World!')
    'Test no text when free space over threshold'
    df1 = df.DF()
    fakebar = FakeBar([df1], window=fake_window)
    df1._configure(fake_qtile, fakebar)
    text = df1.poll()
    assert text == ''
    df1.draw()
    assert df1.layout.colour == df1.foreground

@pytest.mark.usefixtures('patched_df')
def test_df_always_visible(fake_qtile, fake_window):
    if False:
        i = 10
        return i + 15
    'Test text is always displayed'
    df2 = df.DF(visible_on_warn=False)
    fakebar = FakeBar([df2], window=fake_window)
    df2._configure(fake_qtile, fakebar)
    text = df2.poll()
    assert text == '/ (38G|83%)'
    df2.draw()
    assert df2.layout.colour == df2.foreground

@pytest.mark.usefixtures('patched_df')
def test_df_warn_space(fake_qtile, fake_window):
    if False:
        print('Hello World!')
    '\n    Test text is visible and colour changes when space\n    below threshold\n    '
    df3 = df.DF(warn_space=40)
    fakebar = FakeBar([df3], window=fake_window)
    df3._configure(fake_qtile, fakebar)
    text = df3.poll()
    assert text == '/ (38G|83%)'
    df3.draw()
    assert df3.layout.colour == df3.warn_color