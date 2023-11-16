import os
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
try:
    from matplotlib.backends import _macosx
except ImportError:
    pytest.skip('These are mac only tests', allow_module_level=True)

@pytest.mark.backend('macosx')
def test_cached_renderer():
    if False:
        i = 10
        return i + 15
    fig = plt.figure(1)
    fig.canvas.draw()
    assert fig.canvas.get_renderer()._renderer is not None
    fig = plt.figure(2)
    fig.draw_without_rendering()
    assert fig.canvas.get_renderer()._renderer is not None

@pytest.mark.backend('macosx')
def test_savefig_rcparam(monkeypatch, tmp_path):
    if False:
        return 10

    def new_choose_save_file(title, directory, filename):
        if False:
            return 10
        assert directory == str(tmp_path)
        os.makedirs(f'{directory}/test')
        return f'{directory}/test/{filename}'
    monkeypatch.setattr(_macosx, 'choose_save_file', new_choose_save_file)
    fig = plt.figure()
    with mpl.rc_context({'savefig.directory': tmp_path}):
        fig.canvas.toolbar.save_figure()
        save_file = f'{tmp_path}/test/{fig.canvas.get_default_filename()}'
        assert os.path.exists(save_file)
        assert mpl.rcParams['savefig.directory'] == f'{tmp_path}/test'