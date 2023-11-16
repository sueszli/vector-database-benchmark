import plotly
import os
import shutil
import pytest

@pytest.fixture()
def setup():
    if False:
        return 10
    plotly.io.orca.config.restore_defaults(reset_server=False)
here = os.path.dirname(os.path.abspath(__file__))
pytestmark = pytest.mark.usefixtures('setup')

def execute_plotly_example():
    if False:
        while True:
            i = 10
    '\n    Some typical code which would go inside a gallery example.\n    '
    import plotly.graph_objs as go
    import numpy as np
    N = 200
    random_x = np.random.randn(N)
    random_y_0 = np.random.randn(N)
    random_y_1 = np.random.randn(N) - 1
    trace_0 = go.Scatter(x=random_x, y=random_y_0, mode='markers', name='Above')
    fig = go.Figure(data=[trace_0])
    plotly.io.show(fig)

def test_scraper():
    if False:
        while True:
            i = 10
    from plotly.io._sg_scraper import plotly_sg_scraper
    assert plotly.io.renderers.default == 'sphinx_gallery_png'
    block = ''
    import tempfile
    tempdir = tempfile.mkdtemp()
    gallery_conf = {'src_dir': tempdir, 'examples_dirs': here}
    names = iter(['0', '1', '2'])
    block_vars = {'image_path_iterator': names, 'src_file': os.path.join(here, 'plot_example.py')}
    execute_plotly_example()
    res = plotly_sg_scraper(block, block_vars, gallery_conf)
    shutil.rmtree(tempdir)
    assert '.. raw:: html' in res