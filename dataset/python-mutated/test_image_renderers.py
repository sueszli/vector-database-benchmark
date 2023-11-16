import base64
import sys
import json
import pytest
import numpy as np
from plotly import io as pio
import plotly.graph_objs as go
from plotly.offline.offline import _get_jconfig
if sys.version_info >= (3, 3):
    import unittest.mock as mock
else:
    import mock
plotly_mimetype = 'application/vnd.plotly.v1+json'

@pytest.fixture
def fig1(request):
    if False:
        return 10
    return go.Figure(data=[{'type': 'scatter', 'marker': {'color': 'green'}, 'y': np.array([2, 1, 3, 2, 4, 2])}], layout={'title': {'text': 'Figure title'}})

def test_png_renderer_mimetype(fig1):
    if False:
        i = 10
        return i + 15
    pio.renderers.default = 'png'
    pio.renderers['png'].width = 400
    pio.renderers['png'].height = 500
    pio.renderers['png'].scale = 1
    image_bytes = pio.to_image(fig1, width=400, height=500, scale=1)
    image_str = base64.b64encode(image_bytes).decode('utf8')
    expected = {'image/png': image_str}
    pio.renderers.render_on_display = False
    with mock.patch('IPython.display.display') as mock_display:
        fig1._ipython_display_()
    mock_display.assert_not_called()
    pio.renderers.render_on_display = True
    with mock.patch('IPython.display.display') as mock_display:
        fig1._ipython_display_()
    mock_display.assert_called_once_with(expected, raw=True)

def test_svg_renderer_show(fig1):
    if False:
        while True:
            i = 10
    pio.renderers.default = 'svg'
    pio.renderers['svg'].width = 400
    pio.renderers['svg'].height = 500
    pio.renderers['svg'].scale = 1
    with mock.patch('IPython.display.display') as mock_display:
        pio.show(fig1)
    mock_call_args = mock_display.call_args
    mock_arg1 = mock_call_args[0][0]
    assert list(mock_arg1) == ['image/svg+xml']
    assert mock_arg1['image/svg+xml'].startswith('<svg class="main-svg" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="400" height="500"')
    mock_kwargs = mock_call_args[1]
    assert mock_kwargs == {'raw': True}

def test_pdf_renderer_show_override(fig1):
    if False:
        while True:
            i = 10
    pio.renderers.default = None
    pio.renderers['png'].width = 400
    pio.renderers['png'].height = 500
    pio.renderers['png'].scale = 1
    image_bytes_png = pio.to_image(fig1, format='png', width=400, height=500, scale=1)
    image_str_png = base64.b64encode(image_bytes_png).decode('utf8')
    with mock.patch('IPython.display.display') as mock_display:
        pio.show(fig1, renderer='png')
    expected_bundle = {'image/png': image_str_png}
    mock_display.assert_called_once_with(expected_bundle, raw=True)

def test_mimetype_combination(fig1):
    if False:
        print('Hello World!')
    pio.renderers.default = 'png+jupyterlab'
    pio.renderers['png'].width = 400
    pio.renderers['png'].height = 500
    pio.renderers['png'].scale = 1
    image_bytes = pio.to_image(fig1, format='png', width=400, height=500, scale=1)
    image_str = base64.b64encode(image_bytes).decode('utf8')
    plotly_mimetype_dict = json.loads(pio.to_json(fig1, remove_uids=False))
    plotly_mimetype_dict['config'] = {'plotlyServerURL': _get_jconfig()['plotlyServerURL']}
    expected = {'image/png': image_str, plotly_mimetype: plotly_mimetype_dict}
    pio.renderers.render_on_display = False
    with mock.patch('IPython.display.display') as mock_display:
        fig1._ipython_display_()
    mock_display.assert_not_called()
    pio.renderers.render_on_display = True
    with mock.patch('IPython.display.display') as mock_display:
        fig1._ipython_display_()
    mock_display.assert_called_once_with(expected, raw=True)