import plotly.graph_objs as go
import plotly.io as pio
import pytest
import plotly
import json
import os
import tempfile
from unittest.mock import MagicMock
from pathlib import Path

@pytest.fixture
def fig1(request):
    if False:
        print('Hello World!')
    return go.Figure(data=[{'type': 'scattergl', 'marker': {'color': 'green'}}, {'type': 'parcoords', 'dimensions': [{'values': [1, 2, 3]}, {'values': [3, 2, 1]}], 'line': {'color': 'blue'}}], layout={'title': 'Figure title'})
opts = {'separators': (',', ':'), 'cls': plotly.utils.PlotlyJSONEncoder}
pretty_opts = {'indent': 2, 'cls': plotly.utils.PlotlyJSONEncoder}

def test_to_json(fig1):
    if False:
        for i in range(10):
            print('nop')
    assert pio.to_json(fig1, remove_uids=False) == json.dumps(fig1, **opts)

def test_to_json_remove_uids(fig1):
    if False:
        print('Hello World!')
    dict1 = fig1.to_dict()
    for trace in dict1['data']:
        trace.pop('uid', None)
    assert pio.to_json(fig1) == json.dumps(dict1, **opts)

def test_to_json_validate(fig1):
    if False:
        for i in range(10):
            print('nop')
    dict1 = fig1.to_dict()
    dict1['layout']['bogus'] = 37
    with pytest.raises(ValueError):
        pio.to_json(dict1)

def test_to_json_validate_false(fig1):
    if False:
        i = 10
        return i + 15
    dict1 = fig1.to_dict()
    dict1['layout']['bogus'] = 37
    assert pio.to_json(dict1, validate=False) == json.dumps(dict1, **opts)

def test_to_json_pretty_print(fig1):
    if False:
        i = 10
        return i + 15
    assert pio.to_json(fig1, remove_uids=False, pretty=True) == json.dumps(fig1, **pretty_opts)

def test_from_json(fig1):
    if False:
        i = 10
        return i + 15
    fig1_json = json.dumps(fig1, **opts)
    fig1_loaded = pio.from_json(fig1_json)
    assert isinstance(fig1_loaded, go.Figure)
    assert pio.to_json(fig1_loaded) == pio.to_json(fig1.to_dict())

@pytest.mark.parametrize('fig_type_spec,fig_type', [('Figure', go.Figure), (go.Figure, go.Figure), ('FigureWidget', go.FigureWidget), (go.FigureWidget, go.FigureWidget)])
def test_from_json_output_type(fig1, fig_type_spec, fig_type):
    if False:
        i = 10
        return i + 15
    fig1_json = json.dumps(fig1, **opts)
    fig1_loaded = pio.from_json(fig1_json, output_type=fig_type_spec)
    assert isinstance(fig1_loaded, fig_type)
    assert pio.to_json(fig1_loaded) == pio.to_json(fig1.to_dict())

def test_from_json_invalid(fig1):
    if False:
        for i in range(10):
            print('nop')
    dict1 = fig1.to_dict()
    dict1['data'][0]['marker']['bogus'] = 123
    dict1['data'][0]['marker']['size'] = -1
    bad_json = json.dumps(dict1, **opts)
    with pytest.raises(ValueError):
        pio.from_json(bad_json)

def test_from_json_skip_invalid(fig1):
    if False:
        for i in range(10):
            print('nop')
    dict1 = fig1.to_dict()
    dict1['data'][0]['marker']['bogus'] = 123
    dict1['data'][0]['marker']['size'] = -1
    bad_json = json.dumps(dict1, **opts)
    fig1_loaded = pio.from_json(bad_json, skip_invalid=True)
    assert pio.to_json(fig1_loaded) == pio.to_json(fig1.to_dict())

@pytest.mark.parametrize('fig_type_spec,fig_type', [('Figure', go.Figure), (go.Figure, go.Figure), ('FigureWidget', go.FigureWidget), (go.FigureWidget, go.FigureWidget)])
def test_read_json_from_filelike(fig1, fig_type_spec, fig_type):
    if False:
        while True:
            i = 10
    filemock = MagicMock()
    del filemock.read_text
    filemock.read.return_value = pio.to_json(fig1)
    fig1_loaded = pio.read_json(filemock, output_type=fig_type_spec)
    assert isinstance(fig1_loaded, fig_type)
    assert pio.to_json(fig1_loaded) == pio.to_json(fig1.to_dict())

@pytest.mark.parametrize('fig_type_spec,fig_type', [('Figure', go.Figure), (go.Figure, go.Figure), ('FigureWidget', go.FigureWidget), (go.FigureWidget, go.FigureWidget)])
def test_read_json_from_pathlib(fig1, fig_type_spec, fig_type):
    if False:
        print('Hello World!')
    filemock = MagicMock(spec=Path)
    filemock.read_text.return_value = pio.to_json(fig1)
    fig1_loaded = pio.read_json(filemock, output_type=fig_type_spec)
    assert isinstance(fig1_loaded, fig_type)
    assert pio.to_json(fig1_loaded) == pio.to_json(fig1.to_dict())

@pytest.mark.parametrize('fig_type_spec,fig_type', [('Figure', go.Figure), (go.Figure, go.Figure), ('FigureWidget', go.FigureWidget), (go.FigureWidget, go.FigureWidget)])
def test_read_json_from_file_string(fig1, fig_type_spec, fig_type):
    if False:
        while True:
            i = 10
    with tempfile.TemporaryDirectory() as dir_name:
        path = os.path.join(dir_name, 'fig1.json')
        with open(path, 'w') as f:
            f.write(pio.to_json(fig1))
        fig1_loaded = pio.read_json(path, output_type=fig_type_spec)
        assert isinstance(fig1_loaded, fig_type)
        assert pio.to_json(fig1_loaded) == pio.to_json(fig1.to_dict())

@pytest.mark.parametrize('pretty', [True, False])
@pytest.mark.parametrize('remove_uids', [True, False])
def test_write_json_filelike(fig1, pretty, remove_uids):
    if False:
        for i in range(10):
            print('nop')
    filemock = MagicMock()
    del filemock.write_text
    pio.write_json(fig1, filemock, pretty=pretty, remove_uids=remove_uids)
    expected = pio.to_json(fig1, pretty=pretty, remove_uids=remove_uids)
    filemock.write.assert_called_once_with(expected)

@pytest.mark.parametrize('pretty', [True, False])
@pytest.mark.parametrize('remove_uids', [True, False])
def test_write_json_pathlib(fig1, pretty, remove_uids):
    if False:
        for i in range(10):
            print('nop')
    filemock = MagicMock(spec=Path)
    pio.write_json(fig1, filemock, pretty=pretty, remove_uids=remove_uids)
    expected = pio.to_json(fig1, pretty=pretty, remove_uids=remove_uids)
    filemock.write_text.assert_called_once_with(expected)

@pytest.mark.parametrize('pretty', [True, False])
@pytest.mark.parametrize('remove_uids', [True, False])
def test_write_json_from_file_string(fig1, pretty, remove_uids):
    if False:
        return 10
    with tempfile.TemporaryDirectory() as dir_name:
        path = os.path.join(dir_name, 'fig1.json')
        pio.write_json(fig1, path, pretty=pretty, remove_uids=remove_uids)
        with open(path, 'r') as f:
            result = f.read()
        expected = pio.to_json(fig1, pretty=pretty, remove_uids=remove_uids)
        assert result == expected