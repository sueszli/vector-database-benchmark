import sys
from . import version_skip

@version_skip
def test_dependencies_not_imported():
    if False:
        while True:
            i = 10
    assert 'plotly' not in sys.modules
    assert 'numpy' not in sys.modules
    assert 'pandas' not in sys.modules
    import plotly.graph_objects as go
    fig = go.Figure().add_scatter(x=[0], y=[1])
    fig.to_json()
    assert 'plotly' in sys.modules
    assert 'numpy' not in sys.modules
    assert 'pandas' not in sys.modules
    import numpy as np
    fig = go.Figure().add_scatter(x=np.array([0]), y=np.array([1]))
    fig.to_json()
    assert 'numpy' in sys.modules
    assert 'pandas' not in sys.modules
    import pandas as pd
    fig = go.Figure().add_scatter(x=pd.Series([0]), y=pd.Series([1]))
    fig.to_json()
    assert 'pandas' in sys.modules