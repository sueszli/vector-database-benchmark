"""
test_error_bars:
================

A module intended for use with Nose.

"""
from plotly.graph_objs import ErrorX, ErrorY
from plotly.exceptions import PlotlyDictKeyError

def test_instantiate_error_x():
    if False:
        for i in range(10):
            print('nop')
    ErrorX()
    ErrorX(array=[1, 2, 3], arrayminus=[2, 1, 2], color='red', copy_ystyle=False, symmetric=False, thickness=2, type='percent', value=1, valueminus=4, visible=True, width=5)

def test_instantiate_error_y():
    if False:
        return 10
    ErrorY()
    ErrorY(array=[1, 2, 3], arrayminus=[2, 1, 2], color='red', symmetric=False, thickness=2, type='percent', value=1, valueminus=4, visible=True, width=5)

def test_key_error():
    if False:
        return 10
    assert ErrorX(value=0.1, typ='percent', color='red') == {'color': 'red', 'typ': 'percent', 'value': 0.1}