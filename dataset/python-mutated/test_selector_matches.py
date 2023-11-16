import plotly.graph_objects as go
from plotly.basedatatypes import BaseFigure

def test_selector_none():
    if False:
        return 10
    assert BaseFigure._selector_matches({}, None) == True

def test_selector_empty_dict():
    if False:
        return 10
    assert BaseFigure._selector_matches(dict(hello='everybody'), {}) == True

def test_selector_matches_subset_of_obj():
    if False:
        return 10
    assert BaseFigure._selector_matches(dict(hello='everybody', today='cloudy', myiq=55), dict(myiq=55, today='cloudy')) == True

def test_selector_has_nonmatching_key():
    if False:
        return 10
    assert BaseFigure._selector_matches(dict(hello='everybody', today='cloudy', myiq=55), dict(myiq=55, cronenberg='scanners')) == False

def test_selector_has_nonmatching_value():
    if False:
        for i in range(10):
            print('nop')
    assert BaseFigure._selector_matches(dict(hello='everybody', today='cloudy', myiq=55), dict(myiq=55, today='sunny')) == False

def test_baseplotlytypes_could_match():
    if False:
        return 10
    obj = go.layout.Annotation(x=1, y=2, text='pat metheny')
    sel = go.layout.Annotation(x=1, y=2, text='pat metheny')
    assert BaseFigure._selector_matches(obj, sel) == True

def test_baseplotlytypes_could_not_match():
    if False:
        while True:
            i = 10
    obj = go.layout.Annotation(x=1, y=3, text='pat metheny')
    sel = go.layout.Annotation(x=1, y=2, text='pat metheny')
    assert BaseFigure._selector_matches(obj, sel) == False

def test_baseplotlytypes_cannot_match_subset():
    if False:
        while True:
            i = 10
    obj = go.layout.Annotation(x=1, y=2, text='pat metheny')
    sel = go.layout.Annotation(x=1, y=2)
    assert BaseFigure._selector_matches(obj, sel) == False

def test_function_selector_could_match():
    if False:
        i = 10
        return i + 15
    obj = go.layout.Annotation(x=1, y=2, text='pat metheny')

    def _sel(d):
        if False:
            i = 10
            return i + 15
        return d['x'] == 1 and d['y'] == 2 and (d['text'] == 'pat metheny')
    assert BaseFigure._selector_matches(obj, _sel) == True

def test_function_selector_could_not_match():
    if False:
        print('Hello World!')
    obj = go.layout.Annotation(x=1, y=2, text='pat metheny')

    def _sel(d):
        if False:
            for i in range(10):
                print('nop')
        return d['x'] == 1 and d['y'] == 3 and (d['text'] == 'pat metheny')
    assert BaseFigure._selector_matches(obj, _sel) == False

def test_string_selector_matches_type_key():
    if False:
        i = 10
        return i + 15
    assert BaseFigure._selector_matches(dict(type='bar'), 'bar')
    assert BaseFigure._selector_matches(dict(type='scatter'), 'bar') == False