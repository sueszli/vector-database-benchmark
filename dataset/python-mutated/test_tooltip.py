import numpy as np
import pytest
from pandas import DataFrame
pytest.importorskip('jinja2')
from pandas.io.formats.style import Styler

@pytest.fixture
def df():
    if False:
        print('Hello World!')
    return DataFrame(data=[[0, 1, 2], [3, 4, 5], [6, 7, 8]], columns=['A', 'B', 'C'], index=['x', 'y', 'z'])

@pytest.fixture
def styler(df):
    if False:
        i = 10
        return i + 15
    return Styler(df, uuid_len=0)

@pytest.mark.parametrize('ttips', [DataFrame(data=[['Min', 'Max'], [np.nan, '']], columns=['A', 'C'], index=['x', 'y']), DataFrame(data=[['Max', 'Min', 'Bad-Col']], columns=['C', 'A', 'D'], index=['x'])])
def test_tooltip_render(ttips, styler):
    if False:
        i = 10
        return i + 15
    result = styler.set_tooltips(ttips).to_html()
    assert '#T_ .pd-t {\n  visibility: hidden;\n' in result
    assert '#T_ #T__row0_col0:hover .pd-t {\n  visibility: visible;\n}' in result
    assert '#T_ #T__row0_col0 .pd-t::after {\n  content: "Min";\n}' in result
    assert 'class="data row0 col0" >0<span class="pd-t"></span></td>' in result
    assert '#T_ #T__row0_col2:hover .pd-t {\n  visibility: visible;\n}' in result
    assert '#T_ #T__row0_col2 .pd-t::after {\n  content: "Max";\n}' in result
    assert 'class="data row0 col2" >2<span class="pd-t"></span></td>' in result
    assert '#T_ #T__row1_col0:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert '#T_ #T__row1_col1:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert '#T_ #T__row0_col1:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert '#T_ #T__row1_col2:hover .pd-t {\n  visibility: visible;\n}' not in result
    assert 'Bad-Col' not in result

def test_tooltip_ignored(styler):
    if False:
        i = 10
        return i + 15
    result = styler.to_html()
    assert '<style type="text/css">\n</style>' in result
    assert '<span class="pd-t"></span>' not in result

def test_tooltip_css_class(styler):
    if False:
        print('Hello World!')
    result = styler.set_tooltips(DataFrame([['tooltip']], index=['x'], columns=['A']), css_class='other-class', props=[('color', 'green')]).to_html()
    assert '#T_ .other-class {\n  color: green;\n' in result
    assert '#T_ #T__row0_col0 .other-class::after {\n  content: "tooltip";\n' in result
    result = styler.set_tooltips(DataFrame([['tooltip']], index=['x'], columns=['A']), css_class='another-class', props='color:green;color:red;').to_html()
    assert '#T_ .another-class {\n  color: green;\n  color: red;\n}' in result