from io import BytesIO, StringIO
import pytest
import pandas as pd
import pandas._testing as tm
pytest.importorskip('tabulate')

def test_simple():
    if False:
        i = 10
        return i + 15
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == '|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |'

def test_empty_frame():
    if False:
        return 10
    buf = StringIO()
    df = pd.DataFrame({'id': [], 'first_name': [], 'last_name': []}).set_index('id')
    df.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == '| id   | first_name   | last_name   |\n|------|--------------|-------------|'

def test_other_tablefmt():
    if False:
        return 10
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, tablefmt='jira')
    result = buf.getvalue()
    assert result == '||    ||   0 ||\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |'

def test_other_headers():
    if False:
        return 10
    buf = StringIO()
    df = pd.DataFrame([1, 2, 3])
    df.to_markdown(buf=buf, headers=['foo', 'bar'])
    result = buf.getvalue()
    assert result == '|   foo |   bar |\n|------:|------:|\n|     0 |     1 |\n|     1 |     2 |\n|     2 |     3 |'

def test_series():
    if False:
        return 10
    buf = StringIO()
    s = pd.Series([1, 2, 3], name='foo')
    s.to_markdown(buf=buf)
    result = buf.getvalue()
    assert result == '|    |   foo |\n|---:|------:|\n|  0 |     1 |\n|  1 |     2 |\n|  2 |     3 |'

def test_no_buf():
    if False:
        return 10
    df = pd.DataFrame([1, 2, 3])
    result = df.to_markdown()
    assert result == '|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |'

@pytest.mark.parametrize('index', [True, False])
def test_index(index):
    if False:
        while True:
            i = 10
    df = pd.DataFrame([1, 2, 3])
    result = df.to_markdown(index=index)
    if index:
        expected = '|    |   0 |\n|---:|----:|\n|  0 |   1 |\n|  1 |   2 |\n|  2 |   3 |'
    else:
        expected = '|   0 |\n|----:|\n|   1 |\n|   2 |\n|   3 |'
    assert result == expected

def test_showindex_disallowed_in_kwargs():
    if False:
        i = 10
        return i + 15
    df = pd.DataFrame([1, 2, 3])
    with pytest.raises(ValueError, match="Pass 'index' instead of 'showindex"):
        df.to_markdown(index=True, showindex=True)

def test_markdown_pos_args_deprecatation():
    if False:
        return 10
    df = pd.DataFrame({'a': [1, 2, 3]})
    msg = "Starting with pandas version 3.0 all arguments of to_markdown except for the argument 'buf' will be keyword-only."
    with tm.assert_produces_warning(FutureWarning, match=msg):
        buffer = BytesIO()
        df.to_markdown(buffer, 'grid')