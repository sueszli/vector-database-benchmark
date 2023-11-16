import plotly.graph_objects as go
from _plotly_utils.exceptions import PlotlyKeyError
import pytest

def error_substr(s, r):
    if False:
        return 10
    "remove a part of the error message we don't want to compare"
    return s.replace(r, '')

@pytest.fixture
def some_fig():
    if False:
        print('Hello World!')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[]))
    fig.add_shape(type='rect', x0=1, x1=2, y0=3, y1=4)
    fig.add_shape(type='rect', x0=10, x1=20, y0=30, y1=40)
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    return fig

def test_raises_on_bad_index(some_fig):
    if False:
        print('Hello World!')
    raised = False
    try:
        x0 = some_fig['layout.shapes[2].x0']
    except KeyError as e:
        raised = True
        assert e.args[0].find('Bad property path:\nlayout.shapes[2].x0\n              ^') >= 0
    assert raised

def test_raises_on_bad_dot_property(some_fig):
    if False:
        while True:
            i = 10
    raised = False
    try:
        x2000 = some_fig['layout.shapes[1].x2000']
    except KeyError as e:
        raised = True
        assert e.args[0].find('Bad property path:\nlayout.shapes[1].x2000\n                 ^^^^^') and (e.args[0].find('Did you mean "x0"?') >= 0) >= 0
    assert raised

def test_raises_on_bad_ancestor_dot_property(some_fig):
    if False:
        for i in range(10):
            print('nop')
    raised = False
    try:
        x2000 = some_fig['layout.shapa[1].x2000']
    except KeyError as e:
        raised = True
        assert e.args[0].find('Bad property path:\nlayout.shapa[1].x2000\n       ^^^^^') and (e.args[0].find('Did you mean "shapes"?') >= 0) >= 0
    assert raised

def test_raises_on_bad_indexed_underscore_property(some_fig):
    if False:
        for i in range(10):
            print('nop')
    raised = False
    try:
        some_fig.data[0].line['colr'] = 'blue'
    except ValueError as e_correct:
        raised = True
        e_correct_substr = error_substr(e_correct.args[0], '\nBad property path:\ncolr\n^^^^')
    assert len(e_correct_substr) > 0
    assert raised
    raised = False
    try:
        some_fig['data[0].line_colr'] = 'blue'
    except ValueError as e:
        raised = True
        e_substr = error_substr(e.args[0], '\nBad property path:\ndata[0].line_colr\n             ^^^^')
        assert e.args[0].find('Bad property path:\ndata[0].line_colr\n             ^^^^') >= 0 and e.args[0].find('Did you mean "color"?') >= 0 and (e_substr == e_correct_substr)
    assert raised
    raised = False
    try:
        some_fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], line=dict(colr='blue')))
    except ValueError as e_correct:
        raised = True
        e_correct_substr = error_substr(e_correct.args[0], '\nBad property path:\ncolr\n^^^^')
    assert raised
    raised = False
    try:
        some_fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], line_colr='blue'))
    except ValueError as e:
        raised = True
        e_substr = error_substr(e.args[0], '\nBad property path:\nline_colr\n     ^^^^')
        assert (e.args[0].find('Bad property path:\nline_colr\n     ^^^^') and (e.args[0].find('Did you mean "color"?') >= 0) >= 0) and e_substr == e_correct_substr
    assert raised
    raised = False
    try:
        fig2 = go.Figure(layout=dict(title=dict(txt='two')))
    except ValueError as e_correct:
        raised = True
        e_correct_substr = error_substr(e_correct.args[0], '\nBad property path:\ntxt\n^^^')
    assert raised
    raised = False
    try:
        fig2 = go.Figure(layout_title_txt='two')
    except TypeError as e:
        raised = True
        e_substr = error_substr(e.args[0], '\nBad property path:\nlayout_title_txt\n             ^^^')
        e_substr = error_substr(e_substr, 'invalid Figure property: layout_title_txt\n')
        assert e.args[0].find('Bad property path:\nlayout_title_txt\n             ^^^') >= 0 and e.args[0].find('Did you mean "text"?') >= 0 and (e_substr == e_correct_substr)
    assert raised
    raised = False
    try:
        some_fig.update_layout(geo=dict(ltaxis=dict(showgrid=True)))
    except ValueError as e_correct:
        raised = True
        e_correct_substr = error_substr(e_correct.args[0], '\nBad property path:\nltaxis\n^^^^^^')
    assert raised
    raised = False
    try:
        some_fig.update_layout(geo_ltaxis_showgrid=True)
    except ValueError as e:
        raised = True
        e_substr = error_substr(e.args[0], '\nBad property path:\ngeo_ltaxis_showgrid\n    ^^^^^^')
        assert e.args[0].find('Bad property path:\ngeo_ltaxis_showgrid\n    ^^^^^^') >= 0 and e.args[0].find('Did you mean "lataxis"?') >= 0 and (e_substr == e_correct_substr)
    assert raised

def test_describes_subscripting_error(some_fig):
    if False:
        i = 10
        return i + 15
    raised = False
    try:
        some_fig.data[0].text['yo']
    except TypeError as e:
        raised = True
        e_correct_substr = e.args[0]
    assert raised
    raised = False
    try:
        some_fig.update_traces(text_yo='hey')
    except ValueError as e:
        raised = True
        print(e.args[0])
        e_substr = error_substr(e.args[0], "\n\nInvalid value received for the 'text' property of scatter\n\n    The 'text' property is a string and must be specified as:\n      - A string\n      - A number that will be converted to a string\n      - A tuple, list, or one-dimensional numpy array of the above\n\nProperty does not support subscripting:\ntext_yo\n^^^^")
        assert e.args[0].find('\nProperty does not support subscripting:\ntext_yo\n^^^^') >= 0 and e_substr == e_correct_substr
    assert raised
    raised = False
    try:
        some_fig.data[0].textfont.family['yo']
    except TypeError as e:
        raised = True
        e_correct_substr = e.args[0]
    assert raised
    raised = False
    try:
        go.Figure(go.Scatter()).update_traces(textfont_family_yo='hey')
    except ValueError as e:
        raised = True
        e_substr = error_substr(e.args[0], "\n\nInvalid value received for the 'family' property of scatter.textfont\n\n    The 'family' property is a string and must be specified as:\n      - A non-empty string\n      - A tuple, list, or one-dimensional numpy array of the above\n\nProperty does not support subscripting:\ntextfont_family_yo\n         ^^^^^^")
        assert e.args[0].find('\nProperty does not support subscripting:\ntextfont_family_yo\n         ^^^^^^') >= 0 and e_substr == e_correct_substr
    assert raised

def test_described_subscript_error_on_type_error(some_fig):
    if False:
        return 10
    raised = False
    try:
        some_fig['layout_template_layout_plot_bgcolor'] = 1
    except ValueError as e:
        raised = True
        e_correct_substr = e.args[0]
        start_at = e_correct_substr.find("    The 'plot_bgcolor'")
        e_correct_substr = e_correct_substr[start_at:]
        e_correct_substr += '\n\nProperty does not support subscripting:\ntemplate_layout_plot_bgcolor_x\n                ^^^^^^^^^^^^'
    assert raised
    raised = False
    try:
        some_fig.update_layout(template_layout_plot_bgcolor_x=1)
    except ValueError as e:
        raised = True
        print(e.args[0])
        e_substr = error_substr(e.args[0], "string indices must be integers\n\nInvalid value received for the 'plot_bgcolor' property of layout\n\n")
        assert e_substr == e_correct_substr
    assert raised

def test_subscript_error_exception_types(some_fig):
    if False:
        i = 10
        return i + 15
    with pytest.raises(ValueError):
        some_fig.update_layout(width_yo=100)
    with pytest.raises(KeyError):
        yo = some_fig['layout_width_yo']
    some_fig.update_layout(width=100)
    with pytest.raises(ValueError):
        some_fig.update_layout(width_yo=100)
    with pytest.raises(KeyError):
        yo = some_fig['layout_width_yo']

def form_error_string(call, exception, subs):
    if False:
        for i in range(10):
            print('nop')
    '\n    call is a function that raises exception.\n    exception is an exception class, e.g., KeyError.\n    subs is a list of replacements to be performed on the exception string. Each\n    replacement is only performed once on the exception string so the\n    replacement of multiple occurences of a pattern is specified by repeating a\n    (pattern,relacement) pair in the list.\n    returns modified exception string\n    '
    raised = False
    try:
        call()
    except exception as e:
        raised = True
        msg = e.args[0]
        for (pat, rep) in subs:
            msg = msg.replace(pat, rep, 1)
    assert raised
    return msg

def check_error_string(call, exception, correct_str, subs):
    if False:
        print('Hello World!')
    raised = False
    try:
        call()
    except exception as e:
        raised = True
        msg = e.args[0]
        for (pat, rep) in subs:
            msg = msg.replace(pat, rep, 1)
        print('MSG')
        print(msg)
        print('CORRECT')
        print(correct_str)
        assert msg == correct_str
    assert raised

def test_leading_underscore_errors(some_fig):
    if False:
        while True:
            i = 10

    def _raise_bad_property_path_form():
        if False:
            for i in range(10):
                print('nop')
        some_fig.update_layout(bogus=7)

    def _raise_bad_property_path_real():
        if False:
            return 10
        some_fig.update_layout(_hey_yall=7)
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('bogus', '_hey'), ('bogus', '_hey_yall'), ('^^^^^', '^^^^'), ('Did you mean "boxgap"', 'Did you mean "geo"'), ('Did you mean "boxgap"', 'Did you mean "geo"')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_trailing_underscore_errors(some_fig):
    if False:
        for i in range(10):
            print('nop')

    def _raise_bad_property_path_form():
        if False:
            i = 10
            return i + 15
        some_fig.update_layout(title_text_bogus='hi')

    def _raise_bad_property_path_real():
        if False:
            print('Hello World!')
        some_fig.update_layout(title_text_='hi')
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('Property does not support subscripting', 'Property does not support subscripting and path has trailing underscores'), ('text_bogus', 'text_'), ('^^^^', '^^^^^')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_embedded_underscore_errors(some_fig):
    if False:
        while True:
            i = 10

    def _raise_bad_property_path_form():
        if False:
            print('Hello World!')
        some_fig.update_layout(title_font_bogusey='hi')

    def _raise_bad_property_path_real():
        if False:
            return 10
        some_fig.update_layout(title_font__family='hi')
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('bogusey', '_family'), ('bogusey', '_family'), ('Did you mean "color"?', 'Did you mean "family"?'), ('Did you mean "color"?', 'Did you mean "family"?')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_solo_underscore_errors(some_fig):
    if False:
        while True:
            i = 10

    def _raise_bad_property_path_form():
        if False:
            print('Hello World!')
        some_fig.update_layout(bogus='hi')

    def _raise_bad_property_path_real():
        if False:
            print('Hello World!')
        some_fig.update_layout(_='hi')
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('bogus', '_'), ('bogus', '_'), ('^^^^^', '^'), ('Did you mean "boxgap"', 'Did you mean "geo"'), ('Did you mean "boxgap"', 'Did you mean "geo"')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_repeated_underscore_errors(some_fig):
    if False:
        return 10

    def _raise_bad_property_path_form():
        if False:
            i = 10
            return i + 15
        some_fig.update_layout(bogus='hi')

    def _raise_bad_property_path_real():
        if False:
            print('Hello World!')
        some_fig.update_layout(__='hi')
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('bogus', '__'), ('bogus', '__'), ('^^^^^', '^^'), ('Did you mean "boxgap"', 'Did you mean "geo"'), ('Did you mean "boxgap"', 'Did you mean "geo"')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_leading_underscore_errors_dots_and_subscripts(some_fig):
    if False:
        while True:
            i = 10
    some_fig.add_annotation(text='hi')

    def _raise_bad_property_path_form():
        if False:
            for i in range(10):
                print('nop')
        some_fig['layout.annotations[0].bogus_family'] = 'hi'

    def _raise_bad_property_path_real():
        if False:
            return 10
        some_fig['layout.annotations[0]._font_family'] = 'hi'
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('bogus', '_font'), ('bogus', '_font'), ('^^^^^', '^^^^^')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_trailing_underscore_errors_dots_and_subscripts(some_fig):
    if False:
        print('Hello World!')
    some_fig.add_annotation(text='hi')

    def _raise_bad_property_path_form():
        if False:
            while True:
                i = 10
        some_fig['layout.annotations[0].font_family_bogus'] = 'hi'

    def _raise_bad_property_path_real():
        if False:
            print('Hello World!')
        some_fig['layout.annotations[0].font_family_'] = 'hi'
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('Property does not support subscripting', 'Property does not support subscripting and path has trailing underscores'), ('family_bogus', 'family_'), ('^^^^^^', '^^^^^^^')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_repeated_underscore_errors_dots_and_subscripts(some_fig):
    if False:
        for i in range(10):
            print('nop')
    some_fig.add_annotation(text='hi')

    def _raise_bad_property_path_form():
        if False:
            return 10
        some_fig['layout.annotations[0].font_bogusey'] = 'hi'

    def _raise_bad_property_path_real():
        if False:
            i = 10
            return i + 15
        some_fig['layout.annotations[0].font__family'] = 'hi'
    correct_err_str = form_error_string(_raise_bad_property_path_form, ValueError, [('bogusey', '_family'), ('bogusey', '_family'), ('Did you mean "color"?', 'Did you mean "family"?'), ('Did you mean "color"?', 'Did you mean "family"?')])
    check_error_string(_raise_bad_property_path_real, ValueError, correct_err_str, [])

def test_single_prop_path_key_guess(some_fig):
    if False:
        print('Hello World!')
    raised = False
    try:
        some_fig.layout.shapes[0]['typ'] = 'sandwich'
    except ValueError as e:
        raised = True
        assert e.args[0].find('Did you mean "type"?') >= 0
    assert raised