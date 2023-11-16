"""Test for the text utils module"""
from hamcrest import assert_that, equal_to
from deepchecks.nlp.utils.text import break_to_lines_and_trim

def test_break_to_lines_and_trim():
    if False:
        while True:
            i = 10
    text = 'This is a very long text that should be broken into lines. '
    res_text = break_to_lines_and_trim(text, max_lines=3, min_line_length=55, max_line_length=65)
    assert_that(res_text, equal_to(text.strip()))
    res_text = break_to_lines_and_trim(text, max_lines=3, min_line_length=15, max_line_length=25)
    assert_that(res_text, equal_to('This is a very long text<br>that should be broken<br>into lines.'))
    res_text = break_to_lines_and_trim(text, max_lines=2, min_line_length=15, max_line_length=25)
    assert_that(res_text, equal_to('This is a very long text<br>that should be broken...'))
    res_text = break_to_lines_and_trim(text, max_lines=3, min_line_length=12, max_line_length=13)
    assert_that(res_text, equal_to('This is a ver-<br>y long text t-<br>hat should be...'))