from __future__ import annotations
import pytest
pytest
from bokeh.models import FixedTicker, MathText, PlainText, TeX
import bokeh.models.axes as bma

def test_ticker_accepts_number_sequences() -> None:
    if False:
        print('Hello World!')
    a = bma.Axis(ticker=[-10, 0, 10, 20.7])
    assert isinstance(a.ticker, FixedTicker)
    assert a.ticker.ticks == [-10, 0, 10, 20.7]
    a = bma.Axis()
    a.ticker = [-10, 0, 10, 20.7]
    assert isinstance(a.ticker, FixedTicker)
    assert a.ticker.ticks == [-10, 0, 10, 20.7]

def test_axis_label_with_delimiters_do_not_convert_to_math_text_model() -> None:
    if False:
        return 10
    a = bma.Axis(axis_label='$$\\sin(x+1)$$')
    assert isinstance(a.axis_label, str)
    assert a.axis_label == '$$\\sin(x+1)$$'

def test_axis_label_accepts_math_text_with_declaration() -> None:
    if False:
        while True:
            i = 10
    a = bma.Axis(axis_label=TeX(text='\\sin(x+2)'))
    assert isinstance(a.axis_label, MathText)
    assert a.axis_label.text == '\\sin(x+2)'

def test_axis_label_accepts_math_text_with_declaration_and_dollar_signs() -> None:
    if False:
        while True:
            i = 10
    a = bma.Axis(axis_label=TeX(text='$\\sin(x+3)$'))
    assert isinstance(a.axis_label, MathText)
    assert a.axis_label.text == '$\\sin(x+3)$'

def test_axis_label_accepts_math_text_with_constructor_arg() -> None:
    if False:
        for i in range(10):
            print('nop')
    a = bma.Axis(axis_label=TeX('\\sin(x+4)'))
    assert isinstance(a.axis_label, MathText)
    assert a.axis_label.text == '\\sin(x+4)'

def test_axis_label_accepts_math_text_with_constructor_arg_and_dollar_signs() -> None:
    if False:
        for i in range(10):
            print('nop')
    a = bma.Axis(axis_label=TeX('$\\sin(x+4)$'))
    assert isinstance(a.axis_label, MathText)
    assert a.axis_label.text == '$\\sin(x+4)$'

def test_axis_label_accepts_string_with_dollar_signs() -> None:
    if False:
        for i in range(10):
            print('nop')
    a = bma.Axis(axis_label=PlainText('$\\sin(x+6)$'))
    assert isinstance(a.axis_label, PlainText)
    assert a.axis_label.text == '$\\sin(x+6)$'