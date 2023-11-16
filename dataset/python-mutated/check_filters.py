from __future__ import annotations
from wtforms import Field, Form

class Filter1:

    def __call__(self, value: object) -> None:
        if False:
            print('Hello World!')
        ...

class Filter2:

    def __call__(self, input: None) -> None:
        if False:
            while True:
                i = 10
        ...

def not_a_filter(a: object, b: object) -> None:
    if False:
        while True:
            i = 10
    ...

def also_not_a_filter() -> None:
    if False:
        i = 10
        return i + 15
    ...
form = Form()
form.process(extra_filters={'foo': (str.upper, str.strip, int), 'bar': (Filter1(), Filter2())})
form.process(extra_filters={'foo': [str.upper, str.strip, int], 'bar': [Filter1(), Filter2()]})
field = Field(filters=(str.upper, str.lower, int))
Field(filters=(Filter1(), Filter2()))
Field(filters=[str.upper, str.lower, int])
Field(filters=[Filter1(), Filter2()])
field.process(None, extra_filters=(str.upper, str.lower, int))
field.process(None, extra_filters=(Filter1(), Filter2()))
field.process(None, extra_filters=[str.upper, str.lower, int])
field.process(None, extra_filters=[Filter1(), Filter2()])
Field(filters=(str.upper, str.lower, int, not_a_filter))
Field(filters=(Filter1(), Filter2(), also_not_a_filter))
Field(filters=[str.upper, str.lower, int, also_not_a_filter])
Field(filters=[Filter1(), Filter2(), not_a_filter])
field.process(None, extra_filters=(str.upper, str.lower, int, not_a_filter))
field.process(None, extra_filters=(Filter1(), Filter2(), also_not_a_filter))
field.process(None, extra_filters=[str.upper, str.lower, int, also_not_a_filter])
field.process(None, extra_filters=[Filter1(), Filter2(), not_a_filter])