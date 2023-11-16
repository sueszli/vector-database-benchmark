import inspect
from hypothesis._settings import Verbosity, settings
from hypothesis.internal.compat import escape_unicode_characters
from hypothesis.utils.dynamicvariables import DynamicVariable

def default(value):
    if False:
        for i in range(10):
            print('nop')
    try:
        print(value)
    except UnicodeEncodeError:
        print(escape_unicode_characters(value))
reporter = DynamicVariable(default)

def current_reporter():
    if False:
        print('Hello World!')
    return reporter.value

def with_reporter(new_reporter):
    if False:
        i = 10
        return i + 15
    return reporter.with_value(new_reporter)

def current_verbosity():
    if False:
        i = 10
        return i + 15
    return settings.default.verbosity

def to_text(textish):
    if False:
        return 10
    if inspect.isfunction(textish):
        textish = textish()
    if isinstance(textish, bytes):
        textish = textish.decode()
    return textish

def verbose_report(text):
    if False:
        i = 10
        return i + 15
    if current_verbosity() >= Verbosity.verbose:
        base_report(text)

def debug_report(text):
    if False:
        return 10
    if current_verbosity() >= Verbosity.debug:
        base_report(text)

def report(text):
    if False:
        for i in range(10):
            print('nop')
    if current_verbosity() >= Verbosity.normal:
        base_report(text)

def base_report(text):
    if False:
        while True:
            i = 10
    current_reporter()(to_text(text))