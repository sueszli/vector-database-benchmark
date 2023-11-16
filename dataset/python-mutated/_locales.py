"""Provide class for testing in French locale

"""
import sys
import locale
import pytest
__ALL__ = ['CommaDecimalPointLocale']

def find_comma_decimal_point_locale():
    if False:
        while True:
            i = 10
    'See if platform has a decimal point as comma locale.\n\n    Find a locale that uses a comma instead of a period as the\n    decimal point.\n\n    Returns\n    -------\n    old_locale: str\n        Locale when the function was called.\n    new_locale: {str, None)\n        First French locale found, None if none found.\n\n    '
    if sys.platform == 'win32':
        locales = ['FRENCH']
    else:
        locales = ['fr_FR', 'fr_FR.UTF-8', 'fi_FI', 'fi_FI.UTF-8']
    old_locale = locale.getlocale(locale.LC_NUMERIC)
    new_locale = None
    try:
        for loc in locales:
            try:
                locale.setlocale(locale.LC_NUMERIC, loc)
                new_locale = loc
                break
            except locale.Error:
                pass
    finally:
        locale.setlocale(locale.LC_NUMERIC, locale=old_locale)
    return (old_locale, new_locale)

class CommaDecimalPointLocale:
    """Sets LC_NUMERIC to a locale with comma as decimal point.

    Classes derived from this class have setup and teardown methods that run
    tests with locale.LC_NUMERIC set to a locale where commas (',') are used as
    the decimal point instead of periods ('.'). On exit the locale is restored
    to the initial locale. It also serves as context manager with the same
    effect. If no such locale is available, the test is skipped.

    .. versionadded:: 1.15.0

    """
    (cur_locale, tst_locale) = find_comma_decimal_point_locale()

    def setup_method(self):
        if False:
            while True:
                i = 10
        if self.tst_locale is None:
            pytest.skip('No French locale available')
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    def teardown_method(self):
        if False:
            print('Hello World!')
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if self.tst_locale is None:
            pytest.skip('No French locale available')
        locale.setlocale(locale.LC_NUMERIC, locale=self.tst_locale)

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        locale.setlocale(locale.LC_NUMERIC, locale=self.cur_locale)