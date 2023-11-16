from enum import Enum
import pytest
from mimesis.exceptions import LocaleError
from mimesis.locales import Locale, validate_locale

def test_locale_enum():
    if False:
        print('Hello World!')
    assert len(list(Locale)) == 34
    assert issubclass(Locale, Enum)

def test_validate_locale_missing_locale():
    if False:
        return 10
    with pytest.raises(TypeError):
        validate_locale()

def test_validate_locale_invalid_locale():
    if False:
        print('Hello World!')
    with pytest.raises(LocaleError):
        validate_locale(locale=None)
    with pytest.raises(LocaleError):
        validate_locale(locale='nil')

def test_validate_locale():
    if False:
        print('Hello World!')
    validated_locale = validate_locale('en')
    assert validated_locale == Locale.EN
    assert issubclass(validated_locale.__class__, Enum)
    assert isinstance(validate_locale(Locale.EN), Locale)