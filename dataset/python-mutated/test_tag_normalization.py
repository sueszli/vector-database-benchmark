import pytest
from sentry.utils.tag_normalization import normalize_sdk_tag

@pytest.mark.parametrize(('tag', 'expected'), (('sentry-javascript-angular', 'sentry.javascript.angular'), ('sentry_python', 'sentry.python')))
def test_normalizes_to_dots(tag, expected):
    if False:
        print('Hello World!')
    assert normalize_sdk_tag(tag) == expected

@pytest.mark.parametrize(('tag', 'expected'), (('sentry.javascript.angular', 'sentry.javascript.angular'), ('sentry.javascript.react.native', 'sentry.javascript.react.native'), ('sentry.python.django', 'sentry.python'), ('sentry.native.android.flutter', 'sentry.native.android')))
def test_shortens_non_js(tag, expected):
    if False:
        for i in range(10):
            print('nop')
    assert normalize_sdk_tag(tag) == expected

@pytest.mark.parametrize(('tag', 'expected'), (('sentry.javascript.angular', 'sentry.javascript.angular'), ('sentry.javascript.angular.ivy', 'sentry.javascript.angular'), ('sentry.symfony', 'sentry.php'), ('sentry.unity', 'sentry.native.unity'), ('sentry.javascript.react.native.expo', 'sentry.javascript.react.native')))
def test_uses_synonyms(tag, expected):
    if False:
        while True:
            i = 10
    assert normalize_sdk_tag(tag) == expected

@pytest.mark.parametrize(('tag', 'expected'), (('foo.baz.bar', 'other'), ('sentryfoo', 'other'), ('raven', 'other')))
def test_non_sentry_to_other(tag, expected):
    if False:
        return 10
    assert normalize_sdk_tag(tag) == expected

@pytest.mark.parametrize(('tag', 'expected'), (('sentry.sparql', 'other'), ('sentry.terraform.hcl', 'other'), ('sentry-native', 'other')))
def test_unknown_sentry_to_other(tag, expected):
    if False:
        print('Hello World!')
    assert normalize_sdk_tag(tag) == expected

def test_responses_cached():
    if False:
        i = 10
        return i + 15
    normalize_sdk_tag.cache_clear()
    assert normalize_sdk_tag('sentry.javascript.react') == 'sentry.javascript.react'
    assert normalize_sdk_tag('sentry.javascript.react') == 'sentry.javascript.react'
    assert normalize_sdk_tag.cache_info().hits == 1
    assert normalize_sdk_tag.cache_info().misses == 1