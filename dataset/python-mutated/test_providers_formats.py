import re
import pytest
from faker import Factory
from faker.config import AVAILABLE_LOCALES, DEFAULT_LOCALE, PROVIDERS
locales = AVAILABLE_LOCALES
find_group = re.compile('\\{\\{(\\w+)\\}\\}')

@pytest.mark.parametrize('locale', locales)
def test_no_invalid_formats(locale):
    if False:
        for i in range(10):
            print('nop')
    '\n    For each locale, for each provider, search all the definitions of "formats"\n    and make sure that all the providers in there (e.g. {{group}}) are valid\n    and do not emit empty strings. Empty strings are allowed only if the group\n    is not surrounded by spaces. This is a quick way to make sure that no\n    string is generated with "double spaces", starting spaces or ending spaces.\n    '
    faker = Factory.create(locale)
    errors = []
    for provider in PROVIDERS:
        if provider == 'faker.providers':
            continue
        (prov_cls, lang, default_lang) = Factory._find_provider_class(provider, locale)
        if default_lang is None:
            assert lang is None
        else:
            assert lang in (locale, default_lang or DEFAULT_LOCALE)
        attributes = set(dir(prov_cls))
        for attribute in attributes:
            if not attribute.endswith('formats'):
                continue
            formats = getattr(prov_cls, attribute)
            if not isinstance(formats, (list, tuple)):
                continue
            for format in formats:
                for match in find_group.finditer(format):
                    group = match.group(1)
                    try:
                        attr = faker.format(group)
                    except AttributeError as e:
                        errors.append(str(e))
                        continue
                    touching = False
                    if match.start() != 0 and format[match.start() - 1] != ' ':
                        touching = True
                    if match.end() != len(format) and format[match.end()] != ' ':
                        touching = True
                    if not attr and (not touching):
                        errors.append("Attribute {{%s}} provided an invalid value in format '%s' from %s.%s.%s" % (group, format, provider, locale, attribute))
    assert not errors, 'Errors:\n - ' + '\n - '.join(errors)