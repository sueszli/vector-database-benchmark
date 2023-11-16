import random
import string
from django.core.validators import RegexValidator
from django.utils.regex_helper import _lazy_re_compile
from django.utils.text import slugify
from sentry.api.fields.sentry_slug import DEFAULT_SLUG_ERROR_MESSAGE, MIXED_SLUG_PATTERN

def validate_sentry_slug(slug: str) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Validates a sentry slug matches MIXED_SLUG_PATTERN. Raises ValidationError if it does not.\n    '
    validator = RegexValidator(_lazy_re_compile(MIXED_SLUG_PATTERN), DEFAULT_SLUG_ERROR_MESSAGE, 'invalid')
    validator(slug)

def sentry_slugify(slug: str, allow_unicode=False) -> str:
    if False:
        print('Hello World!')
    "\n    Slugify a string using Django's built-in slugify function. Ensures that the\n    slug is not entirely numeric by adding 3 letter suffix if necessary.\n    "
    slug = slugify(slug, allow_unicode=allow_unicode)
    if slug.isdecimal():
        slug = f"{slug}-{''.join((random.choice(string.ascii_lowercase) for _ in range(3)))}"
    return slug