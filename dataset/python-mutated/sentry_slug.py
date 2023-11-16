from __future__ import annotations
from django.utils.translation import gettext_lazy as _
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema_field
from rest_framework import serializers
DEFAULT_SLUG_ERROR_MESSAGE = _('Enter a valid slug consisting of lowercase letters, numbers, underscores or hyphens. It cannot be entirely numeric.')
'\nStandard slug pattern:\n    (?![0-9]+$) - Negative lookahead to ensure the slug is not entirely numeric\n    [a-z0-9_\\-] - Matches lowercase letters, numbers, underscores, and hyphens\n'
MIXED_SLUG_PATTERN = '^(?![0-9]+$)[a-z0-9_\\-]+$'
'\nOrganization slug pattern:\n    (?![0-9]+$)   - Negative lookahead to ensure the slug is not entirely numeric\n    [a-zA-Z0-9]   - Must start with a lowercase letter or number\n    [a-zA-Z0-9-]* - Matches lowercase letters, numbers, and hyphens\n    (?<!-)        - Negative lookbehind to ensure the slug does not end with a hyphen\n'
ORG_SLUG_PATTERN = '^(?![0-9]+$)[a-zA-Z0-9][a-zA-Z0-9-]*(?<!-)$'

@extend_schema_field(field=OpenApiTypes.STR)
class SentrySlugField(serializers.RegexField):
    """
    A regex field which validates that the input is a valid slug. Default
    allowed characters are lowercase letters, numbers, underscores, and hyphens.
    The slug cannot be entirely numeric.
    """
    default_error_messages = {'invalid': DEFAULT_SLUG_ERROR_MESSAGE}

    def __init__(self, error_messages=None, org_slug: bool=False, *args, **kwargs):
        if False:
            return 10
        if error_messages is None:
            error_messages = self.default_error_messages.copy()
        pattern = MIXED_SLUG_PATTERN
        if org_slug:
            pattern = ORG_SLUG_PATTERN
        super().__init__(pattern, *args, error_messages=error_messages, **kwargs)