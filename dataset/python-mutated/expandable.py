import re
from django import forms
from django.utils.translation import gettext_lazy as _
from utilities.forms.constants import *
from utilities.forms.utils import expand_alphanumeric_pattern, expand_ipaddress_pattern
__all__ = ('ExpandableIPAddressField', 'ExpandableNameField')

class ExpandableNameField(forms.CharField):
    """
    A field which allows for numeric range expansion
      Example: 'Gi0/[1-3]' => ['Gi0/1', 'Gi0/2', 'Gi0/3']
    """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        if not self.help_text:
            self.help_text = _('Alphanumeric ranges are supported for bulk creation. Mixed cases and types within a single range are not supported (example: <code>[ge,xe]-0/0/[0-9]</code>).')

    def to_python(self, value):
        if False:
            i = 10
            return i + 15
        if not value:
            return ''
        if re.search(ALPHANUMERIC_EXPANSION_PATTERN, value):
            return list(expand_alphanumeric_pattern(value))
        return [value]

class ExpandableIPAddressField(forms.CharField):
    """
    A field which allows for expansion of IP address ranges
      Example: '192.0.2.[1-254]/24' => ['192.0.2.1/24', '192.0.2.2/24', '192.0.2.3/24' ... '192.0.2.254/24']
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        if not self.help_text:
            self.help_text = _('Specify a numeric range to create multiple IPs.<br />Example: <code>192.0.2.[1,5,100-254]/24</code>')

    def to_python(self, value):
        if False:
            for i in range(10):
                print('nop')
        if '.' in value and re.search(IP4_EXPANSION_PATTERN, value):
            return list(expand_ipaddress_pattern(value, 4))
        elif ':' in value and re.search(IP6_EXPANSION_PATTERN, value):
            return list(expand_ipaddress_pattern(value, 6))
        return [value]