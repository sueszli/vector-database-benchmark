from typing import Optional, Union
from django import template
from posthog.utils import compact_number
register = template.Library()
Number = Union[int, float]
register.filter(compact_number)

@register.filter
def percentage(value: Optional[Number], decimals: int=1) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Returns a rounded formatted with a specific number of decimal digits and a % sign. Expects a decimal-based ratio.\n    Example:\n      {% percentage 0.2283113 %}\n      =>  "22.8%"\n    '
    if value is None:
        return '-'
    return '{0:.{decimals}f}%'.format(value * 100, decimals=decimals)