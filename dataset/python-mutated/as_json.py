import json
from django import template
register = template.Library()

@register.filter
def as_json(value):
    if False:
        while True:
            i = 10
    return json.dumps(value)