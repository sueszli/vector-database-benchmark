from django import template
register = template.Library()

@register.simple_tag
def go_boom():
    if False:
        i = 10
        return i + 15
    raise Exception('boom')