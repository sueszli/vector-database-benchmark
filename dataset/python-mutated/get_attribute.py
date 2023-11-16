from django import template
register = template.Library()

@register.filter
def get_attribute(obj, name):
    if False:
        print('Hello World!')
    if hasattr(obj, name):
        return getattr(obj, name)
    else:
        return ''