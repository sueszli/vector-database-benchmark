from django import template
register = template.Library()

@register.tag
def badtag(parser, token):
    if False:
        while True:
            i = 10
    raise RuntimeError('I am a bad tag')

@register.simple_tag
def badsimpletag():
    if False:
        i = 10
        return i + 15
    raise RuntimeError('I am a bad simpletag')