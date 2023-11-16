from django import template
register = template.Library()

@register.tag
def badtag(parser, token):
    if False:
        i = 10
        return i + 15
    parser.parse(('endbadtag',))
    parser.delete_first_token()
    return BadNode()

class BadNode(template.Node):

    def render(self, context):
        if False:
            i = 10
            return i + 15
        raise template.TemplateSyntaxError('error')