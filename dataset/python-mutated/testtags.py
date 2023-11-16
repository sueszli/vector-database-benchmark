from django.template import Library, Node
register = Library()

class EchoNode(Node):

    def __init__(self, contents):
        if False:
            print('Hello World!')
        self.contents = contents

    def render(self, context):
        if False:
            while True:
                i = 10
        return ' '.join(self.contents)

@register.tag
def echo(parser, token):
    if False:
        print('Hello World!')
    return EchoNode(token.contents.split()[1:])
register.tag('other_echo', echo)

@register.filter
def upper(value):
    if False:
        print('Hello World!')
    return value.upper()