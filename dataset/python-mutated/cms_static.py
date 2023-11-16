from django import template
from django.templatetags.static import StaticNode
from cms.utils.urlutils import static_with_version
register = template.Library()

@register.tag('static_with_version')
def do_static_with_version(parser, token):
    if False:
        i = 10
        return i + 15
    '\n    Joins the given path with the STATIC_URL setting\n    and appends the CMS version as a GET parameter.\n\n    Usage::\n        {% static_with_version path [as varname] %}\n    Examples::\n        {% static_with_version "myapp/css/base.css" %}\n        {% static_with_version variable_with_path %}\n        {% static_with_version "myapp/css/base.css" as admin_base_css %}\n        {% static_with_version variable_with_path as varname %}\n    '
    return StaticWithVersionNode.handle_token(parser, token)

class StaticWithVersionNode(StaticNode):

    def url(self, context):
        if False:
            while True:
                i = 10
        path = self.path.resolve(context)
        path_with_version = static_with_version(path)
        return self.handle_simple(path_with_version)