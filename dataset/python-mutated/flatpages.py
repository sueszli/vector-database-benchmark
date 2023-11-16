from django import template
from django.conf import settings
from django.contrib.flatpages.models import FlatPage
from django.contrib.sites.shortcuts import get_current_site
register = template.Library()

class FlatpageNode(template.Node):

    def __init__(self, context_name, starts_with=None, user=None):
        if False:
            while True:
                i = 10
        self.context_name = context_name
        if starts_with:
            self.starts_with = template.Variable(starts_with)
        else:
            self.starts_with = None
        if user:
            self.user = template.Variable(user)
        else:
            self.user = None

    def render(self, context):
        if False:
            return 10
        if 'request' in context:
            site_pk = get_current_site(context['request']).pk
        else:
            site_pk = settings.SITE_ID
        flatpages = FlatPage.objects.filter(sites__id=site_pk)
        if self.starts_with:
            flatpages = flatpages.filter(url__startswith=self.starts_with.resolve(context))
        if self.user:
            user = self.user.resolve(context)
            if not user.is_authenticated:
                flatpages = flatpages.filter(registration_required=False)
        else:
            flatpages = flatpages.filter(registration_required=False)
        context[self.context_name] = flatpages
        return ''

@register.tag
def get_flatpages(parser, token):
    if False:
        while True:
            i = 10
    "\n    Retrieve all flatpage objects available for the current site and\n    visible to the specific user (or visible to all users if no user is\n    specified). Populate the template context with them in a variable\n    whose name is defined by the ``as`` clause.\n\n    An optional ``for`` clause controls the user whose permissions are used in\n    determining which flatpages are visible.\n\n    An optional argument, ``starts_with``, limits the returned flatpages to\n    those beginning with a particular base URL. This argument can be a variable\n    or a string, as it resolves from the template context.\n\n    Syntax::\n\n        {% get_flatpages ['url_starts_with'] [for user] as context_name %}\n\n    Example usage::\n\n        {% get_flatpages as flatpages %}\n        {% get_flatpages for someuser as flatpages %}\n        {% get_flatpages '/about/' as about_pages %}\n        {% get_flatpages prefix as about_pages %}\n        {% get_flatpages '/about/' for someuser as about_pages %}\n    "
    bits = token.split_contents()
    syntax_message = "%(tag_name)s expects a syntax of %(tag_name)s ['url_starts_with'] [for user] as context_name" % {'tag_name': bits[0]}
    if 3 <= len(bits) <= 6:
        if len(bits) % 2 == 0:
            prefix = bits[1]
        else:
            prefix = None
        if bits[-2] != 'as':
            raise template.TemplateSyntaxError(syntax_message)
        context_name = bits[-1]
        if len(bits) >= 5:
            if bits[-4] != 'for':
                raise template.TemplateSyntaxError(syntax_message)
            user = bits[-3]
        else:
            user = None
        return FlatpageNode(context_name, starts_with=prefix, user=user)
    else:
        raise template.TemplateSyntaxError(syntax_message)