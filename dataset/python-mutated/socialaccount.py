from django import template
from django.utils.safestring import mark_safe
from allauth.socialaccount.adapter import get_adapter
from allauth.utils import get_request_param
register = template.Library()

@register.simple_tag(takes_context=True)
def provider_login_url(context, provider, **params):
    if False:
        return 10
    '\n    {% provider_login_url "facebook" next=bla %}\n    {% provider_login_url "openid" openid="http://me.yahoo.com" next=bla %}\n    '
    request = context.get('request')
    if isinstance(provider, str):
        adapter = get_adapter()
        provider = adapter.get_provider(request, provider)
    query = dict(params)
    auth_params = query.get('auth_params', None)
    scope = query.get('scope', None)
    process = query.get('process', None)
    if scope == '':
        del query['scope']
    if auth_params == '':
        del query['auth_params']
    if 'next' not in query:
        next = get_request_param(request, 'next')
        if next:
            query['next'] = next
        elif process == 'redirect':
            query['next'] = request.get_full_path()
    elif not query['next']:
        del query['next']
    return provider.get_login_url(request, **query)

@register.simple_tag(takes_context=True)
def providers_media_js(context):
    if False:
        i = 10
        return i + 15
    request = context['request']
    providers = get_adapter().list_providers(request)
    ret = '\n'.join((p.media_js(request) for p in providers))
    return mark_safe(ret)

@register.simple_tag
def get_social_accounts(user):
    if False:
        while True:
            i = 10
    '\n    {% get_social_accounts user as accounts %}\n\n    Then:\n        {{accounts.twitter}} -- a list of connected Twitter accounts\n        {{accounts.twitter.0}} -- the first Twitter account\n        {% if accounts %} -- if there is at least one social account\n    '
    accounts = {}
    for account in user.socialaccount_set.all().iterator():
        providers = accounts.setdefault(account.provider, [])
        providers.append(account)
    return accounts

@register.simple_tag(takes_context=True)
def get_providers(context):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a list of social authentication providers.\n\n    Usage: `{% get_providers as socialaccount_providers %}`.\n\n    Then within the template context, `socialaccount_providers` will hold\n    a list of social providers configured for the current site.\n    '
    request = context['request']
    adapter = get_adapter()
    providers = adapter.list_providers(request)
    return sorted(providers, key=lambda p: p.name)