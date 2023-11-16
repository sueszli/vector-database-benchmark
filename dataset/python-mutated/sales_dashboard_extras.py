import typing
from django.template.defaulttags import register

@register.filter
def get_item(dictionary: dict, key: typing.Any) -> typing.Any:
    if False:
        return 10
    return isinstance(dictionary, dict) and dictionary.get(key)

@register.simple_tag
def query_transform(request, **kwargs):
    if False:
        print('Hello World!')
    '\n    Merges the existing query params with any new ones passed as kwargs.\n\n    Note that we cannot simply use request.GET.update() as that merges lists rather\n    than replacing the value entirely.\n    '
    updated_query_params = request.GET.copy()
    for (key, value) in kwargs.items():
        updated_query_params[key] = value
    return updated_query_params.urlencode()