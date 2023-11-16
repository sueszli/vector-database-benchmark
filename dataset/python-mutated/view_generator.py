import inspect
import logging
from importlib import import_module
from typing import Any, Callable, Iterable, List, NamedTuple, Type
LOG: logging.Logger = logging.getLogger(__name__)
DynamicURLType = Type[Any]

class DjangoUrls(NamedTuple):
    urls_module: str
    url_resolver_type: DynamicURLType
    url_pattern_type: DynamicURLType

def get_all_views(django_urls: DjangoUrls) -> List[Callable[..., object]]:
    if False:
        return 10
    LOG.info(f'Getting all URLs from `{django_urls.urls_module}`')
    imported_urls_module = import_module(django_urls.urls_module)
    functions_to_model = []

    def visit_all_patterns(url_patterns: Iterable[Any]) -> None:
        if False:
            i = 10
            return i + 15
        for pattern in url_patterns:
            if isinstance(pattern, django_urls.url_resolver_type):
                visit_all_patterns(pattern.url_patterns)
            elif isinstance(pattern, django_urls.url_pattern_type):
                callback = pattern.callback
                if inspect.ismethod(callback) or inspect.isfunction(callback):
                    functions_to_model.append(callback)
                elif hasattr(callback, '__call__'):
                    functions_to_model.append(callback.__call__)
                else:
                    raise TypeError('callback is not a function, method, or functor')
            else:
                raise TypeError('pattern is not url resolver or url pattern.')
    visit_all_patterns(imported_urls_module.urlpatterns)
    return functions_to_model