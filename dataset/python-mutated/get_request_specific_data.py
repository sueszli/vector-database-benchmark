from typing import Callable, Iterable, List, Optional
from .function_tainter import taint_callable_functions
from .model import CallableModel
from .model_generator import ModelGenerator
from .view_generator import DjangoUrls, get_all_views

class RequestSpecificDataGenerator(ModelGenerator[CallableModel]):

    def __init__(self, django_urls: DjangoUrls, whitelisted_views: Optional[List[str]]=None, whitelisted_classes: Optional[List[str]]=None) -> None:
        if False:
            return 10
        self.django_urls: DjangoUrls = django_urls
        self.whitelisted_views: List[str] = whitelisted_views or []
        self.whitelisted_classes: List[str] = whitelisted_classes or []

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            for i in range(10):
                print('nop')
        django_urls = self.django_urls
        if django_urls is None:
            return []
        return get_all_views(django_urls)

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> List[CallableModel]:
        if False:
            print('Hello World!')
        taint_annotation = 'TaintSource[RequestSpecificData]'
        return taint_callable_functions(functions_to_model, taint_annotation=taint_annotation, whitelisted_views=self.whitelisted_views, whitelisted_classes=self.whitelisted_classes)