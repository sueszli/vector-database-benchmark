from typing import Callable, Iterable, List, Optional
from .inspect_parser import extract_qualified_name
from .model import CallableModel
from .model_generator import ModelGenerator
from .view_generator import DjangoUrls, get_all_views

class ExitNodeGenerator(ModelGenerator[CallableModel]):

    def __init__(self, django_urls: DjangoUrls, whitelisted_views: Optional[List[str]]=None, taint_annotation: str='TaintSink[ReturnedToUser]') -> None:
        if False:
            print('Hello World!')
        self.django_urls = django_urls
        self.whitelisted_views: List[str] = whitelisted_views or []
        self.taint_annotation: str = taint_annotation

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            for i in range(10):
                print('nop')
        return get_all_views(self.django_urls)

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> Iterable[CallableModel]:
        if False:
            print('Hello World!')
        exit_nodes = set()
        for view_function in functions_to_model:
            qualified_name = extract_qualified_name(view_function)
            if qualified_name in self.whitelisted_views:
                continue
            try:
                model = CallableModel(returns=self.taint_annotation, callable_object=view_function)
                exit_nodes.add(model)
            except ValueError:
                pass
        return sorted(exit_nodes)