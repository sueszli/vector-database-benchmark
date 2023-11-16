import logging
import re
from typing import Callable, Iterable, List, Optional
from ...api.connection import PyreConnection
from ...api.query import PyreCache
from .generator_specifications import AnnotationSpecification, default_entrypoint_taint, WhitelistSpecification
from .get_methods_of_subclasses import MethodsOfSubclassesGenerator
from .get_models_filtered_by_callable import ModelsFilteredByCallableGenerator
from .model import PyreFunctionDefinitionModel
from .model_generator import ModelGenerator
LOG: logging.Logger = logging.getLogger(__name__)

class DjangoClassBasedViewModels(ModelGenerator[PyreFunctionDefinitionModel]):

    def __init__(self, pyre_connection: PyreConnection, annotations: Optional[AnnotationSpecification]=None, whitelist: Optional[WhitelistSpecification]=None, pyre_cache: Optional[PyreCache]=None) -> None:
        if False:
            print('Hello World!')
        self.pyre_connection = pyre_connection
        self.pyre_cache = pyre_cache
        self.annotations: AnnotationSpecification = annotations or default_entrypoint_taint
        self.whitelist: WhitelistSpecification = whitelist or WhitelistSpecification(parameter_name={'self', 'cls', 'request'}, parameter_type={'django.http.HttpRequest'})

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            i = 10
            return i + 15
        return []

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> List[PyreFunctionDefinitionModel]:
        if False:
            i = 10
            return i + 15
        pattern: re.Pattern = re.compile('(get|post|put|patch|delete|head|options|trace)')

        def matches_pattern(method: PyreFunctionDefinitionModel) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return bool(pattern.search(method.callable_name))
        return list(ModelsFilteredByCallableGenerator(generator_to_filter=MethodsOfSubclassesGenerator(base_classes=['django.views.generic.base.View'], pyre_connection=self.pyre_connection, pyre_cache=self.pyre_cache, annotations=self.annotations, whitelist=self.whitelist), filter=matches_pattern).generate_models())