import os
from importlib import import_module
from typing import Any, Callable, Iterable, List, Type, Union
from .generator_specifications import AllParametersAnnotation
from .model import CallableModel
from .model_generator import ModelGenerator
GraphQLObjectType = Type[Any]

class GraphQLSourceGenerator(ModelGenerator[CallableModel]):

    def __init__(self, graphql_module: Union[List[str], str], graphql_object_type: GraphQLObjectType, args_taint_annotation: str='TaintSource[UserControlled]', return_taint_annotation: str='TaintSink[ReturnedToUser]') -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.graphql_module: Union[List[str], str] = graphql_module
        self.graphql_object_type: GraphQLObjectType = graphql_object_type
        self.args_taint_annotation: str = args_taint_annotation
        self.return_taint_annotation: str = return_taint_annotation

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            i = 10
            return i + 15
        views: List[Callable[..., object]] = []
        modules = []
        module_argument = self.graphql_module
        graphql_modules = [module_argument] if isinstance(module_argument, str) else module_argument
        for graphql_module in graphql_modules:
            for path in os.listdir(os.path.dirname(import_module(graphql_module).__file__)):
                if path.endswith('.py') and path != '__init__.py':
                    modules.append(f'{graphql_module}.{path[:-3]}')

            def visit_all_graphql_resolvers(module_name: str) -> None:
                if False:
                    while True:
                        i = 10
                module = import_module(module_name)
                for key in module.__dict__:
                    element = module.__dict__[key]
                    if not isinstance(element, self.graphql_object_type):
                        continue
                    try:
                        fields = element.fields
                    except AssertionError:
                        fields = []
                    for field in fields:
                        resolver = fields[field].resolve
                        if resolver is not None and resolver.__name__ != '<lambda>':
                            views.append(resolver)
            for module_name in modules:
                visit_all_graphql_resolvers(module_name)
        return views

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> Iterable[CallableModel]:
        if False:
            while True:
                i = 10
        graphql_models = set()
        for view_function in functions_to_model:
            try:
                model = CallableModel(callable_object=view_function, parameter_annotation=AllParametersAnnotation(vararg=self.args_taint_annotation, kwarg=self.args_taint_annotation), returns=self.return_taint_annotation)
                graphql_models.add(model)
            except ValueError:
                pass
        return sorted(graphql_models)