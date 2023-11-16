from typing import Any, Callable, Iterable, List, Optional, Type
try:
    from graphql3 import GraphQLSchema
except ModuleNotFoundError:
    from graphql import GraphQLSchema
from .function_tainter import taint_callable_functions
from .generator_specifications import AllParametersAnnotation, AnnotationSpecification
from .model import CallableModel
from .model_generator import ModelGenerator
GraphQLObjectType = Type[Any]

class DynamicGraphQLSourceGenerator(ModelGenerator[CallableModel]):

    def __init__(self, graphql_schema: GraphQLSchema, graphql_object_type: GraphQLObjectType, annotations: Optional[AnnotationSpecification]=None, resolvers_to_exclude: Optional[List[str]]=None) -> None:
        if False:
            while True:
                i = 10
        super().__init__()
        self.graphql_schema: GraphQLSchema = graphql_schema
        self.graphql_object_type: GraphQLObjectType = graphql_object_type
        self.annotations: AnnotationSpecification = annotations or AnnotationSpecification(parameter_annotation=AllParametersAnnotation(vararg='TaintSource[UserControlled]', kwarg='TaintSource[UserControlled]'), returns='TaintSink[ReturnedToUser]')
        self.resolvers_to_exclude: List[str] = resolvers_to_exclude or []

    def gather_functions_to_model(self) -> Iterable[Callable[..., object]]:
        if False:
            print('Hello World!')
        type_map = self.graphql_schema.type_map
        resolvers: List[Callable[..., object]] = []
        for element in type_map.values():
            if not isinstance(element, self.graphql_object_type):
                continue
            try:
                fields = element.fields
            except AssertionError:
                continue
            for field in fields:
                resolver = fields[field].resolve
                if resolver is not None and resolver.__name__ != '<lambda>' and (f'{resolver.__module__}.{resolver.__name__}' not in self.resolvers_to_exclude):
                    resolvers.append(resolver)
        return resolvers

    def compute_models(self, functions_to_model: Iterable[Callable[..., object]]) -> Iterable[CallableModel]:
        if False:
            print('Hello World!')
        return taint_callable_functions(functions_to_model, annotations=self.annotations)