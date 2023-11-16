import unittest
from dataclasses import dataclass
from graphql3 import GraphQLBoolean, GraphQLField, GraphQLID, GraphQLNonNull, GraphQLObjectType, GraphQLSchema
from ...generate_taint_models.get_dynamic_graphql_sources import DynamicGraphQLSourceGenerator
from .test_functions import __name__ as qualifier, all_functions

def function1(foo) -> bool:
    if False:
        i = 10
        return i + 15
    return True

def function2(foo, *bar) -> bool:
    if False:
        return 10
    return True

def excluded_function(foo) -> bool:
    if False:
        print('Hello World!')
    return True

class TestClass:

    def method1(self, foo) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

    def method2(self, foo, *bar) -> bool:
        if False:
            return 10
        return True

@dataclass
class DirectObject:
    id: int
    resolver1: bool
    resolver2: bool
    resolver3: bool
    resolver4: bool
    lambda_resolver: bool
queryType = GraphQLObjectType(name='queryType', description='GraphQLObject directly created at top level', fields={'no_resolver': GraphQLField(GraphQLNonNull(GraphQLID)), 'resolver1': GraphQLField(GraphQLBoolean, resolve=function1), 'resolver2': GraphQLField(GraphQLBoolean, resolve=function2), 'resolver3': GraphQLField(GraphQLBoolean, resolve=TestClass.method1), 'resolver4': GraphQLField(GraphQLBoolean, resolve=TestClass.method2), 'lambda_resolver': GraphQLField(GraphQLBoolean, resolve=lambda x: x), 'res': GraphQLField(GraphQLBoolean, resolve=excluded_function)})
SCHEMA = GraphQLSchema(query=queryType)

class GetDynamicGraphQLSourcesTest(unittest.TestCase):

    def test_gather_functions_to_model(self) -> None:
        if False:
            i = 10
            return i + 15
        functions = DynamicGraphQLSourceGenerator(graphql_schema=SCHEMA, graphql_object_type=GraphQLObjectType, resolvers_to_exclude=['tools.pyre.tools.generate_taint_models.tests.get_dynamic_graphql_sources_test.excluded_function']).gather_functions_to_model()
        self.assertTrue(excluded_function not in set(functions))
        self.assertTrue({function1, function2, TestClass.method1, TestClass.method2}.issubset(set(functions)))

    def test_compute_models(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        source = 'TaintSource[UserControlled]'
        sink = 'TaintSink[ReturnedToUser]'
        self.assertEqual([*map(str, DynamicGraphQLSourceGenerator(graphql_schema=SCHEMA, graphql_object_type=GraphQLObjectType, resolvers_to_exclude=['excluded_function']).compute_models(all_functions))], [f'def {qualifier}.TestClass.methodA(self, x) -> {sink}: ...', f'def {qualifier}.TestClass.methodB(self, *args: {source}) -> {sink}: ...', f'def {qualifier}.testA() -> {sink}: ...', f'def {qualifier}.testB(x) -> {sink}: ...', f'def {qualifier}.testC(x) -> {sink}: ...', f'def {qualifier}.testD(x, *args: {source}) -> {sink}: ...', f'def {qualifier}.testE(x, **kwargs: {source}) -> {sink}: ...'])