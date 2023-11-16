import os
import unittest
from typing import Callable
from graphql3 import GraphQLBoolean, GraphQLField, GraphQLID, GraphQLNonNull, GraphQLObjectType, GraphQLType
from ...generate_taint_models.get_graphql_sources import GraphQLSourceGenerator
from .test_functions import __name__ as qualifier, all_functions

class GetGraphQLSourcesTest(unittest.TestCase):

    def test_gather_functions_to_model(self) -> None:
        if False:
            i = 10
            return i + 15
        functions = GraphQLSourceGenerator(graphql_module='tools.pyre.tools.generate_taint_models.tests', graphql_object_type=GraphQLObjectType).gather_functions_to_model()
        self.assertTrue({function_1, function_2}.issubset(set(functions)))
        functions = GraphQLSourceGenerator(graphql_module=['tools.pyre.tools.generate_taint_models.tests'], graphql_object_type=GraphQLObjectType).gather_functions_to_model()
        self.assertTrue({function_1, function_2}.issubset(set(functions)))

    def test_compute_models(self) -> None:
        if False:
            while True:
                i = 10
        source = 'TaintSource[UserControlled]'
        sink = 'TaintSink[ReturnedToUser]'
        self.assertEqual([*map(str, GraphQLSourceGenerator(graphql_module='tools.pyre.tools.generate_taint_models.tests', graphql_object_type=GraphQLObjectType).compute_models(all_functions))], [f'def {qualifier}.TestClass.methodA(self, x) -> {sink}: ...', f'def {qualifier}.TestClass.methodB(self, *args: {source}) -> {sink}: ...', f'def {qualifier}.testA() -> {sink}: ...', f'def {qualifier}.testB(x) -> {sink}: ...', f'def {qualifier}.testC(x) -> {sink}: ...', f'def {qualifier}.testD(x, *args: {source}) -> {sink}: ...', f'def {qualifier}.testE(x, **kwargs: {source}) -> {sink}: ...'])

def function_1() -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def function_2() -> None:
    if False:
        while True:
            i = 10
    pass
DirectObjectType = GraphQLObjectType(name='DirectObjectType', description='GraphQLObject directly created at top level', fields={'no_resolver': GraphQLField(GraphQLNonNull(GraphQLID)), 'resolver': GraphQLField(GraphQLBoolean, resolve=function_1), 'lambda_resolver': GraphQLField(GraphQLBoolean, resolve=lambda x: x)})
BrokenObjectType = GraphQLObjectType(name='BrokenObjectType', description='Look ma, no fields', fields={})

def add_field(type: GraphQLType, name: str, resolve: Callable) -> None:
    if False:
        print('Hello World!')
    type._fields[name] = GraphQLField(GraphQLNonNull(GraphQLID), resolve=resolve)
IndirectObjectType = add_field(type=DirectObjectType, name='indirect', resolve=function_2)