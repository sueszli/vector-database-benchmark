import unittest
from pulumi._types import resource_types
import pulumi

class Resource1(pulumi.Resource):
    pass

class Resource2(pulumi.Resource):
    foo: pulumi.Output[str]

class Resource3(pulumi.Resource):
    nested: pulumi.Output['Nested']

class Resource4(pulumi.Resource):
    nested_value: pulumi.Output['Nested'] = pulumi.property('nestedValue')

class Resource5(pulumi.Resource):

    @property
    @pulumi.getter
    def foo(self) -> pulumi.Output[str]:
        if False:
            return 10
        ...

class Resource6(pulumi.Resource):

    @property
    @pulumi.getter
    def nested(self) -> pulumi.Output['Nested']:
        if False:
            i = 10
            return i + 15
        ...

class Resource7(pulumi.Resource):

    @property
    @pulumi.getter(name='nestedValue')
    def nested_value(self) -> pulumi.Output['Nested']:
        if False:
            while True:
                i = 10
        ...

class Resource8(pulumi.Resource):
    foo: pulumi.Output

class Resource9(pulumi.Resource):

    @property
    @pulumi.getter
    def foo(self) -> pulumi.Output:
        if False:
            for i in range(10):
                print('nop')
        ...

class Resource10(pulumi.Resource):
    foo: str

class Resource11(pulumi.Resource):

    @property
    @pulumi.getter
    def foo(self) -> str:
        if False:
            return 10
        ...

class Resource12(pulumi.Resource):

    @property
    @pulumi.getter
    def foo(self):
        if False:
            print('Hello World!')
        ...

@pulumi.output_type
class Nested:
    first: str
    second: str

class ResourceTypesTests(unittest.TestCase):

    def test_resource_types(self):
        if False:
            while True:
                i = 10
        self.assertEqual({}, resource_types(Resource1))
        self.assertEqual({'foo': str}, resource_types(Resource2))
        self.assertEqual({'nested': Nested}, resource_types(Resource3))
        self.assertEqual({'nestedValue': Nested}, resource_types(Resource4))
        self.assertEqual({'foo': str}, resource_types(Resource5))
        self.assertEqual({'nested': Nested}, resource_types(Resource6))
        self.assertEqual({'nestedValue': Nested}, resource_types(Resource7))
        self.assertEqual({}, resource_types(Resource8))
        self.assertEqual({}, resource_types(Resource9))
        self.assertEqual({'foo': str}, resource_types(Resource10))
        self.assertEqual({'foo': str}, resource_types(Resource11))
        self.assertEqual({}, resource_types(Resource12))