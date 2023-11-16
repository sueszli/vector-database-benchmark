from pytest import raises
import graphene
from graphene import relay

class SomeTypeOne(graphene.ObjectType):
    pass

class SomeTypeTwo(graphene.ObjectType):
    pass

class MyUnion(graphene.Union):

    class Meta:
        types = (SomeTypeOne, SomeTypeTwo)

def test_issue():
    if False:
        for i in range(10):
            print('nop')

    class Query(graphene.ObjectType):
        things = relay.ConnectionField(MyUnion)
    with raises(Exception) as exc_info:
        graphene.Schema(query=Query)
    assert str(exc_info.value) == 'Query fields cannot be resolved. IterableConnectionField type has to be a subclass of Connection. Received "MyUnion".'