import graphene

class Query(graphene.ObjectType):
    some_field = graphene.String(from_=graphene.String(name='from'))

    def resolve_some_field(self, info, from_=None):
        if False:
            print('Hello World!')
        return from_

def test_issue():
    if False:
        for i in range(10):
            print('nop')
    query_string = '\n    query myQuery {\n      someField(from: "Oh")\n    }\n    '
    schema = graphene.Schema(query=Query)
    result = schema.execute(query_string)
    assert not result.errors
    assert result.data['someField'] == 'Oh'