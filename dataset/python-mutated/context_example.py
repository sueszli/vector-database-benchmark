import graphene

class User(graphene.ObjectType):
    id = graphene.ID()
    name = graphene.String()

class Query(graphene.ObjectType):
    me = graphene.Field(User)

    def resolve_me(root, info):
        if False:
            for i in range(10):
                print('nop')
        return info.context['user']
schema = graphene.Schema(query=Query)
query = '\n    query something{\n      me {\n        id\n        name\n      }\n    }\n'

def test_query():
    if False:
        return 10
    result = schema.execute(query, context={'user': User(id='1', name='Syrus')})
    assert not result.errors
    assert result.data == {'me': {'id': '1', 'name': 'Syrus'}}
if __name__ == '__main__':
    result = schema.execute(query, context={'user': User(id='X', name='Console')})
    print(result.data['me'])