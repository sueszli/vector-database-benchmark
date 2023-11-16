import graphene

class Query(graphene.ObjectType):
    rand = graphene.String()

class Success(graphene.ObjectType):
    yeah = graphene.String()

class Error(graphene.ObjectType):
    message = graphene.String()

class CreatePostResult(graphene.Union):

    class Meta:
        types = [Success, Error]

class CreatePost(graphene.Mutation):

    class Arguments:
        text = graphene.String(required=True)
    result = graphene.Field(CreatePostResult)

    def mutate(self, info, text):
        if False:
            i = 10
            return i + 15
        result = Success(yeah='yeah')
        return CreatePost(result=result)

class Mutations(graphene.ObjectType):
    create_post = CreatePost.Field()

def test_create_post():
    if False:
        i = 10
        return i + 15
    query_string = '\n    mutation {\n      createPost(text: "Try this out") {\n        result {\n          __typename\n        }\n      }\n    }\n    '
    schema = graphene.Schema(query=Query, mutation=Mutations)
    result = schema.execute(query_string)
    assert not result.errors
    assert result.data['createPost']['result']['__typename'] == 'Success'