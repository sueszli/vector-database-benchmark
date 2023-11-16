import graphene

class MyInputClass(graphene.InputObjectType):

    @classmethod
    def __init_subclass_with_meta__(cls, container=None, _meta=None, fields=None, **options):
        if False:
            print('Hello World!')
        if _meta is None:
            _meta = graphene.types.inputobjecttype.InputObjectTypeOptions(cls)
        _meta.fields = fields
        super(MyInputClass, cls).__init_subclass_with_meta__(container=container, _meta=_meta, **options)

class MyInput(MyInputClass):

    class Meta:
        fields = dict(x=graphene.Field(graphene.Int))

class Query(graphene.ObjectType):
    myField = graphene.Field(graphene.String, input=graphene.Argument(MyInput))

    def resolve_myField(parent, info, input):
        if False:
            while True:
                i = 10
        return 'ok'

def test_issue():
    if False:
        while True:
            i = 10
    query_string = '\n    query myQuery {\n      myField(input: {x: 1})\n    }\n    '
    schema = graphene.Schema(query=Query)
    result = schema.execute(query_string)
    assert not result.errors