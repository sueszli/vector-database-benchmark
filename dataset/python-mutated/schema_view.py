import graphene
from graphene import ObjectType, Schema
from .mutations import PetFormMutation, PetMutation

class QueryRoot(ObjectType):
    thrower = graphene.String(required=True)
    request = graphene.String(required=True)
    test = graphene.String(who=graphene.String())

    def resolve_thrower(self, info):
        if False:
            for i in range(10):
                print('nop')
        raise Exception('Throws!')

    def resolve_request(self, info):
        if False:
            return 10
        return info.context.GET.get('q')

    def resolve_test(self, info, who=None):
        if False:
            for i in range(10):
                print('nop')
        return 'Hello %s' % (who or 'World')

class MutationRoot(ObjectType):
    pet_form_mutation = PetFormMutation.Field()
    pet_mutation = PetMutation.Field()
    write_test = graphene.Field(QueryRoot)

    def resolve_write_test(self, info):
        if False:
            while True:
                i = 10
        return QueryRoot()
schema = Schema(query=QueryRoot, mutation=MutationRoot)