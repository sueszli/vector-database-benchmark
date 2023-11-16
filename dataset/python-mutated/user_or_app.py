import graphene
from ....account import models as account_models
from ....app import models as app_models
from ...account import types as account_types
from ...app import types as app_types

class UserOrApp(graphene.Union):

    class Meta:
        types = (account_types.User, app_types.App)

    @classmethod
    def resolve_type(cls, instance, info):
        if False:
            while True:
                i = 10
        if isinstance(instance, account_models.User):
            return account_types.User
        if isinstance(instance, app_models.App):
            return account_types.App
        return super().resolve_type(instance, info)