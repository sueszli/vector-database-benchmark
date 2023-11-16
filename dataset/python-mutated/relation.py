from django.db.models import F
from common.api import JMSBulkRelationModelViewSet
from .. import serializers
from ..models import User, UserGroup
__all__ = ['UserUserGroupRelationViewSet']

class UserUserGroupRelationViewSet(JMSBulkRelationModelViewSet):
    perm_model = UserGroup
    filterset_fields = ('user', 'usergroup')
    search_fields = filterset_fields
    serializer_class = serializers.User2GroupRelationSerializer
    m2m_field = User.groups.field

    def get_queryset(self):
        if False:
            for i in range(10):
                print('nop')
        return super().get_queryset().annotate(user_display=F('user__name'), usergroup_display=F('usergroup__name'))

    def allow_bulk_destroy(self, qs, filtered):
        if False:
            print('Hello World!')
        if filtered.count() != 1:
            return False
        else:
            return True