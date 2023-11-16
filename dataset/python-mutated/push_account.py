from accounts import serializers
from accounts.const import AutomationTypes
from accounts.models import PushAccountAutomation, ChangeSecretRecord
from orgs.mixins.api import OrgBulkModelViewSet
from .base import AutomationAssetsListApi, AutomationRemoveAssetApi, AutomationAddAssetApi, AutomationNodeAddRemoveApi, AutomationExecutionViewSet
from .change_secret import ChangeSecretRecordViewSet
__all__ = ['PushAccountAutomationViewSet', 'PushAccountAssetsListApi', 'PushAccountRemoveAssetApi', 'PushAccountAddAssetApi', 'PushAccountNodeAddRemoveApi', 'PushAccountExecutionViewSet', 'PushAccountRecordViewSet']

class PushAccountAutomationViewSet(OrgBulkModelViewSet):
    model = PushAccountAutomation
    filter_fields = ('name', 'secret_type', 'secret_strategy')
    search_fields = filter_fields
    serializer_class = serializers.PushAccountAutomationSerializer

class PushAccountExecutionViewSet(AutomationExecutionViewSet):
    rbac_perms = (('list', 'accounts.view_pushaccountexecution'), ('retrieve', 'accounts.view_pushaccountexecution'), ('create', 'accounts.add_pushaccountexecution'))
    tp = AutomationTypes.push_account

    def get_queryset(self):
        if False:
            i = 10
            return i + 15
        queryset = super().get_queryset()
        queryset = queryset.filter(automation__type=self.tp)
        return queryset

class PushAccountRecordViewSet(ChangeSecretRecordViewSet):
    serializer_class = serializers.ChangeSecretRecordSerializer
    tp = AutomationTypes.push_account

    def get_queryset(self):
        if False:
            return 10
        return ChangeSecretRecord.objects.filter(execution__automation__type=AutomationTypes.push_account)

class PushAccountAssetsListApi(AutomationAssetsListApi):
    model = PushAccountAutomation

class PushAccountRemoveAssetApi(AutomationRemoveAssetApi):
    model = PushAccountAutomation
    serializer_class = serializers.PushAccountUpdateAssetSerializer

class PushAccountAddAssetApi(AutomationAddAssetApi):
    model = PushAccountAutomation
    serializer_class = serializers.PushAccountUpdateAssetSerializer

class PushAccountNodeAddRemoveApi(AutomationNodeAddRemoveApi):
    model = PushAccountAutomation
    serializer_class = serializers.PushAccountUpdateNodeSerializer