from enum import Enum
from django.db import models
from sentry.backup.scopes import RelocationScope
from sentry.db.models import DefaultFieldsModel, FlexibleForeignKey, region_silo_only_model
from sentry.db.models.fields.hybrid_cloud_foreign_key import HybridCloudForeignKey
from sentry.utils.cache import cache
from sentry.utils.hashlib import md5_text

class TransactionMetric(Enum):
    DURATION = 1
    LCP = 2
TRANSACTION_METRICS = {TransactionMetric.DURATION.value: 'duration', TransactionMetric.LCP.value: 'lcp'}
PROJECT_TRANSACTION_THRESHOLD_CACHE_TTL = 3600

def get_project_threshold_cache_key(model_name, project_ids, org_id, order_by, value_list):
    if False:
        i = 10
        return i + 15
    return '{}:{}'.format(model_name, md5_text(f'{project_ids}:{org_id}:{order_by}:{value_list}').hexdigest())

def _filter_and_cache(cls, cache_key, project_ids, organization_id, order_by, value_list):
    if False:
        i = 10
        return i + 15
    cache_result = cache.get(cache_key)
    if cache_result is None:
        result = list(cls.objects.filter(project_id__in=project_ids, organization_id=organization_id).order_by(*order_by).values_list(*value_list))
        cache.set(cache_key, result, PROJECT_TRANSACTION_THRESHOLD_CACHE_TTL)
        return result
    else:
        return cache_result

@region_silo_only_model
class ProjectTransactionThresholdOverride(DefaultFieldsModel):
    __relocation_scope__ = RelocationScope.Excluded
    transaction = models.CharField(max_length=200)
    project = FlexibleForeignKey('sentry.Project', db_constraint=False)
    organization = FlexibleForeignKey('sentry.Organization')
    threshold = models.IntegerField()
    metric = models.PositiveSmallIntegerField(default=TransactionMetric.DURATION.value)
    edited_by_id = HybridCloudForeignKey('sentry.User', null=True, on_delete='SET_NULL')

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_projecttransactionthresholdoverride'
        unique_together = (('project', 'transaction'),)

    @classmethod
    def filter(cls, project_ids, organization_id, order_by, value_list):
        if False:
            for i in range(10):
                print('nop')
        cache_key = get_project_threshold_cache_key('sentry_projecttransactionthresholdoverride', project_ids, organization_id, order_by, value_list)
        return _filter_and_cache(cls, cache_key, project_ids, organization_id, order_by, value_list)

@region_silo_only_model
class ProjectTransactionThreshold(DefaultFieldsModel):
    __relocation_scope__ = RelocationScope.Excluded
    project = FlexibleForeignKey('sentry.Project', unique=True, db_constraint=False)
    organization = FlexibleForeignKey('sentry.Organization')
    threshold = models.IntegerField()
    metric = models.PositiveSmallIntegerField(default=TransactionMetric.DURATION.value)
    edited_by_id = HybridCloudForeignKey('sentry.User', null=True, on_delete='SET_NULL')

    class Meta:
        app_label = 'sentry'
        db_table = 'sentry_projecttransactionthreshold'

    @classmethod
    def filter(cls, project_ids, organization_id, order_by, value_list):
        if False:
            i = 10
            return i + 15
        cache_key = get_project_threshold_cache_key('sentry_projecttransactionthreshold', project_ids, organization_id, order_by, value_list)
        return _filter_and_cache(cls, cache_key, project_ids, organization_id, order_by, value_list)