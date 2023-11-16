import logging
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.snuba.dataset import Dataset, EntityKey
from sentry.snuba.entity_subscription import get_entity_key_from_snuba_query
from sentry.snuba.tasks import _create_in_snuba, _delete_from_snuba
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def create_subscription_in_snuba(subscription):
    if False:
        i = 10
        return i + 15
    subscription.subscription_id = _create_in_snuba(subscription)
    subscription.save()

def event_types(self):
    if False:
        return 10
    return [type.event_type for type in self.snubaqueryeventtype_set.all()]

def update_performance_subscriptions(apps, schema_editor):
    if False:
        while True:
            i = 10
    QuerySubscription = apps.get_model('sentry', 'QuerySubscription')
    for subscription in RangeQuerySetWrapperWithProgressBar(QuerySubscription.objects.filter(snuba_query__dataset=Dataset.PerformanceMetrics.value, snuba_query__environment_id__isnull=False, status=0).select_related('snuba_query', 'project')):
        old_subscription_id = subscription.subscription_id
        if old_subscription_id is not None:
            try:
                subscription.snuba_query.event_types = property(event_types)
                create_subscription_in_snuba(subscription)
                entity_key: EntityKey = get_entity_key_from_snuba_query(subscription.snuba_query, subscription.project.organization_id, subscription.project_id)
                _delete_from_snuba(Dataset.PerformanceMetrics, old_subscription_id, entity_key)
            except Exception:
                logging.exception('Failed to recreate performance subscription in snuba', extra={'project': subscription.project.slug, 'subscription_id': subscription.id, 'query': subscription.snuba_query.query, 'aggregate': subscription.snuba_query.aggregate, 'time_window': subscription.snuba_query.time_window, 'resolution': subscription.snuba_query.resolution})

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0406_monitor_cleanup')]
    operations = [migrations.RunPython(update_performance_subscriptions, migrations.RunPython.noop, hints={'tables': ['sentry_querysubscription', 'sentry_snubaquery']})]