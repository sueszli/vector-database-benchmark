import logging
import re
from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.snuba.dataset import Dataset, EntityKey
from sentry.snuba.tasks import _create_in_snuba, _delete_from_snuba
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar
CRASH_RATE_ALERT_AGGREGATE_RE = '^percentage\\([ ]*(sessions_crashed|users_crashed)[ ]*\\,[ ]*(sessions|users)[ ]*\\)'

def create_subscription_in_snuba(subscription):
    if False:
        i = 10
        return i + 15
    subscription.subscription_id = _create_in_snuba(subscription)
    subscription.save()

def event_types(self):
    if False:
        while True:
            i = 10
    return [type.event_type for type in self.snubaqueryeventtype_set.all()]

def map_aggregate_to_entity_key(dataset: Dataset, aggregate: str) -> EntityKey:
    if False:
        while True:
            i = 10
    if dataset == Dataset.Events:
        entity_key = EntityKey.Events
    elif dataset == Dataset.Transactions:
        entity_key = EntityKey.Transactions
    elif dataset in [Dataset.Metrics, Dataset.Sessions]:
        match = re.match(CRASH_RATE_ALERT_AGGREGATE_RE, aggregate)
        if not match:
            raise Exception(f'Only crash free percentage queries are supported for subscriptionsover the {dataset.value} dataset')
        if dataset == Dataset.Metrics:
            count_col_matched = match.group(2)
            if count_col_matched == 'sessions':
                entity_key = EntityKey.MetricsCounters
            else:
                entity_key = EntityKey.MetricsSets
        else:
            entity_key = EntityKey.Sessions
    else:
        raise Exception(f'{dataset} dataset does not have an entity key mapped to it')
    return entity_key

def update_metrics_subscriptions(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    QuerySubscription = apps.get_model('sentry', 'QuerySubscription')
    for subscription in RangeQuerySetWrapperWithProgressBar(QuerySubscription.objects.filter(snuba_query__dataset=Dataset.Metrics.value).select_related('snuba_query')):
        old_subscription_id = subscription.subscription_id
        if old_subscription_id is not None:
            try:
                subscription.snuba_query.event_types = property(event_types)
                create_subscription_in_snuba(subscription)
                entity_key: EntityKey = map_aggregate_to_entity_key(Dataset.Metrics, subscription.snuba_query.aggregate)
                _delete_from_snuba(Dataset.Metrics, old_subscription_id, entity_key)
            except Exception:
                logging.exception('Failed to recreate metrics subscription in snuba', extra={'project': subscription.project.slug, 'subscription_id': subscription.id, 'query': subscription.snuba_query.query, 'aggregate': subscription.snuba_query.aggregate, 'time_window': subscription.snuba_query.time_window, 'resolution': subscription.snuba_query.resolution})

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0292_migrate_sessions_subs_user_counts')]
    operations = [migrations.RunPython(update_metrics_subscriptions, migrations.RunPython.noop, hints={'tables': ['sentry_querysubscription', 'sentry_snubaquery']})]