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
        for i in range(10):
            print('nop')
    subscription.subscription_id = _create_in_snuba(subscription)
    subscription.save()

def map_aggregate_to_entity_key(dataset: Dataset, aggregate: str) -> EntityKey:
    if False:
        for i in range(10):
            print('nop')
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

def delete_subscription_from_snuba(subscription, query_dataset: Dataset):
    if False:
        for i in range(10):
            print('nop')
    entity_key: EntityKey = map_aggregate_to_entity_key(query_dataset, subscription.snuba_query.aggregate)
    _delete_from_snuba(query_dataset, subscription.subscription_id, entity_key)

def event_types(self):
    if False:
        print('Hello World!')
    return [type.event_type for type in self.snubaqueryeventtype_set.all()]

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
                delete_subscription_from_snuba(subscription, Dataset.Metrics)
            except Exception:
                logging.exception('Failed to recreate metrics subscription in snuba', extra={'project': subscription.project.slug, 'subscription_id': subscription.id, 'query': subscription.snuba_query.query, 'aggregate': subscription.snuba_query.aggregate, 'time_window': subscription.snuba_query.time_window, 'resolution': subscription.snuba_query.resolution})

class Migration(CheckedMigration):
    is_dangerous = False
    atomic = False
    dependencies = [('sentry', '0291_add_new_perf_indexer')]
    operations = [migrations.RunPython(update_metrics_subscriptions, migrations.RunPython.noop, hints={'tables': ['sentry_querysubscription', 'sentry_snubaquery']})]