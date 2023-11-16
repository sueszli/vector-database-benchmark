from django.db import IntegrityError, migrations
from django.utils.text import slugify
from sentry.constants import ObjectStatus
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

def schedule(cls, instance, days=30):
    if False:
        print('Hello World!')
    from datetime import timedelta
    from django.utils import timezone
    model = type(instance)
    model_name = model.__name__
    cls.objects.update_or_create(app_label=instance._meta.app_label, model_name=model_name, object_id=instance.pk, defaults={'actor_id': None, 'data': {}, 'date_scheduled': timezone.now() + timedelta(days=days, hours=0)})

def migrate_monitor_slugs(apps, schema_editor):
    if False:
        while True:
            i = 10
    Monitor = apps.get_model('sentry', 'Monitor')
    Rule = apps.get_model('sentry', 'Rule')
    RegionScheduledDeletion = apps.get_model('sentry', 'RegionScheduledDeletion')
    ScheduledDeletion = apps.get_model('sentry', 'ScheduledDeletion')
    MAX_SLUG_LENGTH = 50
    for monitor in RangeQuerySetWrapperWithProgressBar(Monitor.objects.all()):
        monitor_slug = monitor.slug
        slugified = slugify(monitor_slug)[:MAX_SLUG_LENGTH].strip('-')
        if monitor_slug == slugified:
            continue
        try:
            monitor.slug = slugified
            monitor.save()
        except IntegrityError:
            alert_rule_id = monitor.config.get('alert_rule_id')
            if alert_rule_id:
                rule = Rule.objects.filter(project_id=monitor.project_id, id=alert_rule_id).exclude(status__in=[ObjectStatus.PENDING_DELETION, ObjectStatus.DELETION_IN_PROGRESS]).first()
                if rule:
                    rule.status = ObjectStatus.PENDING_DELETION
                    rule.save()
                    schedule(RegionScheduledDeletion, rule, days=0)
            monitor.slug = monitor_slug
            monitor.status = ObjectStatus.PENDING_DELETION
            monitor.save()
            schedule(ScheduledDeletion, monitor, days=0)

class Migration(CheckedMigration):
    is_dangerous = False
    dependencies = [('sentry', '0514_migrate_priority_saved_searches')]
    operations = [migrations.RunPython(migrate_monitor_slugs, migrations.RunPython.noop, hints={'tables': ['sentry_monitor']})]