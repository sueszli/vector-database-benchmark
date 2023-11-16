from django.db import migrations
from sentry.new_migrations.migrations import CheckedMigration
from sentry.utils.query import RangeQuerySetWrapperWithProgressBar

class ObjectStatus:
    VISIBLE = 0
    HIDDEN = 1
    PENDING_DELETION = 2
    DELETION_IN_PROGRESS = 3
    ACTIVE = 0
    DISABLED = 1

def schedule(cls, instance, days=30):
    if False:
        i = 10
        return i + 15
    from datetime import timedelta
    from django.utils import timezone
    model = type(instance)
    model_name = model.__name__
    cls.objects.update_or_create(app_label=instance._meta.app_label, model_name=model_name, object_id=instance.pk, defaults={'actor_id': None, 'data': {}, 'date_scheduled': timezone.now() + timedelta(days=days, hours=0)})

def delete_rules(apps, schema_editor):
    if False:
        i = 10
        return i + 15
    Rule = apps.get_model('sentry', 'Rule')
    RegionScheduledDeletion = apps.get_model('sentry', 'RegionScheduledDeletion')
    for rule in RangeQuerySetWrapperWithProgressBar(Rule.objects.all()):
        if rule.status in (ObjectStatus.PENDING_DELETION, ObjectStatus.DISABLED):
            schedule(RegionScheduledDeletion, rule, days=0)

class Migration(CheckedMigration):
    is_dangerous = True
    dependencies = [('sentry', '0506_null_boolean_fields')]
    operations = [migrations.RunPython(delete_rules, migrations.RunPython.noop, hints={'tables': ['sentry_rule']})]