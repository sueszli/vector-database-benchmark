import datetime
import logging
import django.core.serializers.json
from django.conf import settings
from django.db import migrations, models
from django.utils import timezone, dateparse
logger = logging.getLogger('wagtail.migrations')

def legacy_to_iso_format(date_string, tz=None):
    if False:
        return 10
    dt = datetime.datetime.strptime(date_string, '%d %b %Y %H:%M')
    if settings.USE_TZ:
        dt = timezone.make_aware(dt, datetime.timezone.utc if tz is None else tz)
        dt = timezone.localtime(dt, datetime.timezone.utc)
    return dt

def iso_to_legacy_format(date_string, tz=None):
    if False:
        i = 10
        return i + 15
    dt = dateparse.parse_datetime(date_string)
    if dt is None:
        raise ValueError("date isn't well formatted")
    if settings.USE_TZ:
        dt = timezone.localtime(dt, datetime.timezone.utc if tz is None else tz)
    return dt.strftime('%d %b %Y %H:%M')

def migrate_logs_with_created_only(model, converter):
    if False:
        for i in range(10):
            print('nop')
    for item in model.objects.filter(action__in=['wagtail.revert', 'wagtail.rename', 'wagtail.publish']).only('data').iterator():
        try:
            created = item.data['revision']['created']
            item.data['revision']['created'] = converter(created)
        except ValueError:
            logger.warning("Failed to migrate 'created' timestamp '%s' of %s %s (%s)", item.data['revision']['created'], model.__name__, item.pk, converter.__name__)
            continue
        except (KeyError, TypeError):
            continue
        else:
            item.save(update_fields=['data'])

def migrate_schedule_logs(model, converter):
    if False:
        while True:
            i = 10
    for item in model.objects.filter(action__in=['wagtail.publish.schedule', 'wagtail.schedule.cancel']).only('data').iterator():
        changed = False
        try:
            created = item.data['revision']['created']
            item.data['revision']['created'] = converter(created)
            changed = True
        except ValueError:
            logger.warning("Failed to migrate 'created' timestamp '%s' of %s %s (%s)", created, model.__name__, item.pk, converter.__name__)
        except (KeyError, TypeError):
            pass
        try:
            go_live_at = item.data['revision'].get('go_live_at')
            if go_live_at:
                item.data['revision']['go_live_at'] = converter(go_live_at, tz=timezone.get_default_timezone())
                changed = True
        except ValueError:
            logger.warning("Failed to migrate 'go_live_at' timestamp '%s' of %s %s (%s)", go_live_at, model.__name__, item.pk, converter.__name__)
        except (KeyError, TypeError):
            pass
        if changed:
            item.save(update_fields=['data'])

def migrate_custom_to_iso_format(apps, schema_editor):
    if False:
        while True:
            i = 10
    ModelLogEntry = apps.get_model('wagtailcore.ModelLogEntry')
    PageLogEntry = apps.get_model('wagtailcore.PageLogEntry')
    migrate_logs_with_created_only(ModelLogEntry, legacy_to_iso_format)
    migrate_logs_with_created_only(PageLogEntry, legacy_to_iso_format)
    migrate_schedule_logs(ModelLogEntry, legacy_to_iso_format)
    migrate_schedule_logs(PageLogEntry, legacy_to_iso_format)

def migrate_iso_to_custom_format(apps, schema_editor):
    if False:
        return 10
    ModelLogEntry = apps.get_model('wagtailcore.ModelLogEntry')
    PageLogEntry = apps.get_model('wagtailcore.PageLogEntry')
    migrate_logs_with_created_only(ModelLogEntry, iso_to_legacy_format)
    migrate_logs_with_created_only(PageLogEntry, iso_to_legacy_format)
    migrate_schedule_logs(ModelLogEntry, iso_to_legacy_format)
    migrate_schedule_logs(PageLogEntry, iso_to_legacy_format)

class Migration(migrations.Migration):
    dependencies = [('wagtailcore', '0087_alter_grouppagepermission_unique_together_and_more')]
    operations = [migrations.AlterField(model_name='modellogentry', name='data', field=models.JSONField(blank=True, default=dict, encoder=django.core.serializers.json.DjangoJSONEncoder)), migrations.AlterField(model_name='pagelogentry', name='data', field=models.JSONField(blank=True, default=dict, encoder=django.core.serializers.json.DjangoJSONEncoder)), migrations.RunPython(migrate_custom_to_iso_format, migrate_iso_to_custom_format)]