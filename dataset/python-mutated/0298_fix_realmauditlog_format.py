import json
from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

def update_realmauditlog_values(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    This migration fixes two issues with the RealmAuditLog format for certain event types:\n    * The notifications_stream and signup_notifications_stream fields had the\n      Stream objects passed into `ujson.dumps()` and thus marshalled as a giant\n      JSON object, when the intent was to store the stream ID.\n    * The default_sending_stream would also been marshalled wrong, but are part\n      of a feature that nobody should be using, so we simply assert that\'s the case.\n    * Changes the structure of the extra_data JSON dictionaries for those\n      RealmAuditLog entries with a sub-property field from:\n      {\n          OLD_VALUE: {"property": property, "value": old_value},\n          NEW_VALUE: {"property": property, "value": new_value},\n      }\n\n      to the more natural:\n\n      {\n          OLD_VALUE: old_value,\n          NEW_VALUE: new_value,\n          "property": property,\n      }\n    '
    RealmAuditLog = apps.get_model('zerver', 'RealmAuditLog')
    USER_DEFAULT_SENDING_STREAM_CHANGED = 129
    USER_DEFAULT_REGISTER_STREAM_CHANGED = 130
    USER_DEFAULT_ALL_PUBLIC_STREAMS_CHANGED = 131
    USER_NOTIFICATION_SETTINGS_CHANGED = 132
    REALM_PROPERTY_CHANGED = 207
    SUBSCRIPTION_PROPERTY_CHANGED = 304
    OLD_VALUE = '1'
    NEW_VALUE = '2'
    unlikely_event_types = [USER_DEFAULT_SENDING_STREAM_CHANGED, USER_DEFAULT_REGISTER_STREAM_CHANGED, USER_DEFAULT_ALL_PUBLIC_STREAMS_CHANGED]
    affected_event_types = [REALM_PROPERTY_CHANGED, USER_NOTIFICATION_SETTINGS_CHANGED, SUBSCRIPTION_PROPERTY_CHANGED]
    improperly_marshalled_properties = ['notifications_stream', 'signup_notifications_stream']
    assert not RealmAuditLog.objects.filter(event_type__in=unlikely_event_types).exists()
    for ra in RealmAuditLog.objects.filter(event_type__in=affected_event_types):
        extra_data = json.loads(ra.extra_data)
        old_key = extra_data[OLD_VALUE]
        new_key = extra_data[NEW_VALUE]
        if not isinstance(old_key, dict) and (not isinstance(new_key, dict)):
            continue
        if 'value' not in old_key or 'value' not in new_key:
            continue
        old_value = old_key['value']
        new_value = new_key['value']
        prop = old_key['property']
        if prop != 'authentication_methods':
            if isinstance(old_value, dict):
                assert prop in improperly_marshalled_properties
                old_value = old_value['id']
            if isinstance(new_value, dict):
                assert prop in improperly_marshalled_properties
                new_value = new_value['id']
        assert set(extra_data.keys()) <= {OLD_VALUE, NEW_VALUE}
        ra.extra_data = json.dumps({OLD_VALUE: old_value, NEW_VALUE: new_value, 'property': prop})
        ra.save(update_fields=['extra_data'])

class Migration(migrations.Migration):
    dependencies = [('zerver', '0297_draft')]
    operations = [migrations.RunPython(update_realmauditlog_values, reverse_code=migrations.RunPython.noop, elidable=True)]