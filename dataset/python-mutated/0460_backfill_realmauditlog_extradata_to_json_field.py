import ast
from typing import List, Tuple, Type
import orjson
from django.db import migrations, transaction
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps
from django.db.models import F, JSONField, Model
from django.db.models.functions import Cast, JSONObject
OLD_VALUE = '1'
NEW_VALUE = '2'
USER_FULL_NAME_CHANGED = 124
REALM_DISCOUNT_CHANGED = 209
BATCH_SIZE = 5000
DISCOUNT_DATA_TEMPLATE = 'Audit log entry {id} with event type REALM_DISCOUNT_CHANGED is skipped.\nThe data consistency needs to be manually checked.\n  Discount data to remove after the upcoming JSONField migration:\n{data_to_remove}\n  Discount data to keep after the upcoming JSONField migration:\n{data_to_keep}\n'
OVERWRITE_TEMPLATE = 'Audit log entry with id {id} has extra_data_json been inconsistently overwritten.\n  The old value is:\n{old_value}\n  The new value is:\n{new_value}\n'

@transaction.atomic
def do_bulk_backfill_extra_data(audit_log_model: Type[Model], id_lower_bound: int, id_upper_bound: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    audit_log_model._default_manager.filter(event_type=USER_FULL_NAME_CHANGED, id__range=(id_lower_bound, id_upper_bound), extra_data_json={}).update(extra_data_json=JSONObject(**{OLD_VALUE: 'extra_data', NEW_VALUE: None}))
    inconsistent_extra_data_json: List[Tuple[int, str, object, object]] = []
    inconsistent_extra_data_json.extend(audit_log_model._default_manager.filter(extra_data__isnull=False, id__range=(id_lower_bound, id_upper_bound)).annotate(new_extra_data_json=Cast('extra_data', output_field=JSONField())).exclude(extra_data__startswith="{'").exclude(event_type=USER_FULL_NAME_CHANGED).exclude(extra_data_json={}).exclude(extra_data_json=F('new_extra_data_json')).values_list('id', 'extra_data', 'extra_data_json', 'new_extra_data_json'))
    audit_log_model._default_manager.filter(extra_data__isnull=False, id__range=(id_lower_bound, id_upper_bound), extra_data_json__inconsistent_old_extra_data__isnull=True).exclude(extra_data__startswith="{'").exclude(event_type=USER_FULL_NAME_CHANGED).update(extra_data_json=Cast('extra_data', output_field=JSONField()))
    python_valued_audit_log_entries = audit_log_model._default_manager.filter(extra_data__startswith="{'", id__range=(id_lower_bound, id_upper_bound), extra_data_json__inconsistent_old_extra_data__isnull=True)
    for audit_log_entry in python_valued_audit_log_entries:
        old_value = audit_log_entry.extra_data_json
        if audit_log_entry.event_type == REALM_DISCOUNT_CHANGED:
            print(DISCOUNT_DATA_TEMPLATE.format(id=audit_log_entry.id, data_to_remove=audit_log_entry.extra_data, data_to_keep=old_value))
            continue
        new_value = ast.literal_eval(audit_log_entry.extra_data)
        if old_value not in ({}, new_value):
            inconsistent_extra_data_json.append((audit_log_entry.id, audit_log_entry.extra_data, old_value, new_value))
        audit_log_entry.extra_data_json = new_value
    audit_log_model._default_manager.bulk_update(python_valued_audit_log_entries, fields=['extra_data_json'])
    if inconsistent_extra_data_json:
        audit_log_entries = []
        for (audit_log_entry_id, old_extra_data, old_extra_data_json, new_extra_data_json) in inconsistent_extra_data_json:
            audit_log_entry = audit_log_model._default_manager.get(id=audit_log_entry_id)
            assert isinstance(old_extra_data_json, dict)
            if 'inconsistent_old_extra_data' in old_extra_data_json:
                continue
            assert isinstance(new_extra_data_json, dict)
            audit_log_entry.extra_data_json = {**new_extra_data_json, 'inconsistent_old_extra_data': old_extra_data, 'inconsistent_old_extra_data_json': old_extra_data_json}
            audit_log_entries.append(audit_log_entry)
            print(OVERWRITE_TEMPLATE.format(id=audit_log_entry_id, old_value=orjson.dumps(old_extra_data_json).decode(), new_value=orjson.dumps(new_extra_data_json).decode()))
        audit_log_model._default_manager.bulk_update(audit_log_entries, fields=['extra_data_json'])

def backfill_extra_data(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if False:
        while True:
            i = 10
    audit_log_model = apps.get_model('zerver', 'RealmAuditLog')
    if not audit_log_model.objects.filter(extra_data__isnull=False).exists():
        return
    audit_log_entries = audit_log_model.objects.filter(extra_data__isnull=False)
    id_lower_bound = audit_log_entries.earliest('id').id
    id_upper_bound = audit_log_entries.latest('id').id
    while id_lower_bound <= id_upper_bound:
        do_bulk_backfill_extra_data(audit_log_model, id_lower_bound, min(id_lower_bound + BATCH_SIZE, id_upper_bound))
        id_lower_bound += BATCH_SIZE + 1

class Migration(migrations.Migration):
    atomic = False
    dependencies = [('zerver', '0459_remove_invalid_characters_from_user_group_name')]
    operations = [migrations.RunPython(backfill_extra_data, reverse_code=migrations.RunPython.noop, elidable=True)]