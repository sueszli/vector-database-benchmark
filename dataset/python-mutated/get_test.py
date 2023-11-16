from __future__ import annotations
import json
import uuid
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from flask.ctx import AppContext
from superset.extensions import db
from tests.integration_tests.key_value.commands.fixtures import ID_KEY, JSON_CODEC, JSON_VALUE, key_value_entry, RESOURCE, UUID_KEY
if TYPE_CHECKING:
    from superset.key_value.models import KeyValueEntry

def test_get_id_entry(app_context: AppContext, key_value_entry: KeyValueEntry) -> None:
    if False:
        i = 10
        return i + 15
    from superset.key_value.commands.get import GetKeyValueCommand
    value = GetKeyValueCommand(resource=RESOURCE, key=ID_KEY, codec=JSON_CODEC).run()
    assert value == JSON_VALUE

def test_get_uuid_entry(app_context: AppContext, key_value_entry: KeyValueEntry) -> None:
    if False:
        return 10
    from superset.key_value.commands.get import GetKeyValueCommand
    value = GetKeyValueCommand(resource=RESOURCE, key=UUID_KEY, codec=JSON_CODEC).run()
    assert value == JSON_VALUE

def test_get_id_entry_missing(app_context: AppContext, key_value_entry: KeyValueEntry) -> None:
    if False:
        return 10
    from superset.key_value.commands.get import GetKeyValueCommand
    value = GetKeyValueCommand(resource=RESOURCE, key=456, codec=JSON_CODEC).run()
    assert value is None

def test_get_expired_entry(app_context: AppContext) -> None:
    if False:
        return 10
    from superset.key_value.commands.get import GetKeyValueCommand
    from superset.key_value.models import KeyValueEntry
    entry = KeyValueEntry(id=678, uuid=uuid.uuid4(), resource=RESOURCE, value=bytes(json.dumps(JSON_VALUE), encoding='utf-8'), expires_on=datetime.now() - timedelta(days=1))
    db.session.add(entry)
    db.session.commit()
    value = GetKeyValueCommand(resource=RESOURCE, key=ID_KEY, codec=JSON_CODEC).run()
    assert value is None
    db.session.delete(entry)
    db.session.commit()

def test_get_future_expiring_entry(app_context: AppContext) -> None:
    if False:
        print('Hello World!')
    from superset.key_value.commands.get import GetKeyValueCommand
    from superset.key_value.models import KeyValueEntry
    id_ = 789
    entry = KeyValueEntry(id=id_, uuid=uuid.uuid4(), resource=RESOURCE, value=bytes(json.dumps(JSON_VALUE), encoding='utf-8'), expires_on=datetime.now() + timedelta(days=1))
    db.session.add(entry)
    db.session.commit()
    value = GetKeyValueCommand(resource=RESOURCE, key=id_, codec=JSON_CODEC).run()
    assert value == JSON_VALUE
    db.session.delete(entry)
    db.session.commit()