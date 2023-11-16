import logging
from datetime import datetime
from typing import Any, Optional, Union
from uuid import UUID
from sqlalchemy.exc import SQLAlchemyError
from superset import db
from superset.commands.base import BaseCommand
from superset.key_value.commands.create import CreateKeyValueCommand
from superset.key_value.exceptions import KeyValueCreateFailedError, KeyValueUpsertFailedError
from superset.key_value.models import KeyValueEntry
from superset.key_value.types import Key, KeyValueCodec, KeyValueResource
from superset.key_value.utils import get_filter
from superset.utils.core import get_user_id
logger = logging.getLogger(__name__)

class UpsertKeyValueCommand(BaseCommand):
    resource: KeyValueResource
    value: Any
    key: Union[int, UUID]
    codec: KeyValueCodec
    expires_on: Optional[datetime]

    def __init__(self, resource: KeyValueResource, key: Union[int, UUID], value: Any, codec: KeyValueCodec, expires_on: Optional[datetime]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Upsert a key value entry\n\n        :param resource: the resource (dashboard, chart etc)\n        :param key: the key to update\n        :param value: the value to persist in the key-value store\n        :param codec: codec used to encode the value\n        :param expires_on: entry expiration time\n        :return: the key associated with the updated value\n        '
        self.resource = resource
        self.key = key
        self.value = value
        self.codec = codec
        self.expires_on = expires_on

    def run(self) -> Key:
        if False:
            print('Hello World!')
        try:
            return self.upsert()
        except (KeyValueCreateFailedError, SQLAlchemyError) as ex:
            db.session.rollback()
            raise KeyValueUpsertFailedError() from ex

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        pass

    def upsert(self) -> Key:
        if False:
            print('Hello World!')
        filter_ = get_filter(self.resource, self.key)
        entry: KeyValueEntry = db.session.query(KeyValueEntry).filter_by(**filter_).autoflush(False).first()
        if entry:
            entry.value = self.codec.encode(self.value)
            entry.expires_on = self.expires_on
            entry.changed_on = datetime.now()
            entry.changed_by_fk = get_user_id()
            db.session.merge(entry)
            db.session.commit()
            return Key(entry.id, entry.uuid)
        return CreateKeyValueCommand(resource=self.resource, value=self.value, codec=self.codec, key=self.key, expires_on=self.expires_on).run()