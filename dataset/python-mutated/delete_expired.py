import logging
from datetime import datetime
from sqlalchemy import and_
from sqlalchemy.exc import SQLAlchemyError
from superset import db
from superset.commands.base import BaseCommand
from superset.key_value.exceptions import KeyValueDeleteFailedError
from superset.key_value.models import KeyValueEntry
from superset.key_value.types import KeyValueResource
logger = logging.getLogger(__name__)

class DeleteExpiredKeyValueCommand(BaseCommand):
    resource: KeyValueResource

    def __init__(self, resource: KeyValueResource):
        if False:
            while True:
                i = 10
        '\n        Delete all expired key-value pairs\n\n        :param resource: the resource (dashboard, chart etc)\n        :return: was the entry deleted or not\n        '
        self.resource = resource

    def run(self) -> None:
        if False:
            i = 10
            return i + 15
        try:
            self.delete_expired()
        except SQLAlchemyError as ex:
            db.session.rollback()
            raise KeyValueDeleteFailedError() from ex

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def delete_expired(self) -> None:
        if False:
            i = 10
            return i + 15
        db.session.query(KeyValueEntry).filter(and_(KeyValueEntry.resource == self.resource.value, KeyValueEntry.expires_on <= datetime.now())).delete()
        db.session.commit()