"""convert key-value entries to json

Revision ID: 9c2a5681ddfd
Revises: f3c2d8ec8595
Create Date: 2023-05-01 12:03:17.079862

"""
revision = '9c2a5681ddfd'
down_revision = 'f3c2d8ec8595'
import io
import json
import pickle
from alembic import op
from sqlalchemy import Column, Integer, LargeBinary, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from superset import db
from superset.migrations.shared.utils import paginated_update
Base = declarative_base()
VALUE_MAX_SIZE = 2 ** 24 - 1
RESOURCES_TO_MIGRATE = ('app', 'dashboard_permalink', 'explore_permalink')

class RestrictedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if False:
            for i in range(10):
                print('nop')
        if not (module == 'superset.utils.core' and name == 'DatasourceType'):
            raise pickle.UnpicklingError(f'Unpickling of {module}.{name} is forbidden')
        return super().find_class(module, name)

class KeyValueEntry(Base):
    __tablename__ = 'key_value'
    id = Column(Integer, primary_key=True)
    resource = Column(String(32), nullable=False)
    value = Column(LargeBinary(length=VALUE_MAX_SIZE), nullable=False)

def upgrade():
    if False:
        print('Hello World!')
    bind = op.get_bind()
    session: Session = db.Session(bind=bind)
    truncated_count = 0
    for entry in paginated_update(session.query(KeyValueEntry).filter(KeyValueEntry.resource.in_(RESOURCES_TO_MIGRATE))):
        try:
            value = RestrictedUnpickler(io.BytesIO(entry.value)).load() or {}
        except pickle.UnpicklingError as ex:
            if str(ex) == 'pickle data was truncated':
                truncated_count += 1
                value = {}
            else:
                raise
        entry.value = bytes(json.dumps(value), encoding='utf-8')
    if truncated_count:
        print(f'Replaced {truncated_count} corrupted values with an empty value')

def downgrade():
    if False:
        return 10
    bind = op.get_bind()
    session: Session = db.Session(bind=bind)
    for entry in paginated_update(session.query(KeyValueEntry).filter(KeyValueEntry.resource.in_(RESOURCES_TO_MIGRATE))):
        value = json.loads(entry.value) or {}
        entry.value = pickle.dumps(value)