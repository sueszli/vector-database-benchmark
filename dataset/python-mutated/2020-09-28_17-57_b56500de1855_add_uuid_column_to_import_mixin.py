"""add_uuid_column_to_import_mixin

Revision ID: b56500de1855
Revises: 18532d70ab98
Create Date: 2020-09-28 17:57:23.128142

"""
import json
import os
from json.decoder import JSONDecodeError
from uuid import uuid4
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import load_only
from sqlalchemy_utils import UUIDType
from superset import db
from superset.migrations.shared.utils import assign_uuids
from superset.utils import core as utils
revision = 'b56500de1855'
down_revision = '18532d70ab98'
Base = declarative_base()

class ImportMixin:
    id = sa.Column(sa.Integer, primary_key=True)
    uuid = sa.Column(UUIDType(binary=True), primary_key=False, default=uuid4)
table_names = ['dbs', 'dashboards', 'slices', 'tables', 'table_columns', 'sql_metrics', 'clusters', 'datasources', 'columns', 'metrics', 'dashboard_email_schedules', 'slice_email_schedules']
models = {table_name: type(table_name, (Base, ImportMixin), {'__tablename__': table_name}) for table_name in table_names}
models['dashboards'].position_json = sa.Column(utils.MediumText())
default_batch_size = int(os.environ.get('BATCH_SIZE', 200))

def update_position_json(dashboard, session, uuid_map):
    if False:
        while True:
            i = 10
    try:
        layout = json.loads(dashboard.position_json or '{}')
    except JSONDecodeError:
        layout = {}
    for object_ in layout.values():
        if isinstance(object_, dict) and object_['type'] == 'CHART' and object_['meta']['chartId']:
            chart_id = object_['meta']['chartId']
            if chart_id in uuid_map:
                object_['meta']['uuid'] = str(uuid_map[chart_id])
            elif object_['meta'].get('uuid'):
                del object_['meta']['uuid']
    dashboard.position_json = json.dumps(layout, indent=4)
    session.merge(dashboard)

def update_dashboards(session, uuid_map):
    if False:
        print('Hello World!')
    message = 'Updating dashboard position json with slice uuid..' if uuid_map else 'Cleaning up slice uuid from dashboard position json..'
    print(f'\n{message}\r', end='')
    query = session.query(models['dashboards'])
    dashboard_count = query.count()
    for (i, dashboard) in enumerate(query.all()):
        update_position_json(dashboard, session, uuid_map)
        if i and i % default_batch_size == 0:
            session.commit()
        print(f'{message} {i + 1}/{dashboard_count}\r', end='')
    session.commit()
    print(f'{message} Done.      \n')

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for (table_name, model) in models.items():
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.add_column(sa.Column('uuid', UUIDType(binary=True), primary_key=False, default=uuid4))
        assign_uuids(model, session)
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.create_unique_constraint(f'uq_{table_name}_uuid', ['uuid'])
    slice_uuid_map = {slc.id: slc.uuid for slc in session.query(models['slices']).options(load_only('id', 'uuid')).all()}
    update_dashboards(session, slice_uuid_map)

def downgrade():
    if False:
        print('Hello World!')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    update_dashboards(session, {})
    for table_name in models:
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.drop_constraint(f'uq_{table_name}_uuid', type_='unique')
            batch_op.drop_column('uuid')