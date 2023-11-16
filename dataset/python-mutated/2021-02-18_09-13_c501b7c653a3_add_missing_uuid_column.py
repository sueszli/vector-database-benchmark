"""add missing uuid column

Revision ID: c501b7c653a3
Revises: 070c043f2fdb
Create Date: 2021-02-18 09:13:00.028317

"""
revision = 'c501b7c653a3'
down_revision = '070c043f2fdb'
import logging
from importlib import import_module
from uuid import uuid4
import sqlalchemy as sa
from alembic import op
from sqlalchemy.engine.reflection import Inspector
from sqlalchemy.orm import load_only
from sqlalchemy_utils import UUIDType
from superset import db
add_uuid_column_to_import_mixin = import_module('superset.migrations.versions.2020-09-28_17-57_b56500de1855_add_uuid_column_to_import_mixin')
assign_uuids = add_uuid_column_to_import_mixin.assign_uuids
models = add_uuid_column_to_import_mixin.models
update_dashboards = add_uuid_column_to_import_mixin.update_dashboards

def has_uuid_column(table_name, bind):
    if False:
        while True:
            i = 10
    inspector = Inspector.from_engine(bind)
    columns = {column['name'] for column in inspector.get_columns(table_name)}
    has_uuid_column = 'uuid' in columns
    if has_uuid_column:
        logging.info('Table %s already has uuid column, skipping...', table_name)
    else:
        logging.info("Table %s doesn't have uuid column, adding...", table_name)
    return has_uuid_column

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for (table_name, model) in models.items():
        if has_uuid_column(table_name, bind):
            continue
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.add_column(sa.Column('uuid', UUIDType(binary=True), primary_key=False, default=uuid4))
        assign_uuids(model, session)
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.create_unique_constraint(f'uq_{table_name}_uuid', ['uuid'])
    slice_uuid_map = {slc.id: slc.uuid for slc in session.query(models['slices']).options(load_only('id', 'uuid')).all()}
    update_dashboards(session, slice_uuid_map)

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    This script fixes b56500de1855_add_uuid_column_to_import_mixin.py by adding any\n    uuid columns that might have been skipped. There's no downgrade.\n    "
    pass