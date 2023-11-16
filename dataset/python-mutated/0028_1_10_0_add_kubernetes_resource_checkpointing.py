"""Add Kubernetes resource check-pointing

Revision ID: 33ae817a1ff4
Revises: 947454bf1dff
Create Date: 2017-09-11 15:26:47.598494

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect
revision = '33ae817a1ff4'
down_revision = 'd2ae31099d61'
branch_labels = None
depends_on = None
airflow_version = '1.10.0'
RESOURCE_TABLE = 'kube_resource_version'

def upgrade():
    if False:
        i = 10
        return i + 15
    conn = op.get_bind()
    inspector = inspect(conn)
    if RESOURCE_TABLE not in inspector.get_table_names():
        columns_and_constraints = [sa.Column('one_row_id', sa.Boolean, server_default=sa.true(), primary_key=True), sa.Column('resource_version', sa.String(255))]
        if conn.dialect.name in {'mysql'}:
            columns_and_constraints.append(sa.CheckConstraint('one_row_id<>0', name='kube_resource_version_one_row_id'))
        elif conn.dialect.name not in {'mssql'}:
            columns_and_constraints.append(sa.CheckConstraint('one_row_id', name='kube_resource_version_one_row_id'))
        table = op.create_table(RESOURCE_TABLE, *columns_and_constraints)
        op.bulk_insert(table, [{'resource_version': ''}])

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    conn = op.get_bind()
    inspector = inspect(conn)
    if RESOURCE_TABLE in inspector.get_table_names():
        op.drop_table(RESOURCE_TABLE)