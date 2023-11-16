"""Add processor_subdir column to DagModel, SerializedDagModel and CallbackRequest tables.

Revision ID: ecb43d2a1842
Revises: 1486deb605b4
Create Date: 2022-08-26 11:30:11.249580

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'ecb43d2a1842'
down_revision = '1486deb605b4'
branch_labels = None
depends_on = None
airflow_version = '2.4.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Apply add processor_subdir to DagModel and SerializedDagModel'
    conn = op.get_bind()
    with op.batch_alter_table('dag') as batch_op:
        if conn.dialect.name == 'mysql':
            batch_op.add_column(sa.Column('processor_subdir', sa.Text(length=2000), nullable=True))
        else:
            batch_op.add_column(sa.Column('processor_subdir', sa.String(length=2000), nullable=True))
    with op.batch_alter_table('serialized_dag') as batch_op:
        if conn.dialect.name == 'mysql':
            batch_op.add_column(sa.Column('processor_subdir', sa.Text(length=2000), nullable=True))
        else:
            batch_op.add_column(sa.Column('processor_subdir', sa.String(length=2000), nullable=True))
    with op.batch_alter_table('callback_request') as batch_op:
        batch_op.drop_column('dag_directory')
        if conn.dialect.name == 'mysql':
            batch_op.add_column(sa.Column('processor_subdir', sa.Text(length=2000), nullable=True))
        else:
            batch_op.add_column(sa.Column('processor_subdir', sa.String(length=2000), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add processor_subdir to DagModel and SerializedDagModel'
    conn = op.get_bind()
    with op.batch_alter_table('dag', schema=None) as batch_op:
        batch_op.drop_column('processor_subdir')
    with op.batch_alter_table('serialized_dag', schema=None) as batch_op:
        batch_op.drop_column('processor_subdir')
    with op.batch_alter_table('callback_request') as batch_op:
        batch_op.drop_column('processor_subdir')
        if conn.dialect.name == 'mysql':
            batch_op.add_column(sa.Column('dag_directory', sa.Text(length=1000), nullable=True))
        else:
            batch_op.add_column(sa.Column('dag_directory', sa.String(length=1000), nullable=True))