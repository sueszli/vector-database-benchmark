"""add data_compressed to serialized_dag

Revision ID: a3bcd0914482
Revises: e655c0453f75
Create Date: 2022-02-03 22:40:59.841119

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'a3bcd0914482'
down_revision = 'e655c0453f75'
branch_labels = None
depends_on = None
airflow_version = '2.3.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('serialized_dag') as batch_op:
        batch_op.alter_column('data', existing_type=sa.JSON, nullable=True)
        batch_op.add_column(sa.Column('data_compressed', sa.LargeBinary, nullable=True))

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('serialized_dag') as batch_op:
        batch_op.alter_column('data', existing_type=sa.JSON, nullable=False)
        batch_op.drop_column('data_compressed')