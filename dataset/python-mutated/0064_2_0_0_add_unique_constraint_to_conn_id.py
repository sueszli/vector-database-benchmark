"""Add unique constraint to ``conn_id``

Revision ID: 8d48763f6d53
Revises: 8f966b9c467a
Create Date: 2020-05-03 16:55:01.834231

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.models.base import COLLATION_ARGS
revision = '8d48763f6d53'
down_revision = '8f966b9c467a'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply Add unique constraint to ``conn_id`` and set it as non-nullable'
    try:
        with op.batch_alter_table('connection') as batch_op:
            batch_op.alter_column('conn_id', nullable=False, existing_type=sa.String(250, **COLLATION_ARGS))
            batch_op.create_unique_constraint(constraint_name='unique_conn_id', columns=['conn_id'])
    except sa.exc.IntegrityError:
        raise Exception('Make sure there are no duplicate connections with the same conn_id or null values')

def downgrade():
    if False:
        return 10
    'Unapply Add unique constraint to ``conn_id`` and set it as non-nullable'
    with op.batch_alter_table('connection') as batch_op:
        batch_op.drop_constraint(constraint_name='unique_conn_id', type_='unique')
        batch_op.alter_column('conn_id', nullable=True, existing_type=sa.String(250))