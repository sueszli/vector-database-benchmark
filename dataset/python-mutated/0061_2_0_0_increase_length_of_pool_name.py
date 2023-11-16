"""Increase length of pool name

Revision ID: b25a55525161
Revises: bbf4a7ad0465
Create Date: 2020-03-09 08:48:14.534700

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.models.base import COLLATION_ARGS
revision = 'b25a55525161'
down_revision = 'bbf4a7ad0465'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Increase column length of pool name from 50 to 256 characters'
    with op.batch_alter_table('slot_pool', table_args=sa.UniqueConstraint('pool')) as batch_op:
        batch_op.alter_column('pool', type_=sa.String(256, **COLLATION_ARGS))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Revert Increased length of pool name from 256 to 50 characters'
    with op.batch_alter_table('slot_pool', table_args=sa.UniqueConstraint('pool')) as batch_op:
        batch_op.alter_column('pool', type_=sa.String(50))