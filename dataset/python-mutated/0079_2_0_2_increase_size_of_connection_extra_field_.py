"""Increase size of ``connection.extra`` field to handle multiple RSA keys

Revision ID: 449b4072c2da
Revises: 82b7c48c147f
Create Date: 2020-03-16 19:02:55.337710

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '449b4072c2da'
down_revision = '82b7c48c147f'
branch_labels = None
depends_on = None
airflow_version = '2.0.2'

def upgrade():
    if False:
        return 10
    'Apply increase_length_for_connection_password'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('extra', existing_type=sa.VARCHAR(length=5000), type_=sa.TEXT(), existing_nullable=True)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply increase_length_for_connection_password'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('extra', existing_type=sa.TEXT(), type_=sa.VARCHAR(length=5000), existing_nullable=True)