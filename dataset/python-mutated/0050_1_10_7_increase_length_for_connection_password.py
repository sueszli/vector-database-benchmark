"""Increase length for connection password

Revision ID: fe461863935f
Revises: 08364691d074
Create Date: 2019-12-08 09:47:09.033009

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'fe461863935f'
down_revision = '08364691d074'
branch_labels = None
depends_on = None
airflow_version = '1.10.7'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply Increase length for connection password'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('password', existing_type=sa.VARCHAR(length=500), type_=sa.String(length=5000), existing_nullable=True)

def downgrade():
    if False:
        while True:
            i = 10
    'Unapply Increase length for connection password'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('password', existing_type=sa.String(length=5000), type_=sa.VARCHAR(length=500), existing_nullable=True)