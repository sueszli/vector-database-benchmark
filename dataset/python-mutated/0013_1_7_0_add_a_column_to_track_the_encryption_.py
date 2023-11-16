"""Add a column to track the encryption state of the 'Extra' field in connection

Revision ID: bba5a7cfc896
Revises: bbc73705a13e
Create Date: 2016-01-29 15:10:32.656425

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'bba5a7cfc896'
down_revision = 'bbc73705a13e'
branch_labels = None
depends_on = None
airflow_version = '1.7.0'

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('connection', sa.Column('is_extra_encrypted', sa.Boolean, default=False))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('connection', 'is_extra_encrypted')