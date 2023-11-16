"""Add ``notification_sent`` column to ``sla_miss`` table

Revision ID: bbc73705a13e
Revises: 4446e08588
Create Date: 2016-01-14 18:05:54.871682

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'bbc73705a13e'
down_revision = '4446e08588'
branch_labels = None
depends_on = None
airflow_version = '1.7.0'

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('sla_miss', sa.Column('notification_sent', sa.Boolean, default=False))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('sla_miss', 'notification_sent')