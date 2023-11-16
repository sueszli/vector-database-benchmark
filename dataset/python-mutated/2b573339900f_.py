"""Add ProjectUpdateEvent, PipelineUpdateEvent models, event_types

Revision ID: 2b573339900f
Revises: 23def7128481
Create Date: 2022-05-17 08:59:51.787022

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '2b573339900f'
down_revision = '23def7128481'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:created'),\n        ('project:updated'),\n        ('project:deleted'),\n        ('project:pipeline:created'),\n        ('project:pipeline:updated'),\n        ('project:pipeline:deleted')\n        ;\n        ")
    op.add_column('events', sa.Column('update', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('events', 'update')