"""Add OneOffJobUpdateEvent, CronJobUpdateEvent and related event_types

Revision ID: a4b1f48ddab5
Revises: a863be01327d
Create Date: 2022-05-17 12:47:18.027113

"""
import sqlalchemy as sa
from alembic import op
revision = 'a4b1f48ddab5'
down_revision = 'a863be01327d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:cron-job:updated'),\n        ('project:one-off-job:updated')\n        ;\n        ")
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass