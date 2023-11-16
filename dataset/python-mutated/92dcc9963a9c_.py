"""Add CronJobEvent and CronJobPipelineRunEvent models and event types

Revision ID: 92dcc9963a9c
Revises: 814961a3d525
Create Date: 2022-04-25 09:29:37.229886

"""
from alembic import op
revision = '92dcc9963a9c'
down_revision = '814961a3d525'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:cron-job:created'),\n        ('project:cron-job:started'),\n        ('project:cron-job:deleted'),\n        ('project:cron-job:cancelled'),\n        ('project:cron-job:failed'),\n        ('project:cron-job:paused'),\n        ('project:cron-job:unpaused'),\n        ('project:cron-job:run:started'),\n        ('project:cron-job:run:succeeded'),\n        ('project:cron-job:run:failed'),\n        ('project:cron-job:run:pipeline-run:created'),\n        ('project:cron-job:run:pipeline-run:started'),\n        ('project:cron-job:run:pipeline-run:cancelled'),\n        ('project:cron-job:run:pipeline-run:failed'),\n        ('project:cron-job:run:pipeline-run:deleted'),\n        ('project:cron-job:run:pipeline-run:succeeded')\n        ;\n        ")
    pass

def downgrade():
    if False:
        while True:
            i = 10
    pass