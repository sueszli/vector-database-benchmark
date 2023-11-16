"""Add index for scheduled deployments

Revision ID: a0284438370e
Revises: af52717cf201
Create Date: 2022-11-05 18:06:19.568896

"""
from alembic import op
revision = 'a0284438370e'
down_revision = 'af52717cf201'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute("\n        CREATE INDEX ix_flow_run__scheduler_deployment_id_auto_scheduled_next_scheduled_start_time \n        ON flow_run (deployment_id, auto_scheduled, next_scheduled_start_time) \n        WHERE state_type = 'SCHEDULED';\n        ")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('\n        DROP INDEX ix_flow_run__scheduler_deployment_id_auto_scheduled_next_scheduled_start_time;\n        ')