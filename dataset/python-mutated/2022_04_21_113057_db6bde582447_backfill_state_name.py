"""Backfill state_name

Revision ID: db6bde582447
Revises: 7f5f335cace3
Create Date: 2022-04-21 11:30:57.542292

"""
import sqlalchemy as sa
from alembic import op
revision = 'db6bde582447'
down_revision = '7f5f335cace3'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    '\n    Backfills state_name column for task_run and flow_run tables.\n\n    This is a data only migration that can be run as many\n    times as desired.\n    '
    update_flow_run_state_name_in_batches = '\n        UPDATE flow_run\n        SET state_name = (SELECT name from flow_run_state where flow_run.state_id = flow_run_state.id)\n        WHERE flow_run.id in (SELECT id from flow_run where state_name is null and state_id is not null limit 500);\n    '
    update_task_run_state_name_in_batches = '\n        UPDATE task_run\n        SET state_name = (SELECT name from task_run_state where task_run.state_id = task_run_state.id)\n        WHERE task_run.id in (SELECT id from task_run where state_name is null and state_id is not null limit 500);\n    '
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        while True:
            result = conn.execute(sa.text(update_flow_run_state_name_in_batches))
            if result.rowcount <= 0:
                break
        while True:
            result = conn.execute(sa.text(update_task_run_state_name_in_batches))
            if result.rowcount <= 0:
                break

def downgrade():
    if False:
        i = 10
        return i + 15
    '\n    Data only migration. No action on downgrade.\n    '