"""Backfill state_name

Revision ID: 14dc68cc5853
Revises: 605ebb4e9155
Create Date: 2022-04-21 09:55:19.820177

"""
import sqlalchemy as sa
from alembic import op
revision = '14dc68cc5853'
down_revision = '605ebb4e9155'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    '\n    Backfills state_name column for task_run and flow_run tables.\n\n    This is a data only migration that can be run as many\n    times as desired.\n    '
    update_flow_run_state_name_in_batches = '\n        WITH null_flow_run_state_name_cte as (SELECT id from flow_run where state_name is null and state_id is not null limit 500)\n        UPDATE flow_run\n        SET state_name = flow_run_state.name\n        FROM flow_run_state, null_flow_run_state_name_cte\n        WHERE flow_run.state_id = flow_run_state.id\n        AND flow_run.id = null_flow_run_state_name_cte.id;\n    '
    update_task_run_state_name_in_batches = '\n        WITH null_task_run_state_name_cte as (SELECT id from task_run where state_name is null and state_id is not null limit 500)\n        UPDATE task_run\n        SET state_name = task_run_state.name\n        FROM task_run_state, null_task_run_state_name_cte\n        WHERE task_run.state_id = task_run_state.id\n        AND task_run.id = null_task_run_state_name_cte.id;\n    '
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
        return 10
    '\n    Data only migration. No action on downgrade.\n    '