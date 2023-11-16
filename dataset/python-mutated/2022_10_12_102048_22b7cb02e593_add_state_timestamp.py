"""Add state_timestamp

Revision ID: 22b7cb02e593
Revises: e757138e954a
Create Date: 2022-10-12 10:20:48.760447

"""
import sqlalchemy as sa
from alembic import op
import prefect
revision = '22b7cb02e593'
down_revision = 'e757138e954a'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('flow_run', schema=None) as batch_op:
        batch_op.add_column(sa.Column('state_timestamp', prefect.server.utilities.database.Timestamp(timezone=True), nullable=True))
        batch_op.create_index('ix_flow_run__state_timestamp', ['state_timestamp'], unique=False)
    with op.batch_alter_table('task_run', schema=None) as batch_op:
        batch_op.add_column(sa.Column('state_timestamp', prefect.server.utilities.database.Timestamp(timezone=True), nullable=True))
        batch_op.create_index('ix_task_run__state_timestamp', ['state_timestamp'], unique=False)
    update_flow_run_state_timestamp_in_batches = '\n        UPDATE flow_run\n        SET state_timestamp = (SELECT timestamp from flow_run_state where flow_run.state_id = flow_run_state.id)\n        WHERE flow_run.id in (SELECT id from flow_run where state_timestamp is null and state_id is not null limit 500);\n    '
    update_task_run_state_timestamp_in_batches = '\n        UPDATE task_run\n        SET state_timestamp = (SELECT timestamp from task_run_state where task_run.state_id = task_run_state.id)\n        WHERE task_run.id in (SELECT id from task_run where state_timestamp is null and state_id is not null limit 500);\n    '
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        while True:
            result = conn.execute(sa.text(update_flow_run_state_timestamp_in_batches))
            if result.rowcount <= 0:
                break
        while True:
            result = conn.execute(sa.text(update_task_run_state_timestamp_in_batches))
            if result.rowcount <= 0:
                break

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('task_run', schema=None) as batch_op:
        batch_op.drop_index('ix_task_run__state_timestamp')
        batch_op.drop_column('state_timestamp')
    with op.batch_alter_table('flow_run', schema=None) as batch_op:
        batch_op.drop_index('ix_flow_run__state_timestamp')
        batch_op.drop_column('state_timestamp')