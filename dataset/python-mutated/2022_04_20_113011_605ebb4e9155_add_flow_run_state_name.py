"""Add flow_run.state_name

Revision ID: 605ebb4e9155
Revises: 2e7e1428ffce
Create Date: 2022-04-20 11:30:11.934795

"""
import sqlalchemy as sa
from alembic import op
revision = '605ebb4e9155'
down_revision = '2e7e1428ffce'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('flow_run', sa.Column('state_name', sa.String(), nullable=True))
    op.add_column('task_run', sa.Column('state_name', sa.String(), nullable=True))
    with op.get_context().autocommit_block():
        op.execute('\n            CREATE INDEX CONCURRENTLY IF NOT EXISTS\n            ix_flow_run__state_name\n            ON flow_run(state_name)\n            ')
        op.execute('\n            CREATE INDEX CONCURRENTLY IF NOT EXISTS\n            ix_task_run__state_name\n            ON task_run(state_name)\n            ')

def downgrade():
    if False:
        while True:
            i = 10
    with op.get_context().autocommit_block():
        op.execute('DROP INDEX CONCURRENTLY IF EXISTS ix_flow_run__state_name')
        op.execute('DROP INDEX CONCURRENTLY IF EXISTS ix_task_run__state_name')
    op.drop_column('flow_run', 'state_name')
    op.drop_column('task_run', 'state_name')