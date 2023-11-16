"""Index FlowRun.flow_runner_type

Revision ID: f327e877e423
Revises: e1ff4973a9eb
Create Date: 2022-02-21 11:12:38.518778

"""
from alembic import op
revision = 'f327e877e423'
down_revision = 'e1ff4973a9eb'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.create_index(op.f('ix_flow_run__flow_runner_type'), 'flow_run', ['flow_runner_type'], unique=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_index(op.f('ix_flow_run__flow_runner_type'), table_name='flow_run')