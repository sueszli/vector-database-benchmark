"""empty message

Revision ID: 3fbff22943d6
Revises: bab71acdcd61
Create Date: 2022-04-25 10:34:01.569021

"""
from alembic import op
revision = '3fbff22943d6'
down_revision = 'bab71acdcd61'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_index(op.f('ix_events_type_events_project_uuid_events_job_uuid_events_run_index'), 'events', ['type', 'project_uuid', 'job_uuid', 'run_index'], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_index(op.f('ix_events_type_events_project_uuid_events_job_uuid_events_run_index'), table_name='events')