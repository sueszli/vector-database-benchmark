"""empty message

Revision ID: 439807debe51
Revises: 1a01451ec045
Create Date: 2021-11-24 09:28:48.710453

"""
from alembic import op
revision = '439807debe51'
down_revision = '1a01451ec045'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.create_index('ix_jobs_project_uuid_next_scheduled_time_status', 'jobs', ['project_uuid', 'next_scheduled_time', 'status'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('ix_jobs_project_uuid_next_scheduled_time_status', table_name='jobs')