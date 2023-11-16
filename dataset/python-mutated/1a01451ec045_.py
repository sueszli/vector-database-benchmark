"""empty message

Revision ID: 1a01451ec045
Revises: 00e31eec40c6
Create Date: 2021-11-24 09:22:49.405798

"""
from alembic import op
revision = '1a01451ec045'
down_revision = '00e31eec40c6'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.create_index('ix_jobs_next_scheduled_time_status', 'jobs', ['next_scheduled_time', 'status'], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_index('ix_jobs_next_scheduled_time_status', table_name='jobs')