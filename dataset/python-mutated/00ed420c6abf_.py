"""Add CronJobRunEvent.total_pipeline_runs field

Revision ID: 00ed420c6abf
Revises: 7b9a23ad5946
Create Date: 2022-04-28 13:38:19.223004

"""
import sqlalchemy as sa
from alembic import op
revision = '00ed420c6abf'
down_revision = '7b9a23ad5946'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('events', sa.Column('total_pipeline_runs', sa.Integer(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('events', 'total_pipeline_runs')