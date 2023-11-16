"""
create upload_limit constraint

Revision ID: 1d88dd9242e1
Revises: aa3a4757f33a
Create Date: 2022-12-07 14:15:34.126364
"""
from alembic import op
revision = '1d88dd9242e1'
down_revision = 'aa3a4757f33a'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_check_constraint('projects_upload_limit_max_value', 'projects', 'upload_limit <= 1073741824')

def downgrade():
    if False:
        return 10
    op.drop_constraint('projects_upload_limit_max_value', 'projects')