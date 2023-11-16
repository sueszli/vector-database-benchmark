"""empty message

Revision ID: a25c8a60484c
Revises: 439807debe51
Create Date: 2021-12-21 09:32:21.827493

"""
import sqlalchemy as sa
from alembic import op
revision = 'a25c8a60484c'
down_revision = '439807debe51'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('jobs', sa.Column('max_retained_pipeline_runs', sa.Integer(), server_default=sa.text('-1'), nullable=False))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('jobs', 'max_retained_pipeline_runs')