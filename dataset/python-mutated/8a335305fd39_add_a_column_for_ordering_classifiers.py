"""
Add a column for ordering classifiers

Revision ID: 8a335305fd39
Revises: 4490777c984f
Create Date: 2022-07-22 00:06:40.868910
"""
import sqlalchemy as sa
from alembic import op
revision = '8a335305fd39'
down_revision = '4490777c984f'

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('trove_classifiers', sa.Column('ordering', sa.Integer(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('trove_classifiers', 'ordering')