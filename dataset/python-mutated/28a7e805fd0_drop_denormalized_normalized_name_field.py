"""
Drop denormalized normalized_name field

Revision ID: 28a7e805fd0
Revises: 91508cc5c2
Create Date: 2015-04-05 00:45:54.649441
"""
from alembic import op
revision = '28a7e805fd0'
down_revision = '91508cc5c2'

def upgrade():
    if False:
        return 10
    op.drop_column('packages', 'normalized_name')

def downgrade():
    if False:
        i = 10
        return i + 15
    raise RuntimeError(f'Cannot downgrade past revision: {revision!r}')