"""
remove_organization_project_is_active

Revision ID: b08bcde4183c
Revises: 94c844c2da96
Create Date: 2022-05-24 19:22:41.034512
"""
import sqlalchemy as sa
from alembic import op
revision = 'b08bcde4183c'
down_revision = '94c844c2da96'

def upgrade():
    if False:
        while True:
            i = 10
    op.drop_column('organization_project', 'is_active')

def downgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('organization_project', sa.Column('is_active', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False))