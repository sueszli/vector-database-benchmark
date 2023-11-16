"""
Add unique constraint for release version

Revision ID: 08aedc089eaf
Revises: ee4c59b2ef3a
Create Date: 2019-11-12 22:51:31.463150
"""
from alembic import op
revision = '08aedc089eaf'
down_revision = 'ee4c59b2ef3a'

def upgrade():
    if False:
        while True:
            i = 10
    op.create_unique_constraint(None, 'releases', ['project_id', 'version'])

def downgrade():
    if False:
        return 10
    op.drop_constraint(None, 'releases', type_='unique')