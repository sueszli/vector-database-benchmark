"""
create missing primary key constraints

Revision ID: b5bb5d08543d
Revises: 08aedc089eaf
Create Date: 2019-12-19 14:27:47.230249
"""
from alembic import op
revision = 'b5bb5d08543d'
down_revision = '08aedc089eaf'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_primary_key(None, 'release_files', ['id'])
    op.create_primary_key(None, 'release_dependencies', ['id'])
    op.create_primary_key(None, 'roles', ['id'])

def downgrade():
    if False:
        while True:
            i = 10
    raise RuntimeError('Order No. 227 - Ни шагу назад!')