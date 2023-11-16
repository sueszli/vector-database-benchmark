"""Rename block data table

Revision ID: d9d98a9ebb6f
Revises: 
Create Date: 2022-02-19 20:55:43.397189

"""
from alembic import op
revision = 'd9d98a9ebb6f'
down_revision = '679e695af6ba'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute('\n        ALTER TABLE block_data\n        RENAME TO block;\n        ')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('\n        ALTER TABLE block\n        RENAME TO block_data;\n        ')