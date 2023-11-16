"""Rename block data table

Revision ID: 4c4a6a138053
Revises: 
Create Date: 2022-02-19 21:02:55.886313

"""
from alembic import op
revision = '4c4a6a138053'
down_revision = '28ae48128c75'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('\n        ALTER TABLE block_data\n        RENAME TO block;\n        ')

def downgrade():
    if False:
        return 10
    op.execute('\n        ALTER TABLE block\n        RENAME TO block_data;\n        ')