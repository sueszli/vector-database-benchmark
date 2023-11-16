"""Adds indexes for block filtering

Revision ID: a205b458d997
Revises: 9e2a1c08c6f1
Create Date: 2022-06-21 09:36:40.029598

"""
from alembic import op
revision = 'a205b458d997'
down_revision = '9e2a1c08c6f1'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.execute('\n        CREATE INDEX ix_block_type_name_case_insensitive on block_type (name COLLATE NOCASE);\n        ')

def downgrade():
    if False:
        while True:
            i = 10
    op.execute('\n        DROP INDEX ix_block_type_name_case_insensitive;\n        ')