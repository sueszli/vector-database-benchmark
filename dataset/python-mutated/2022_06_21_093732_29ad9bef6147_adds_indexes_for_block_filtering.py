"""Adds indexes for block filtering

Revision ID: 29ad9bef6147
Revises: d335ad57d5ba
Create Date: 2022-06-21 09:37:32.382898

"""
from alembic import op
revision = '29ad9bef6147'
down_revision = 'd335ad57d5ba'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    with op.get_context().autocommit_block():
        op.execute('\n            CREATE INDEX CONCURRENTLY \n            trgm_ix_block_type_name \n            ON block_type USING gin (name gin_trgm_ops);\n            ')
        op.execute('\n            CREATE INDEX CONCURRENTLY\n            ix_block_schema__capabilities\n            ON block_schema USING gin (capabilities)\n            ')

def downgrade():
    if False:
        while True:
            i = 10
    with op.get_context().autocommit_block():
        op.execute('\n            DROP INDEX CONCURRENTLY trgm_ix_block_type_name;\n            ')
        op.execute('\n            DROP INDEX CONCURRENTLY ix_block_schema__capabilities;\n            ')