"""Index and backfill block_type_name

Revision ID: 22ef3915ccd8
Revises: cef24af2ec34
Create Date: 2023-10-30 10:37:20.922002

"""
import sqlalchemy as sa
from alembic import op
revision = '22ef3915ccd8'
down_revision = 'cef24af2ec34'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('block_document', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_block_document__block_type_name_name'), ['block_type_name', 'name'], unique=False)
    backfill_query = '\n        WITH null_block_type_name_cte AS (SELECT id from block_document where block_type_name is null limit 500)\n        UPDATE block_document\n        SET block_type_name = block_type.name\n        FROM block_type, null_block_type_name_cte\n        WHERE block_document.block_type_id = block_type.id\n        AND block_document.id = null_block_type_name_cte.id;\n    '
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        while True:
            result = conn.execute(sa.text(backfill_query))
            if result.rowcount <= 0:
                break

def downgrade():
    if False:
        return 10
    with op.get_context().autocommit_block():
        op.execute('\n             DROP INDEX\n             ix_block_document__block_type_name_name\n             ')