"""Add condition_failed status to block run status

Revision ID: dfe49d040487
Revises: e7beb59b44f9
Create Date: 2023-06-12 10:25:55.734358

"""
from alembic import op
import sqlalchemy as sa
revision = 'dfe49d040487'
down_revision = 'e7beb59b44f9'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    if bind.engine.name == 'postgresql':
        with op.get_context().autocommit_block():
            op.execute("ALTER TYPE blockrunstatus ADD VALUE 'CONDITION_FAILED'")

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    if bind.engine.name == 'postgresql':
        op.execute('ALTER TYPE blockrunstatus RENAME TO blockrunstatus_old')
        op.execute("CREATE TYPE blockrunstatus AS ENUM('INITIAL', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', 'UPSTREAM_FAILED')")
        op.execute('ALTER TABLE block_run ALTER COLUMN status TYPE blockrunstatus USING status::text::blockrunstatus')
        op.execute('DROP TYPE blockrunstatus_old')