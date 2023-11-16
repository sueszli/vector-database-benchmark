"""enable security pragmas

Revision ID: faac8092c123
Revises: 15ac9509fc68
Create Date: 2018-03-31 10:44:26.533395

"""
import sqlalchemy as sa
from alembic import op
revision = 'faac8092c123'
down_revision = '15ac9509fc68'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    conn = op.get_bind()
    conn.execute(sa.text('PRAGMA secure_delete = ON'))
    conn.execute(sa.text('PRAGMA auto_vacuum = FULL'))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    pass