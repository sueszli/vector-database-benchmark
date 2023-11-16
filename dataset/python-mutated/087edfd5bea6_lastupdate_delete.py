"""LastUpdate Delete

Revision ID: 087edfd5bea6
Revises: d638d24c18fd
Create Date: 2023-02-05 23:56:17.674566

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = '087edfd5bea6'
down_revision = 'd638d24c18fd'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.drop_column('product', 'LastUpdate')

def downgrade() -> None:
    if False:
        print('Hello World!')
    op.add_column('product', sa.Column('LastUpdate', postgresql.TIMESTAMP(), autoincrement=False, nullable=True))