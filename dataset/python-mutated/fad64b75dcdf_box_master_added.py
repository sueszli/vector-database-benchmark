"""box master added

Revision ID: fad64b75dcdf
Revises: 087edfd5bea6
Create Date: 2023-02-06 00:01:10.408590

"""
from alembic import op
import sqlalchemy as sa
revision = 'fad64b75dcdf'
down_revision = '087edfd5bea6'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('product', sa.Column('box', sa.Integer(), nullable=True))
    op.add_column('product', sa.Column('master', sa.Integer(), nullable=True))
    op.add_column('product', sa.Column('lastupdate', sa.DateTime(), nullable=True))

def downgrade() -> None:
    if False:
        return 10
    op.drop_column('product', 'lastupdate')
    op.drop_column('product', 'master')
    op.drop_column('product', 'box')