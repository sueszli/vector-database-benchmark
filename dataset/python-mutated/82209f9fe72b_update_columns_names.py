"""UPdate Columns Names

Revision ID: 82209f9fe72b
Revises: 25c52949e17e
Create Date: 2023-01-18 05:59:20.389625

"""
from alembic import op
import sqlalchemy as sa
revision = '82209f9fe72b'
down_revision = '25c52949e17e'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('product', sa.Column('codebarinner', sa.String(), nullable=True))
    op.add_column('product', sa.Column('codebarmaster', sa.String(), nullable=True))
    op.drop_constraint('product_codebarInner_key', 'product', type_='unique')
    op.drop_constraint('product_codebarMaster_key', 'product', type_='unique')
    op.create_unique_constraint(None, 'product', ['codebarmaster'])
    op.create_unique_constraint(None, 'product', ['codebarinner'])
    op.drop_column('product', 'codebarInner')
    op.drop_column('product', 'codebarMaster')

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('product', sa.Column('codebarMaster', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('product', sa.Column('codebarInner', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.drop_constraint(None, 'product', type_='unique')
    op.drop_constraint(None, 'product', type_='unique')
    op.create_unique_constraint('product_codebarMaster_key', 'product', ['codebarMaster'])
    op.create_unique_constraint('product_codebarInner_key', 'product', ['codebarInner'])
    op.drop_column('product', 'codebarmaster')
    op.drop_column('product', 'codebarinner')