"""sequoia_pgp

Revision ID: 811334d7105f
Revises: c5a02eb52f2d
Create Date: 2023-06-29 18:19:59.314380

"""
import sqlalchemy as sa
from alembic import op
revision = '811334d7105f'
down_revision = 'c5a02eb52f2d'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.add_column(sa.Column('pgp_fingerprint', sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column('pgp_public_key', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('pgp_secret_key', sa.Text(), nullable=True))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    pass