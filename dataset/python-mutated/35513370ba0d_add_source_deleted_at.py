"""add Source.deleted_at

Revision ID: 35513370ba0d
Revises: 523fff3f969c
Create Date: 2020-05-06 22:28:01.214359

"""
import sqlalchemy as sa
from alembic import op
revision = '35513370ba0d'
down_revision = '523fff3f969c'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.add_column(sa.Column('deleted_at', sa.DateTime(), nullable=True))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.drop_column('deleted_at')