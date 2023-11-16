"""add skip bool & skip_reason to task

Revision ID: 9e7ec4a9e3f2
Revises: 7b8f0011e0b0
Create Date: 2023-02-01 21:46:49.971052

"""
import sqlalchemy as sa
import sqlmodel
from alembic import op
revision = '9e7ec4a9e3f2'
down_revision = '55361f323d12'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.add_column('task', sa.Column('skipped', sa.Boolean(), server_default=sa.text('false'), nullable=False))
    op.add_column('task', sa.Column('skip_reason', sqlmodel.sql.sqltypes.AutoString(length=512), nullable=True))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('task', 'skip_reason')
    op.drop_column('task', 'skipped')