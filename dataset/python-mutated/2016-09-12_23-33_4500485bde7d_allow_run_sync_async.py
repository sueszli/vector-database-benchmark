"""allow_run_sync_async

Revision ID: 4500485bde7d
Revises: 41f6a59a61f2
Create Date: 2016-09-12 23:33:14.789632

"""
revision = '4500485bde7d'
down_revision = '41f6a59a61f2'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('dbs', sa.Column('allow_run_async', sa.Boolean(), nullable=True))
    op.add_column('dbs', sa.Column('allow_run_sync', sa.Boolean(), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    try:
        op.drop_column('dbs', 'allow_run_sync')
        op.drop_column('dbs', 'allow_run_async')
    except Exception:
        pass