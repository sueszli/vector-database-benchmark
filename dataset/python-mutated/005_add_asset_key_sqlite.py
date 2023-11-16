"""add_asset_key.

Revision ID: c39c047fa021
Revises: 727ffe943a9f
Create Date: 2020-04-28 09:35:54.768791

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect
revision = 'c39c047fa021'
down_revision = '727ffe943a9f'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if 'event_logs' in has_tables:
        columns = [x.get('name') for x in inspector.get_columns('event_logs')]
        if 'asset_key' not in columns:
            op.add_column('event_logs', sa.Column('asset_key', sa.String))
            op.create_index('idx_asset_key', 'event_logs', ['asset_key'], unique=False)
            op.create_index('idx_step_key', 'event_logs', ['step_key'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if 'event_logs' in has_tables:
        columns = [x.get('name') for x in inspector.get_columns('event_logs')]
        if 'asset_key' in columns:
            op.drop_column('event_logs', 'asset_key')
            op.drop_index('idx_asset_key', 'event_logs')
            op.drop_index('idx_step_key', 'event_logs')