"""create run_id idx.

Revision ID: 07f83cc13695
Revises: 5c18fd3c2957
Create Date: 2020-06-11 09:27:11.922143

"""
from alembic import op
from sqlalchemy import inspect
revision = '07f83cc13695'
down_revision = '5c18fd3c2957'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if 'event_logs' in has_tables:
        indices = [x.get('name') for x in inspector.get_indexes('event_logs')]
        if 'idx_run_id' not in indices:
            op.create_index('idx_run_id', 'event_logs', ['run_id'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if 'event_logs' in has_tables:
        indices = [x.get('name') for x in inspector.get_indexes('event_logs')]
        if 'idx_run_id' in indices:
            op.drop_index('idx_run_id', 'event_logs')