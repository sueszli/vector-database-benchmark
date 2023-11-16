"""add run tags index.

Revision ID: c9159e740d7e
Revises: c34498c29964
Create Date: 2020-12-01 12:19:34.460760

"""
from alembic import op
from sqlalchemy import inspect
revision = 'c9159e740d7e'
down_revision = 'c34498c29964'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if 'run_tags' in has_tables:
        indices = [x.get('name') for x in inspector.get_indexes('run_tags')]
        if 'idx_run_tags' not in indices:
            op.create_index('idx_run_tags', 'run_tags', ['key', 'value'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    inspector = inspect(op.get_bind())
    has_tables = inspector.get_table_names()
    if 'run_tags' in has_tables:
        indices = [x.get('name') for x in inspector.get_indexes('run_tags')]
        if 'idx_run_tags' in indices:
            op.drop_index('idx_run_tags', 'run_tags')