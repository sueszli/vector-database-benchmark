"""Add ``is_encrypted`` column in ``connection`` table

Revision ID: 1507a7289a2f
Revises: e3a246e0dc1
Create Date: 2015-08-18 18:57:51.927315

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect
revision = '1507a7289a2f'
down_revision = 'e3a246e0dc1'
branch_labels = None
depends_on = None
airflow_version = '1.5.0'
connectionhelper = sa.Table('connection', sa.MetaData(), sa.Column('id', sa.Integer, primary_key=True), sa.Column('is_encrypted'))

def upgrade():
    if False:
        i = 10
        return i + 15
    conn = op.get_bind()
    inspector = inspect(conn)
    if 'connection' in inspector.get_table_names():
        col_names = [c['name'] for c in inspector.get_columns('connection')]
        if 'is_encrypted' in col_names:
            return
    op.add_column('connection', sa.Column('is_encrypted', sa.Boolean, unique=False, default=False))
    conn = op.get_bind()
    conn.execute(connectionhelper.update().values(is_encrypted=False))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('connection', 'is_encrypted')