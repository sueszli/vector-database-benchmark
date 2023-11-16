"""add primary key cols

Revision ID: d9092588866f
Revises: e62c379ac8f4
Create Date: 2023-03-03 14:20:07.082211

"""
import sqlalchemy as db
from alembic import op
from dagster._core.storage.migration.utils import get_primary_key, has_column, has_table
from sqlalchemy.dialects import sqlite
revision = 'd9092588866f'
down_revision = 'e62c379ac8f4'
branch_labels = None
depends_on = None

def _create_primary_key(tablename):
    if False:
        for i in range(10):
            print('nop')
    if op.get_context().dialect.name == 'sqlite':
        with op.batch_alter_table(tablename, recreate='always') as batch_op:
            batch_op.add_column(db.Column('id', db.BigInteger().with_variant(sqlite.INTEGER(), 'sqlite'), primary_key=True, autoincrement=True))
    elif op.get_context().dialect.name == 'mysql':
        primary_key = get_primary_key(tablename)
        if primary_key and primary_key.get('constrained_columns') == ['my_row_id']:
            op.execute(f'ALTER TABLE {tablename} ALTER COLUMN my_row_id SET VISIBLE')
            op.execute(f'ALTER TABLE {tablename} RENAME COLUMN my_row_id TO id')
        else:
            op.execute(f'ALTER TABLE {tablename} ADD COLUMN id BIGINT PRIMARY KEY AUTO_INCREMENT')
    else:
        op.add_column(tablename, db.Column('id', db.BigInteger(), primary_key=True, autoincrement=True))

def upgrade():
    if False:
        i = 10
        return i + 15
    if has_table('kvs') and (not has_column('kvs', 'id')):
        _create_primary_key('kvs')
    if has_table('instance_info') and (not has_column('instance_info', 'id')):
        _create_primary_key('instance_info')
    if has_table('daemon_heartbeats') and (not has_column('daemon_heartbeats', 'id')):
        _create_primary_key('daemon_heartbeats')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass