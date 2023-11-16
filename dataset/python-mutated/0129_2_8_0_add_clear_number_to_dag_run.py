"""add new field 'clear_number' to dagrun

Revision ID: 375a816bbbf4
Revises: 405de8318b3a
Create Date: 2023-09-05 19:27:30.531558

"""
import sqlalchemy as sa
from alembic import op
revision = '375a816bbbf4'
down_revision = '405de8318b3a'
branch_labels = None
depends_on = None
airflow_version = '2.8.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Apply add cleared column to dagrun'
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        with op.batch_alter_table('dag_run') as batch_op:
            batch_op.add_column(sa.Column('clear_number', sa.Integer, default=0))
            batch_op.alter_column('clear_number', existing_type=sa.Integer, nullable=False)
    else:
        with op.batch_alter_table('dag_run') as batch_op:
            batch_op.add_column(sa.Column('clear_number', sa.Integer, default=0, nullable=False, server_default='0'))

def downgrade():
    if False:
        print('Hello World!')
    'Unapply add cleared column to pool'
    with op.batch_alter_table('dag_run') as batch_op:
        batch_op.drop_column('clear_number')