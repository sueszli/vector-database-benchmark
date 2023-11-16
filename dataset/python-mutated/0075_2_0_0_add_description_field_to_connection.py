"""Add description field to ``connection`` table

Revision ID: 61ec73d9401f
Revises: 2c6edca13270
Create Date: 2020-09-10 14:56:30.279248

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '61ec73d9401f'
down_revision = '2c6edca13270'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Add description field to ``connection`` table'
    conn = op.get_bind()
    with op.batch_alter_table('connection') as batch_op:
        if conn.dialect.name == 'mysql':
            batch_op.add_column(sa.Column('description', sa.Text(length=5000), nullable=True))
        else:
            batch_op.add_column(sa.Column('description', sa.String(length=5000), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Add description field to ``connection`` table'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.drop_column('description')