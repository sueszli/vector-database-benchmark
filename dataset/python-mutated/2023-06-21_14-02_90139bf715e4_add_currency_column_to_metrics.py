"""add_currency_column_to_metrics

Revision ID: 90139bf715e4
Revises: 83e1abbe777f
Create Date: 2023-06-21 14:02:08.200955

"""
revision = '90139bf715e4'
down_revision = '83e1abbe777f'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('metrics', sa.Column('currency', sa.String(128), nullable=True))
    op.add_column('sql_metrics', sa.Column('currency', sa.String(128), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('sql_metrics') as batch_op_sql_metrics:
        batch_op_sql_metrics.drop_column('currency')
    with op.batch_alter_table('metrics') as batch_op_metrics:
        batch_op_metrics.drop_column('currency')