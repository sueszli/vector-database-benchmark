"""Adding metric warning_text

Revision ID: 19a814813610
Revises: ca69c70ec99b
Create Date: 2017-09-15 15:09:40.495345

"""
revision = '19a814813610'
down_revision = 'ca69c70ec99b'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('metrics', sa.Column('warning_text', sa.Text(), nullable=True))
    op.add_column('sql_metrics', sa.Column('warning_text', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('sql_metrics') as batch_op_sql_metrics:
        batch_op_sql_metrics.drop_column('warning_text')
    with op.batch_alter_table('metrics') as batch_op_metrics:
        batch_op_metrics.drop_column('warning_text')