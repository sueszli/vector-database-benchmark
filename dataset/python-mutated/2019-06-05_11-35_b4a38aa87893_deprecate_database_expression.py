"""deprecate database expression

Revision ID: b4a38aa87893
Revises: ab8c66efdd01
Create Date: 2019-06-05 11:35:16.222519

"""
revision = 'b4a38aa87893'
down_revision = 'ab8c66efdd01'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('table_columns') as batch_op:
        batch_op.drop_column('database_expression')

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('table_columns') as batch_op:
        batch_op.add_column(sa.Column('database_expression', sa.String(255)))