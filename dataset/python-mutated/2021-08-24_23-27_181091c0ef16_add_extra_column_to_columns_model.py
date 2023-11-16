"""add_extra_column_to_columns_model

Revision ID: 181091c0ef16
Revises: 07071313dd52
Create Date: 2021-08-24 23:27:30.403308

"""
revision = '181091c0ef16'
down_revision = '021b81fe4fbb'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('table_columns') as batch_op:
        batch_op.add_column(sa.Column('extra', sa.Text(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('table_columns') as batch_op:
        batch_op.drop_column('extra')