"""add schema to table model

Revision ID: bb51420eaf83
Revises: 867bf4f117f9
Create Date: 2016-04-11 22:41:06.185955

"""
revision = 'bb51420eaf83'
down_revision = '867bf4f117f9'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('tables', sa.Column('schema', sa.String(length=255), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('tables', 'schema')