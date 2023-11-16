"""add fetch values predicate

Revision ID: 732f1c06bcbf
Revises: d6db5a5cdb5d
Create Date: 2017-03-03 09:15:56.800930

"""
revision = '732f1c06bcbf'
down_revision = 'd6db5a5cdb5d'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('datasources', sa.Column('fetch_values_from', sa.String(length=100), nullable=True))
    op.add_column('tables', sa.Column('fetch_values_predicate', sa.String(length=1000), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('tables', 'fetch_values_predicate')
    op.drop_column('datasources', 'fetch_values_from')