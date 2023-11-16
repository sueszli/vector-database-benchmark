"""adding verbose_name to druid column

Revision ID: b318dfe5fb6c
Revises: d6db5a5cdb5d
Create Date: 2017-03-08 11:48:10.835741

"""
revision = 'b318dfe5fb6c'
down_revision = 'd6db5a5cdb5d'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('columns', sa.Column('verbose_name', sa.String(length=1024), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('columns', 'verbose_name')