"""empty message

Revision ID: 5a7bad26f2a7
Revises: 4e6a06bad7a8
Create Date: 2015-10-05 10:32:15.850753

"""
revision = '5a7bad26f2a7'
down_revision = '4e6a06bad7a8'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('dashboards', sa.Column('css', sa.Text(), nullable=True))
    op.add_column('dashboards', sa.Column('description', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('dashboards', 'description')
    op.drop_column('dashboards', 'css')