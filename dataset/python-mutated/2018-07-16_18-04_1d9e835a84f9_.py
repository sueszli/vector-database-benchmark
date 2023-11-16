"""empty message

Revision ID: 1d9e835a84f9
Revises: 3dda56f1c4c6
Create Date: 2018-07-16 18:04:07.764659

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.sql import expression
revision = '1d9e835a84f9'
down_revision = '3dda56f1c4c6'

def upgrade():
    if False:
        return 10
    op.add_column('dbs', sa.Column('allow_csv_upload', sa.Boolean(), nullable=False, server_default=expression.true()))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('dbs', 'allow_csv_upload')