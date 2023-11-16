"""Add json_metadata to the tables table.

Revision ID: b46fa1b0b39e
Revises: ef8843b41dac
Create Date: 2016-10-05 11:30:31.748238

"""
revision = 'b46fa1b0b39e'
down_revision = 'ef8843b41dac'
import logging
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('tables', sa.Column('params', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    try:
        op.drop_column('tables', 'params')
    except Exception as ex:
        logging.warning(str(ex))