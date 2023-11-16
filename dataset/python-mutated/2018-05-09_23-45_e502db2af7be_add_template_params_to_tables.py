"""add template_params to tables

Revision ID: e502db2af7be
Revises: 5ccf602336a0
Create Date: 2018-05-09 23:45:14.296283

"""
revision = 'e502db2af7be'
down_revision = '5ccf602336a0'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('tables', sa.Column('template_params', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    try:
        op.drop_column('tables', 'template_params')
    except Exception as ex:
        logging.warning(str(ex))