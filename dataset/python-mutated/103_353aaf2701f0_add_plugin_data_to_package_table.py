"""Add plugin_data to package table

Revision ID: 353aaf2701f0
Revises: ff13667243ed
Create Date: 2022-07-13 23:29:52.681437

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = '353aaf2701f0'
down_revision = 'ff13667243ed'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column(u'package', sa.Column(u'plugin_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column(u'package', u'plugin_data')