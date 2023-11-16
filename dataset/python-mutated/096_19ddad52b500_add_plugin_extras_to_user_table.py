"""Add plugin_extras to user table

Revision ID: 19ddad52b500
Revises: 9fadda785b07
Create Date: 2020-05-12 22:19:37.878470

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = u'19ddad52b500'
down_revision = u'9fadda785b07'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column(u'user', sa.Column(u'plugin_extras', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column(u'user', u'plugin_extras')