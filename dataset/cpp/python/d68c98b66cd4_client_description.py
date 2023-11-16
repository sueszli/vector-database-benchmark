"""client-description

Revision ID: d68c98b66cd4
Revises: 99a28a4418e1
Create Date: 2018-04-13 10:50:17.968636

"""
# revision identifiers, used by Alembic.
revision = 'd68c98b66cd4'
down_revision = '99a28a4418e1'
branch_labels = None
depends_on = None

import sqlalchemy as sa
from alembic import op


def upgrade():
    engine = op.get_bind().engine
    tables = sa.inspect(engine).get_table_names()
    if 'oauth_clients' in tables:
        op.add_column(
            'oauth_clients', sa.Column('description', sa.Unicode(length=1023))
        )


def downgrade():
    op.drop_column('oauth_clients', 'description')
