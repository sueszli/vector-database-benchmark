"""persist user_options

Revision ID: 4dc2d5a8c53c
Revises: 896818069c98
Create Date: 2019-02-28 14:14:27.423927

"""
# revision identifiers, used by Alembic.
revision = '4dc2d5a8c53c'
down_revision = '896818069c98'
branch_labels = None
depends_on = None

import sqlalchemy as sa
from alembic import op

from jupyterhub.orm import JSONDict


def upgrade():
    engine = op.get_bind().engine
    tables = sa.inspect(engine).get_table_names()
    if 'spawners' in tables:
        op.add_column('spawners', sa.Column('user_options', JSONDict()))


def downgrade():
    op.drop_column('spawners', sa.Column('user_options'))
