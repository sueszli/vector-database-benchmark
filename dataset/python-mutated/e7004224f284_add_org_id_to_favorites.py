"""add_org_id_to_favorites

Revision ID: e7004224f284
Revises: d4c798575877
Create Date: 2018-05-10 09:46:31.169938

"""
from alembic import op
import sqlalchemy as sa
revision = 'e7004224f284'
down_revision = 'd4c798575877'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.add_column('favorites', sa.Column('org_id', sa.Integer(), nullable=False))
    op.create_foreign_key(None, 'favorites', 'organizations', ['org_id'], ['id'])

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_constraint(None, 'favorites', type_='foreignkey')
    op.drop_column('favorites', 'org_id')