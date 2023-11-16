"""
Add canonical_version column

Revision ID: f7577b6938c1
Revises: b75709859292
Create Date: 2018-02-28 15:54:48.867703
"""
import sqlalchemy as sa
from alembic import op
revision = 'f7577b6938c1'
down_revision = 'b75709859292'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('releases', sa.Column('canonical_version', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('releases', 'canonical_version')