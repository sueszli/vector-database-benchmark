"""adding slug to dash

Revision ID: 1a48a5411020
Revises: 289ce07647b
Create Date: 2015-12-04 09:42:16.973264

"""
revision = '1a48a5411020'
down_revision = '289ce07647b'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('dashboards', sa.Column('slug', sa.String(length=255), nullable=True))
    try:
        op.create_unique_constraint('idx_unique_slug', 'dashboards', ['slug'])
    except:
        pass

def downgrade():
    if False:
        return 10
    op.drop_constraint(None, 'dashboards', type_='unique')
    op.drop_column('dashboards', 'slug')