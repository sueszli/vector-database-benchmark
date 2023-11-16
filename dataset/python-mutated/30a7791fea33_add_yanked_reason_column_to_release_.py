"""
Add yanked_reason column to Release table

Revision ID: 30a7791fea33
Revises: 43b0e796a40d
Create Date: 2020-05-09 20:25:19.454034
"""
import sqlalchemy as sa
from alembic import op
revision = '30a7791fea33'
down_revision = '43b0e796a40d'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('releases', sa.Column('yanked_reason', sa.Text(), server_default='', nullable=False))

def downgrade():
    if False:
        return 10
    op.drop_column('releases', 'yanked_reason')