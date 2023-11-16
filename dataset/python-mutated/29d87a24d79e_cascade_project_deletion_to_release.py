"""
Cascade Project deletion to Release

Revision ID: 29d87a24d79e
Revises: c0682028c857
Create Date: 2018-03-09 22:37:21.343619
"""
from alembic import op
revision = '29d87a24d79e'
down_revision = 'c0682028c857'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_constraint('releases_name_fkey', 'releases', type_='foreignkey')
    op.create_foreign_key('releases_name_fkey', 'releases', 'packages', ['name'], ['name'], onupdate='CASCADE', ondelete='CASCADE')

def downgrade():
    if False:
        return 10
    op.drop_constraint('releases_name_fkey', 'releases', type_='foreignkey')
    op.create_foreign_key('releases_name_fkey', 'releases', 'packages', ['name'], ['name'], onupdate='CASCADE')