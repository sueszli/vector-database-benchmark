"""initial

Revision ID: 059
Revises: (none)
Create Date: 2021-09-07 20:00:00.000000

This empty Alembic revision is used as a placeholder revision for upgrades from older versions
of the database.
"""
revision = '059'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    pass

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass