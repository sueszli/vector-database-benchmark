"""initial revision for mysql support.

Revision ID: ba4050312958
Revises: 521d4caca7ad
Create Date: 2021-02-22 23:08:27.392579

"""
revision = 'ba4050312958'
down_revision = '521d4caca7ad'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass