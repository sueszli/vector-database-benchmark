"""empty message

Revision ID: ff2c8613b0df
Revises: 97e836f74622, e38ac5633523
Create Date: 2022-02-03 14:00:53.436792

"""
revision = 'ff2c8613b0df'
down_revision = ('97e836f74622', 'e38ac5633523')
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