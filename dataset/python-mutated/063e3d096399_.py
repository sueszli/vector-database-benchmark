"""empty message

Revision ID: 063e3d096399
Revises: 191980130560, dfe5c529e6da
Create Date: 2022-05-04 13:07:22.580446

"""
revision = '063e3d096399'
down_revision = ('191980130560', 'dfe5c529e6da')
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    pass

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass