"""empty message

Revision ID: c18bd4186f15
Revises: ('46ba6aaaac97', 'ec1f88a35cc6')
Create Date: 2018-07-24 14:29:41.341098

"""
revision = 'c18bd4186f15'
down_revision = ('46ba6aaaac97', 'ec1f88a35cc6')

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    pass

def downgrade():
    if False:
        print('Hello World!')
    pass