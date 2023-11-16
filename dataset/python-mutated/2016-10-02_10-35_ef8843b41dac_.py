"""empty message

Revision ID: ef8843b41dac
Revises: ('3b626e2a6783', 'ab3d66c4246e')
Create Date: 2016-10-02 10:35:38.825231

"""
revision = 'ef8843b41dac'
down_revision = ('3b626e2a6783', 'ab3d66c4246e')

def upgrade():
    if False:
        i = 10
        return i + 15
    pass

def downgrade():
    if False:
        return 10
    pass