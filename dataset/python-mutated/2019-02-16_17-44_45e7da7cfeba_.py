"""empty message

Revision ID: 45e7da7cfeba
Revises: ('e553e78e90c5', 'c82ee8a39623')
Create Date: 2019-02-16 17:44:44.493427

"""
revision = '45e7da7cfeba'
down_revision = ('e553e78e90c5', 'c82ee8a39623')
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    pass

def downgrade():
    if False:
        i = 10
        return i + 15
    pass