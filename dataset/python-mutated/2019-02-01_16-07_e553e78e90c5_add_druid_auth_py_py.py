"""add_druid_auth_py.py

Revision ID: e553e78e90c5
Revises: 18dc26817ad2
Create Date: 2019-02-01 16:07:04.268023

"""
revision = 'e553e78e90c5'
down_revision = '18dc26817ad2'
import sqlalchemy as sa
from alembic import op
from sqlalchemy_utils import EncryptedType

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('clusters', sa.Column('broker_pass', sa.LargeBinary(), nullable=True))
    op.add_column('clusters', sa.Column('broker_user', sa.String(length=255), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('clusters', 'broker_user')
    op.drop_column('clusters', 'broker_pass')