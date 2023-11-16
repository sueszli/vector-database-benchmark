"""
Set User.is_staff to nullable

Revision ID: 62f6becc7653
Revises: 522918187b73
Create Date: 2018-08-17 16:16:20.397865
"""
from alembic import op
revision = '62f6becc7653'
down_revision = '522918187b73'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('accounts_user', 'is_staff', nullable=True)

def downgrade():
    if False:
        return 10
    op.alter_column('accounts_user', 'is_staff', nullable=False)