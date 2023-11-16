"""Increase note max size

Revision ID: 1c847ae1209a
Revises: bbd695323107
Create Date: 2016-07-29 11:18:00.550197

"""
revision = '1c847ae1209a'
down_revision = '2ce75615b24d'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        while True:
            i = 10
    op.alter_column('itemaudit', 'notes', type_=sa.VARCHAR(1024), existing_type=sa.VARCHAR(length=512), nullable=True)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('itemaudit', 'notes', type_=sa.VARCHAR(512), existing_type=sa.VARCHAR(length=1024), nullable=True)