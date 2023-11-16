"""UpdateProductTableOrder

Revision ID: c8ba2e311e15
Revises: 1a47f164770d
Create Date: 2023-01-18 12:37:00.114310

"""
from alembic import op
import sqlalchemy as sa
revision = 'c8ba2e311e15'
down_revision = '1a47f164770d'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    pass

def downgrade() -> None:
    if False:
        print('Hello World!')
    pass