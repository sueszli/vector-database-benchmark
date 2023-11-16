"""UpdateProductTable

Revision ID: 1a47f164770d
Revises: 82209f9fe72b
Create Date: 2023-01-18 12:33:10.867941

"""
from alembic import op
import sqlalchemy as sa
revision = '1a47f164770d'
down_revision = '82209f9fe72b'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    pass

def downgrade() -> None:
    if False:
        print('Hello World!')
    pass