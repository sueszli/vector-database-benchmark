"""create sqlalchemy_searchable expressions

Revision ID: 7ce5925f832b
Revises: 1038c2174f5d
Create Date: 2023-09-29 16:48:29.517762

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy_searchable import sql_expressions
revision = '7ce5925f832b'
down_revision = '1038c2174f5d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.execute(sql_expressions)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass