"""Added lang column for ISO-639-1 codes

Revision ID: ef0b52902560
Revises: d24b37426857
Create Date: 2022-12-28 18:24:21.393973

"""
import sqlalchemy as sa
import sqlmodel
from alembic import op
revision = 'ef0b52902560'
down_revision = 'd24b37426857'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.add_column('post', sa.Column('lang', sqlmodel.sql.sqltypes.AutoString(length=200), nullable=False, default='en-US'))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('post', 'lang')