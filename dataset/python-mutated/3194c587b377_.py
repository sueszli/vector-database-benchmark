"""Add schedule to orchest_webserver job model

Revision ID: 3194c587b377
Revises: b92423befebe
Create Date: 2021-01-22 14:13:37.438082

"""
import sqlalchemy as sa
from alembic import op
revision = '3194c587b377'
down_revision = 'b92423befebe'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('jobs', sa.Column('schedule', sa.String(length=100), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('jobs', 'schedule')