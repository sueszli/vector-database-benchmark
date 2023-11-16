"""add builder description format

Revision ID: 061
Revises: 060

"""
import sqlalchemy as sa
from alembic import op
revision = '061'
down_revision = '060'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('builders') as batch_op:
        batch_op.add_column(sa.Column('description_format', sa.Text, nullable=True))
        batch_op.add_column(sa.Column('description_html', sa.Text, nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('builders', 'description_format')
    op.drop_column('builders', 'description_html')