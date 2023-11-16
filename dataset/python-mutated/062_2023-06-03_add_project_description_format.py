"""add project description format

Revision ID: 062
Revises: 061

"""
import sqlalchemy as sa
from alembic import op
revision = '062'
down_revision = '061'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('projects') as batch_op:
        batch_op.add_column(sa.Column('description_format', sa.Text, nullable=True))
        batch_op.add_column(sa.Column('description_html', sa.Text, nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('projects', 'description_format')
    op.drop_column('projects', 'description_html')