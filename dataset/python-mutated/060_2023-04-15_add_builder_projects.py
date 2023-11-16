"""add builder projects

Revision ID: 060
Revises: 059

"""
import sqlalchemy as sa
from alembic import op
revision = '060'
down_revision = '059'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    hash_length = 40
    op.create_table('projects', sa.Column('id', sa.Integer, primary_key=True), sa.Column('name', sa.Text, nullable=False), sa.Column('name_hash', sa.String(hash_length), nullable=False), sa.Column('slug', sa.String(50), nullable=False), sa.Column('description', sa.Text, nullable=True), mysql_DEFAULT_CHARSET='utf8')
    with op.batch_alter_table('builders') as batch_op:
        batch_op.add_column(sa.Column('projectid', sa.Integer, sa.ForeignKey('projects.id', name='fk_builders_projectid', ondelete='SET NULL'), nullable=True))
    op.create_index('builders_projectid', 'builders', ['projectid'])
    op.create_index('projects_name_hash', 'projects', ['name_hash'], unique=True)

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_index('builders_projectid')
    op.drop_column('builders', 'project')
    op.drop_table('projects')
    op.drop_index('projects_name_hash')