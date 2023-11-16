"""Remove unused interactive session fields.

Revision ID: e38ac5633523
Revises: da828f0ba13b
Create Date: 2022-01-25 17:36:45.949861

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'e38ac5633523'
down_revision = 'da828f0ba13b'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.drop_constraint('uq_interactive_sessions_jupyter_server_ip', 'interactive_sessions', type_='unique')
    op.drop_constraint('uq_interactive_sessions_notebook_server_info', 'interactive_sessions', type_='unique')
    op.drop_column('interactive_sessions', 'jupyter_server_ip')
    op.drop_column('interactive_sessions', 'container_ids')
    op.drop_column('interactive_sessions', 'notebook_server_info')

def downgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('interactive_sessions', sa.Column('notebook_server_info', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('interactive_sessions', sa.Column('container_ids', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True))
    op.add_column('interactive_sessions', sa.Column('jupyter_server_ip', sa.VARCHAR(length=15), autoincrement=False, nullable=True))
    op.create_unique_constraint('uq_interactive_sessions_notebook_server_info', 'interactive_sessions', ['notebook_server_info'])
    op.create_unique_constraint('uq_interactive_sessions_jupyter_server_ip', 'interactive_sessions', ['jupyter_server_ip'])