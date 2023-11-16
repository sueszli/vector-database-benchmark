"""Add published column to dashboards

Revision ID: d6ffdf31bdd4
Revises: b4a38aa87893
Create Date: 2018-03-30 14:00:44.929483

"""
revision = 'd6ffdf31bdd4'
down_revision = 'b4a38aa87893'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('dashboards') as batch_op:
        batch_op.add_column(sa.Column('published', sa.Boolean(), nullable=True))
    op.execute("UPDATE dashboards SET published='1'")

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('dashboards') as batch_op:
        batch_op.drop_column('published')