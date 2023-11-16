"""add_save_form_column_to_db_model

Revision ID: 453530256cea
Revises: f1410ed7ec95
Create Date: 2021-04-30 10:55:07.009994

"""
revision = '453530256cea'
down_revision = 'f1410ed7ec95'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('dbs') as batch_op:
        batch_op.add_column(sa.Column('configuration_method', sa.VARCHAR(255), server_default='sqlalchemy_form'))

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('dbs') as batch_op:
        batch_op.drop_column('configuration_method')