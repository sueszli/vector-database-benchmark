"""Make connection login/password TEXT

Revision ID: bd5dfbe21f88
Revises: f7bf2a57d0a6
Create Date: 2023-09-14 17:16:24.942390

"""
import sqlalchemy as sa
from alembic import op
revision = 'bd5dfbe21f88'
down_revision = 'f7bf2a57d0a6'
branch_labels = None
depends_on = None
airflow_version = '2.8.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply Make connection login/password TEXT'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('login', existing_type=sa.VARCHAR(length=500), type_=sa.Text(), existing_nullable=True)
        batch_op.alter_column('password', existing_type=sa.VARCHAR(length=5000), type_=sa.Text(), existing_nullable=True)

def downgrade():
    if False:
        return 10
    'Unapply Make connection login/password TEXT'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('password', existing_type=sa.Text(), type_=sa.VARCHAR(length=5000), existing_nullable=True)
        batch_op.alter_column('login', existing_type=sa.Text(), type_=sa.VARCHAR(length=500), existing_nullable=True)