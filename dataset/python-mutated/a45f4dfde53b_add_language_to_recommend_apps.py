"""add language to recommend apps

Revision ID: a45f4dfde53b
Revises: 9f4e3427ea84
Create Date: 2023-05-25 17:50:32.052335

"""
from alembic import op
import sqlalchemy as sa
revision = 'a45f4dfde53b'
down_revision = '9f4e3427ea84'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('recommended_apps', schema=None) as batch_op:
        batch_op.add_column(sa.Column('language', sa.String(length=255), server_default=sa.text("'en-US'::character varying"), nullable=False))
        batch_op.drop_index('recommended_app_is_listed_idx')
        batch_op.create_index('recommended_app_is_listed_idx', ['is_listed', 'language'], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('recommended_apps', schema=None) as batch_op:
        batch_op.drop_index('recommended_app_is_listed_idx')
        batch_op.create_index('recommended_app_is_listed_idx', ['is_listed'], unique=False)
        batch_op.drop_column('language')