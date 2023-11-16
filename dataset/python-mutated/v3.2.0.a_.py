"""Add index to study_id column in trials table

Revision ID: v3.2.0.a
Revises: v3.0.0.d
Create Date: 2023-02-25 13:21:00.730272

"""
from alembic import op
revision = 'v3.2.0.a'
down_revision = 'v3.0.0.d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.create_index(op.f('trials_study_id_key'), 'trials', ['study_id'], unique=False)

def downgrade():
    if False:
        print('Hello World!')
    op.drop_index(op.f('trials_study_id_key'), table_name='trials')