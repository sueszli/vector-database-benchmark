"""empty message

Revision ID: c43d1ef75230
Revises: c18beeddd321
Create Date: 2022-03-08 11:59:04.561090

"""
import sqlalchemy as sa
from alembic import op
revision = 'c43d1ef75230'
down_revision = 'c18beeddd321'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.create_foreign_key(op.f('fk_environment_image_builds_project_uuid_environment_uuid_environments'), 'environment_image_builds', 'environments', ['project_uuid', 'environment_uuid'], ['project_uuid', 'uuid'], ondelete='CASCADE')
    op.drop_index('project_uuid', table_name='environment_images')
    op.create_index('project_uuid', 'environment_images', ['environment_uuid', sa.text('tag DESC')], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_index('project_uuid', table_name='environment_images')
    op.create_index('project_uuid', 'environment_images', ['environment_uuid'], unique=False)
    op.drop_constraint(op.f('fk_environment_image_builds_project_uuid_environment_uuid_environments'), 'environment_image_builds', type_='foreignkey')