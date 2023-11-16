"""Add EnvironmentImageBuildEvent and related event_types

Revision ID: 0eaf40410361
Revises: 19ce7297c194
Create Date: 2022-05-18 11:05:53.335758

"""
import sqlalchemy as sa
from alembic import op
revision = '0eaf40410361'
down_revision = '19ce7297c194'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:environment:image-build:created'),\n        ('project:environment:image-build:started'),\n        ('project:environment:image-build:cancelled'),\n        ('project:environment:image-build:failed'),\n        ('project:environment:image-build:succeeded')\n        ;\n        ")
    op.add_column('events', sa.Column('image_tag', sa.Integer(), nullable=True))
    op.create_foreign_key(op.f('fk_events_project_uuid_environment_uuid_image_tag_environment_image_builds'), 'events', 'environment_image_builds', ['project_uuid', 'environment_uuid', 'image_tag'], ['project_uuid', 'environment_uuid', 'image_tag'], ondelete='CASCADE')

def downgrade():
    if False:
        print('Hello World!')
    op.drop_constraint(op.f('fk_events_project_uuid_environment_uuid_image_tag_environment_image_builds'), 'events', type_='foreignkey')
    op.drop_column('events', 'image_tag')