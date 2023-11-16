"""Add EnvironmentEvent model and related event_types

Revision ID: 19ce7297c194
Revises: a4b1f48ddab5
Create Date: 2022-05-18 08:24:35.071687

"""
import sqlalchemy as sa
from alembic import op
revision = '19ce7297c194'
down_revision = 'a4b1f48ddab5'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:environment:created'),\n        ('project:environment:deleted')\n        ;\n        ")
    op.add_column('events', sa.Column('environment_uuid', sa.String(length=36), nullable=True))
    op.create_foreign_key(op.f('fk_events_project_uuid_environment_uuid_environments'), 'events', 'environments', ['project_uuid', 'environment_uuid'], ['project_uuid', 'uuid'], ondelete='CASCADE')

def downgrade():
    if False:
        return 10
    op.drop_constraint(op.f('fk_events_project_uuid_environment_uuid_environments'), 'events', type_='foreignkey')
    op.drop_column('events', 'environment_uuid')