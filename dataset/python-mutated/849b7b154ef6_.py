"""Add InteractiveSessionEvent model and event types

Revision ID: 849b7b154ef6
Revises: ad0b4cda3e50
Create Date: 2022-05-13 14:35:48.413549

"""
import sqlalchemy as sa
from alembic import op
revision = '849b7b154ef6'
down_revision = 'ad0b4cda3e50'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.execute('\n        alter table event_types alter column name type character varying(150);\n        alter table events alter column type type character varying(150);\n        alter table subscriptions alter column event_type type character varying(150);\n        ')
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:pipeline:interactive-session:started'),\n        ('project:pipeline:interactive-session:stopped'),\n        ('project:pipeline:interactive-session:service-restarted'),\n        ('project:pipeline:interactive-session:failed'),\n        ('project:pipeline:interactive-session:succeeded')\n        ;\n        ")
    op.add_column('events', sa.Column('pipeline_uuid', sa.String(length=36), nullable=True))
    op.create_foreign_key(op.f('fk_events_project_uuid_pipeline_uuid_interactive_sessions'), 'events', 'interactive_sessions', ['project_uuid', 'pipeline_uuid'], ['project_uuid', 'pipeline_uuid'], ondelete='CASCADE')

def downgrade():
    if False:
        print('Hello World!')
    op.drop_constraint(op.f('fk_events_project_uuid_pipeline_uuid_interactive_sessions'), 'events', type_='foreignkey')
    op.drop_column('events', 'pipeline_uuid')