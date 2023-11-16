"""Add PipelineEvent, InteractivePipelineRunEvent model and event types

Revision ID: 637920f5715f
Revises: 849b7b154ef6
Create Date: 2022-05-16 09:25:56.549523

"""
import sqlalchemy as sa
from alembic import op
revision = '637920f5715f'
down_revision = '849b7b154ef6'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.execute("\n        INSERT INTO event_types (name) values\n        ('project:pipeline:interactive-session:pipeline-run:created'),\n        ('project:pipeline:interactive-session:pipeline-run:started'),\n        ('project:pipeline:interactive-session:pipeline-run:cancelled'),\n        ('project:pipeline:interactive-session:pipeline-run:failed'),\n        ('project:pipeline:interactive-session:pipeline-run:succeeded')\n        ;\n        ")
    op.create_foreign_key(op.f('fk_events_project_uuid_pipeline_uuid_pipelines'), 'events', 'pipelines', ['project_uuid', 'pipeline_uuid'], ['project_uuid', 'uuid'], ondelete='CASCADE')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_constraint(op.f('fk_events_project_uuid_pipeline_uuid_pipelines'), 'events', type_='foreignkey')