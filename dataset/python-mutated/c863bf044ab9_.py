"""empty message

Revision ID: c863bf044ab9
Revises: 9eda3c5ad4f6
Create Date: 2022-01-13 12:09:39.994080

"""
from alembic import op
revision = 'c863bf044ab9'
down_revision = '9eda3c5ad4f6'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.execute('\n        WITH run_param_values as (\n        select run_values.uuid, jsonb_agg(run_values.value) as values from\n        (select uuid, (jsonb_each(parameters)).* from pipeline_runs) as run_values\n        group by uuid\n        )\n        UPDATE pipeline_runs\n        SET parameters_text_search_values = run_param_values.values\n        FROM run_param_values\n        WHERE pipeline_runs.uuid = run_param_values.uuid;\n        ')

def downgrade():
    if False:
        i = 10
        return i + 15
    op.execute("UPDATE pipeline_runs set parameters_text_search_values = '[]'::jsonb;")