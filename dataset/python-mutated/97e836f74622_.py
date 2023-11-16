"""empty message

Revision ID: 97e836f74622
Revises: 8708c2c44585
Create Date: 2022-01-13 13:28:08.049155

"""
import sqlalchemy as sa
from alembic import op
revision = '97e836f74622'
down_revision = '8708c2c44585'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_index('ix_job_pipeline_runs_text_search', 'pipeline_runs', [sa.text("to_tsvector('simple', lower(CAST(pipeline_run_index AS TEXT)) || ' ' || CASE WHEN (status = 'STARTED') THEN 'running' WHEN (status = 'ABORTED') THEN 'cancelled' WHEN (status = 'FAILURE') THEN 'failed' ELSE lower(status) END || ' ' || lower(CAST(parameters_text_search_values AS TEXT)))")], unique=False, postgresql_using='gin')

def downgrade():
    if False:
        print('Hello World!')
    op.drop_index('ix_job_pipeline_runs_text_search', table_name='pipeline_runs', postgresql_using='gin')