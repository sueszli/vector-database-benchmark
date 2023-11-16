"""Migrate jobs.total_scheduled_pipeline_runs values.


Revision ID: da828f0ba13b
Revises: 7c2f9f12f9ca
Create Date: 2021-12-21 15:11:58.657960

"""
from alembic import op
revision = 'da828f0ba13b'
down_revision = '7c2f9f12f9ca'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.execute('\n        WITH tmp as (\n        SELECT jobs.uuid, runs.count FROM jobs JOIN (SELECT job_uuid, count(*) FROM\n        pipeline_runs WHERE job_uuid IS NOT NULL GROUP BY job_uuid) AS runs ON\n        jobs.uuid = runs.job_uuid)\n        UPDATE jobs\n        SET total_scheduled_pipeline_runs = tmp.count\n        FROM tmp\n        WHERE jobs.uuid = tmp.uuid;\n        ')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('UPDATE jobs set total_scheduled_pipeline_runs = 0;')