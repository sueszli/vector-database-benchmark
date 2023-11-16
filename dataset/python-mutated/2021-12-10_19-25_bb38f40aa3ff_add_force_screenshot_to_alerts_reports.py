"""Add force_screenshot to alerts/reports

Revision ID: bb38f40aa3ff
Revises: 31bb738bd1d2
Create Date: 2021-12-10 19:25:29.802949

"""
revision = 'bb38f40aa3ff'
down_revision = '31bb738bd1d2'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class ReportSchedule(Base):
    __tablename__ = 'report_schedule'
    id = sa.Column(sa.Integer, primary_key=True)
    type = sa.Column(sa.String(50), nullable=False)
    force_screenshot = sa.Column(sa.Boolean, default=False)
    chart_id = sa.Column(sa.Integer, nullable=True)

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.add_column(sa.Column('force_screenshot', sa.Boolean(), default=False))
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for report in session.query(ReportSchedule).all():
        report.force_screenshot = report.type == 'Alert' and report.chart_id is not None
    session.commit()

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.drop_column('force_screenshot')