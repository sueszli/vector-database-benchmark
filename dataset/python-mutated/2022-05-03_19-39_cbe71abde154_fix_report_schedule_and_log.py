"""fix report schedule and execution log

Revision ID: cbe71abde154
Revises: a9422eeaae74
Create Date: 2022-05-03 19:39:32.074608

"""
revision = 'cbe71abde154'
down_revision = 'a9422eeaae74'
from alembic import op
from sqlalchemy import Column, Float, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
from superset.reports.models import ReportState
Base = declarative_base()

class ReportExecutionLog(Base):
    __tablename__ = 'report_execution_log'
    id = Column(Integer, primary_key=True)
    state = Column(String(50), nullable=False)
    value = Column(Float)
    value_row_json = Column(Text)

class ReportSchedule(Base):
    __tablename__ = 'report_schedule'
    id = Column(Integer, primary_key=True)
    last_state = Column(String(50))
    last_value = Column(Float)
    last_value_row_json = Column(Text)

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for schedule in session.query(ReportSchedule).filter(ReportSchedule.last_state == ReportState.WORKING).all():
        schedule.last_value = None
        schedule.last_value_row_json = None
    session.commit()
    for execution_log in session.query(ReportExecutionLog).filter(ReportExecutionLog.state == ReportState.WORKING).all():
        execution_log.value = None
        execution_log.value_row_json = None
    session.commit()
    session.close()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass