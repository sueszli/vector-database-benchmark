"""deprecate time_range_endpoints v2

Revision ID: b0d0249074e4
Revises: 2ed890b36b94
Create Date: 2022-04-04 15:04:05.606340

"""
import json
from alembic import op
from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
revision = 'b0d0249074e4'
down_revision = '2ed890b36b94'
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)

def upgrade():
    if False:
        print('Hello World!')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).filter(Slice.params.like('%time_range_endpoints%')):
        params = json.loads(slc.params)
        params.pop('time_range_endpoints', None)
        slc.params = json.dumps(params)
    session.commit()
    session.close()

def downgrade():
    if False:
        i = 10
        return i + 15
    pass