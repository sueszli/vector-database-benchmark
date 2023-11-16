"""change_adhoc_filter_b_from_none_to_empty_array

Revision ID: 7293b0ca7944
Revises: b8d3a24d9131
Create Date: 2022-03-02 16:41:36.350540

"""
revision = '7293b0ca7944'
down_revision = 'ab9a9d86e695'
import json
from alembic import op
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)
    viz_type = Column(String(250))

def upgrade():
    if False:
        return 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).filter(Slice.viz_type == 'mixed_timeseries').all():
        try:
            params = json.loads(slc.params)
            adhoc_filters_b = params.get('adhoc_filters_b', None)
            if not adhoc_filters_b:
                params['adhoc_filters_b'] = []
                slc.params = json.dumps(params, sort_keys=True)
        except Exception:
            pass
    session.commit()
    session.close()

def downgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).filter(Slice.viz_type == 'mixed_timeseries').all():
        try:
            params = json.loads(slc.params)
            adhoc_filters_b = params.get('adhoc_filters_b', [])
            if not adhoc_filters_b:
                del params['adhoc_filters_b']
                slc.params = json.dumps(params, sort_keys=True)
        except Exception:
            pass
    session.commit()
    session.close()