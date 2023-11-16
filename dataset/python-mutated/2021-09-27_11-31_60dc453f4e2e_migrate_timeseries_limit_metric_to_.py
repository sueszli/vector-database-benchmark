"""migrate timeseries_limit_metric to legacy_order_by in pivot_table_v2

Revision ID: 60dc453f4e2e
Revises: 3ebe0993c770
Create Date: 2021-09-27 11:31:53.453164

"""
revision = '60dc453f4e2e'
down_revision = '3ebe0993c770'
import json
from alembic import op
from sqlalchemy import and_, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    slice_name = Column(Text)
    viz_type = Column(String(250))
    params = Column(Text)

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    where_clause = and_(Slice.viz_type == 'pivot_table_v2', Slice.params.like('%"timeseries_limit_metric%'))
    slices = session.query(Slice).filter(where_clause)
    total = slices.count()
    idx = 0
    for slc in slices.yield_per(100):
        idx += 1
        print(f'Upgrading ({idx}/{total}): {slc.slice_name}#{slc.id}')
        params = json.loads(slc.params)
        params['legacy_order_by'] = params.pop('timeseries_limit_metric', None)
        slc.params = json.dumps(params, sort_keys=True, indent=4)
        session.commit()
    session.close()

def downgrade():
    if False:
        return 10
    pass