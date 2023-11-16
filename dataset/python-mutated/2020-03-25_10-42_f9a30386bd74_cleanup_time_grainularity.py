"""cleanup_time_granularity

Revision ID: f9a30386bd74
Revises: b5998378c225
Create Date: 2020-03-25 10:42:11.047328

"""
revision = 'f9a30386bd74'
down_revision = 'b5998378c225'
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
        for i in range(10):
            print('nop')
    '\n    Remove any erroneous time granularity fields from slices foor those visualization\n    types which do not support time granularity.\n\n    :see: https://github.com/apache/superset/pull/8674\n    :see: https://github.com/apache/superset/pull/8764\n    :see: https://github.com/apache/superset/pull/8800\n    :see: https://github.com/apache/superset/pull/8825\n    '
    bind = op.get_bind()
    session = db.Session(bind=bind)
    viz_types = ['area', 'bar', 'big_number', 'compare', 'dual_line', 'line', 'pivot_table', 'table', 'time_pivot', 'time_table']
    erroneous = ['granularity', 'time_grain_sqla']
    for slc in session.query(Slice).filter(Slice.viz_type.notin_(viz_types)).all():
        try:
            params = json.loads(slc.params)
            if any((field in params for field in erroneous)):
                for field in erroneous:
                    if field in params:
                        del params[field]
                slc.params = json.dumps(params, sort_keys=True)
        except Exception:
            pass
    session.commit()
    session.close()

def downgrade():
    if False:
        while True:
            i = 10
    pass