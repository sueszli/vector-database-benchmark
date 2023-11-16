"""update time grain SQLA

Revision ID: 32646df09c64
Revises: 60dc453f4e2e
Create Date: 2021-10-12 11:15:25.559532

"""
revision = '32646df09c64'
down_revision = '60dc453f4e2e'
import json
from alembic import op
from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)

def migrate(mapping: dict[str, str]) -> None:
    if False:
        return 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).all():
        try:
            params = json.loads(slc.params)
            time_grain_sqla = params.get('time_grain_sqla')
            if time_grain_sqla in mapping:
                params['time_grain_sqla'] = mapping[time_grain_sqla]
                slc.params = json.dumps(params, sort_keys=True)
        except Exception:
            pass
    session.commit()
    session.close()

def upgrade():
    if False:
        i = 10
        return i + 15
    migrate(mapping={'PT0.5H': 'PT30M', 'P0.25Y': 'P3M'})

def downgrade():
    if False:
        return 10
    migrate(mapping={'PT30M': 'PT0.5H', 'P3M': 'P0.25Y'})