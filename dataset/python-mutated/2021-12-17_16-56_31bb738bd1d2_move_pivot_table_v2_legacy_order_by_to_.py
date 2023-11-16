"""move_pivot_table_v2_legacy_order_by_to_timeseries_limit_metric

Revision ID: 31bb738bd1d2
Revises: fe23025b9441
Create Date: 2021-12-17 16:56:55.186285

"""
revision = '31bb738bd1d2'
down_revision = 'fe23025b9441'
import json
import logging
from alembic import op
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()
logger = logging.getLogger('alembic')

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)
    viz_type = Column(String(250))

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    slices = session.query(Slice).filter(Slice.viz_type == 'pivot_table_v2').all()
    for slc in slices:
        try:
            params = json.loads(slc.params)
            legacy_order_by = params.pop('legacy_order_by', None)
            if legacy_order_by:
                params['series_limit_metric'] = legacy_order_by
            slc.params = json.dumps(params, sort_keys=True)
        except Exception as e:
            logger.exception(f'An error occurred: parsing params for slice {slc.id} failed.You need to fix it before upgrading your DB.')
            raise e
    session.commit()
    session.close()

def downgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    slices = session.query(Slice).filter(Slice.viz_type == 'pivot_table_v2').all()
    for slc in slices:
        try:
            params = json.loads(slc.params)
            series_limit_metric = params.pop('series_limit_metric', None)
            if series_limit_metric:
                params['legacy_order_by'] = series_limit_metric
            slc.params = json.dumps(params, sort_keys=True)
        except Exception as e:
            logger.exception(f'An error occurred: parsing params for slice {slc.id} failed. You need to fix it before downgrading your DB.')
            raise e
    session.commit()
    session.close()