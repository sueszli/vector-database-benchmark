"""better_filters

Revision ID: fb13d49b72f9
Revises: 6c7537a6004a
Create Date: 2018-12-11 22:03:21.612516

"""
import json
import logging
from alembic import op
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
revision = 'fb13d49b72f9'
down_revision = 'de021a1ca60d'
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)
    viz_type = Column(String(250))
    slice_name = Column(String(250))

def upgrade_slice(slc):
    if False:
        for i in range(10):
            print('nop')
    params = json.loads(slc.params)
    logging.info(f'Upgrading {slc.slice_name}')
    cols = params.get('groupby')
    metric = params.get('metric')
    if cols:
        flts = [{'column': col, 'metric': metric, 'asc': False, 'clearable': True, 'multiple': True} for col in cols]
        params['filter_configs'] = flts
        if 'groupby' in params:
            del params['groupby']
        if 'metric' in params:
            del params['metric']
        slc.params = json.dumps(params, sort_keys=True)

def upgrade():
    if False:
        return 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    filter_box_slices = session.query(Slice).filter_by(viz_type='filter_box')
    for slc in filter_box_slices.all():
        try:
            upgrade_slice(slc)
        except Exception as ex:
            logging.exception(e)
    session.commit()
    session.close()

def downgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    filter_box_slices = session.query(Slice).filter_by(viz_type='filter_box')
    for slc in filter_box_slices.all():
        try:
            params = json.loads(slc.params)
            logging.info(f'Downgrading {slc.slice_name}')
            flts = params.get('filter_configs')
            if not flts:
                continue
            params['metric'] = flts[0].get('metric')
            params['groupby'] = [o.get('column') for o in flts]
            slc.params = json.dumps(params, sort_keys=True)
        except Exception as ex:
            logging.exception(ex)
    session.commit()
    session.close()