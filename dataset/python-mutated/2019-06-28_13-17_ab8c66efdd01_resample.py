"""resample

Revision ID: ab8c66efdd01
Revises: d7c1a0d6f2da
Create Date: 2019-06-28 13:17:59.517089

"""
revision = 'ab8c66efdd01'
down_revision = 'd7c1a0d6f2da'
import json
import logging
from alembic import op
from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).all():
        try:
            params = json.loads(slc.params)
            if 'resample_rule' in params:
                rule = params['resample_rule']
                if rule:
                    how = None
                    if 'resample_how' in params:
                        how = params['resample_how']
                        if how:
                            params['resample_method'] = how
                    if not how and 'fill_method' in params:
                        fill_method = params['resample_fillmethod']
                        if fill_method:
                            params['resample_method'] = fill_method
                    if not 'resample_method' in params:
                        del params['resample_rule']
                else:
                    del params['resample_rule']
                params.pop('resample_fillmethod', None)
                params.pop('resample_how', None)
                slc.params = json.dumps(params, sort_keys=True)
        except Exception as ex:
            logging.exception(ex)
    session.commit()
    session.close()

def downgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).all():
        try:
            params = json.loads(slc.params)
            if 'resample_method' in params:
                method = params['resample_method']
                if method in ['asfreq', 'bfill', 'ffill']:
                    params['resample_fillmethod'] = method
                else:
                    params['resample_how'] = method
                del params['resample_method']
                slc.params = json.dumps(params, sort_keys=True)
        except Exception as ex:
            logging.exception(ex)
    session.commit()
    session.close()