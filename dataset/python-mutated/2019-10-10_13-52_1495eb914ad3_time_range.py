"""time range

Revision ID: 1495eb914ad3
Revises: 258b5280a45e
Create Date: 2019-10-10 13:52:54.544475

"""
import json
import logging
from alembic import op
from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
from superset.legacy import update_time_range
revision = '1495eb914ad3'
down_revision = '258b5280a45e'
Base = declarative_base()

class Slice(Base):
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    params = Column(Text)

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).all():
        try:
            form_data = json.loads(slc.params)
            update_time_range(form_data)
            slc.params = json.dumps(form_data, sort_keys=True)
        except Exception as ex:
            logging.exception(ex)
    session.commit()

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for slc in session.query(Slice).all():
        try:
            form_data = json.loads(slc.params)
            if 'time_range' in form_data:
                try:
                    (since, until) = form_data['time_range'].split(' : ')
                    form_data['since'] = since
                    form_data['until'] = until
                    del form_data['time_range']
                    slc.params = json.dumps(form_data, sort_keys=True)
                except ValueError:
                    pass
        except Exception as ex:
            logging.exception(ex)
    session.commit()