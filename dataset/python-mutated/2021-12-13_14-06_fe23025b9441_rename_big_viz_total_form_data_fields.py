"""rename_big_viz_total_form_data_fields

Revision ID: fe23025b9441
Revises: 3ba29ecbaac5
Create Date: 2021-12-13 14:06:24.426970

"""
revision = 'fe23025b9441'
down_revision = '3ba29ecbaac5'
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
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    slices = session.query(Slice).filter(Slice.viz_type == 'big_number_total').all()
    for slc in slices:
        try:
            params = json.loads(slc.params)
            header_format_selector = params.pop('header_format_selector', None)
            header_timestamp_format = params.pop('header_timestamp_format', None)
            if header_format_selector:
                params['force_timestamp_formatting'] = header_format_selector
            if header_timestamp_format:
                params['time_format'] = header_timestamp_format
            slc.params = json.dumps(params, sort_keys=True)
        except Exception as e:
            logger.exception(f'An error occurred: parsing params for slice {slc.id} failed.You need to fix it before upgrading your DB.')
            raise e
    session.commit()
    session.close()

def downgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    slices = session.query(Slice).filter(Slice.viz_type == 'big_number_total').all()
    for slc in slices:
        try:
            params = json.loads(slc.params)
            time_format = params.pop('time_format', None)
            force_timestamp_formatting = params.pop('force_timestamp_formatting', None)
            if time_format:
                params['header_timestamp_format'] = time_format
            if force_timestamp_formatting:
                params['header_format_selector'] = force_timestamp_formatting
            slc.params = json.dumps(params, sort_keys=True)
        except Exception as e:
            logger.exception(f'An error occurred: parsing params for slice {slc.id} failed. You need to fix it before downgrading your DB.')
            raise e
    session.commit()
    session.close()