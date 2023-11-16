"""rename to schemas_allowed_for_file_upload in dbs.extra

Revision ID: 0ca9e5f1dacd
Revises: b92d69a6643c
Create Date: 2021-11-11 04:18:26.171851

"""
revision = '0ca9e5f1dacd'
down_revision = 'b92d69a6643c'
import json
import logging
from alembic import op
from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from superset import db
Base = declarative_base()

class Database(Base):
    __tablename__ = 'dbs'
    id = Column(Integer, primary_key=True)
    extra = Column(Text)

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for database in session.query(Database).all():
        try:
            extra = json.loads(database.extra)
        except json.decoder.JSONDecodeError as ex:
            logging.warning(str(ex))
            continue
        if 'schemas_allowed_for_csv_upload' in extra:
            extra['schemas_allowed_for_file_upload'] = extra.pop('schemas_allowed_for_csv_upload')
            database.extra = json.dumps(extra)
    session.commit()
    session.close()

def downgrade():
    if False:
        return 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for database in session.query(Database).all():
        try:
            extra = json.loads(database.extra)
        except json.decoder.JSONDecodeError as ex:
            logging.warning(str(ex))
            continue
        if 'schemas_allowed_for_file_upload' in extra:
            extra['schemas_allowed_for_csv_upload'] = extra.pop('schemas_allowed_for_file_upload')
            database.extra = json.dumps(extra)
    session.commit()
    session.close()