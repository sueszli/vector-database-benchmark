"""change_fetch_values_predicate_to_text

Revision ID: 07071313dd52
Revises: 6d20ba9ecb33
Create Date: 2021-08-09 17:32:56.204184

"""
revision = '07071313dd52'
down_revision = '6d20ba9ecb33'
import logging
import sqlalchemy as sa
from alembic import op
from sqlalchemy import func
from superset import db
from superset.connectors.sqla.models import SqlaTable

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('tables') as batch_op:
        batch_op.alter_column('fetch_values_predicate', existing_type=sa.String(length=1000), type_=sa.Text(), existing_nullable=True)

def remove_value_if_too_long():
    if False:
        return 10
    bind = op.get_bind()
    session = db.Session(bind=bind)
    try:
        rows = session.query(SqlaTable).filter(func.length(SqlaTable.fetch_values_predicate) > 1000).all()
        for row in rows:
            row.fetch_values_predicate = None
        logging.info('%d values deleted', len(rows))
        session.commit()
        session.close()
    except Exception as ex:
        logging.warning(ex)

def downgrade():
    if False:
        return 10
    remove_value_if_too_long()
    with op.batch_alter_table('tables') as batch_op:
        batch_op.alter_column('fetch_values_predicate', existing_type=sa.Text(), type_=sa.String(length=1000), existing_nullable=True)