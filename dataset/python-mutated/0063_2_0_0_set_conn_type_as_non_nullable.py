"""Set ``conn_type`` as non-nullable

Revision ID: 8f966b9c467a
Revises: 3c20cacc0044
Create Date: 2020-06-08 22:36:34.534121

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
revision = '8f966b9c467a'
down_revision = '3c20cacc0044'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Apply Set ``conn_type`` as non-nullable'
    Base = declarative_base()

    class Connection(Base):
        __tablename__ = 'connection'
        id = sa.Column(sa.Integer(), primary_key=True)
        conn_id = sa.Column(sa.String(250))
        conn_type = sa.Column(sa.String(500))
    connection = op.get_bind()
    sessionmaker = sa.orm.sessionmaker()
    session = sessionmaker(bind=connection)
    session.query(Connection).filter_by(conn_id='imap_default', conn_type=None).update({Connection.conn_type: 'imap'}, synchronize_session=False)
    session.commit()
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('conn_type', existing_type=sa.VARCHAR(length=500), nullable=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    'Unapply Set ``conn_type`` as non-nullable'
    with op.batch_alter_table('connection', schema=None) as batch_op:
        batch_op.alter_column('conn_type', existing_type=sa.VARCHAR(length=500), nullable=True)