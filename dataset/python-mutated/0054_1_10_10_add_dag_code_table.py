"""Add ``dag_code`` table

Revision ID: 952da73b5eff
Revises: 852ae6c715af
Create Date: 2020-03-12 12:39:01.797462

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.models.dagcode import DagCode
revision = '952da73b5eff'
down_revision = '852ae6c715af'
branch_labels = None
depends_on = None
airflow_version = '1.10.10'

def upgrade():
    if False:
        while True:
            i = 10
    'Create DagCode Table.'
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base()

    class SerializedDagModel(Base):
        __tablename__ = 'serialized_dag'
        dag_id = sa.Column(sa.String(250), primary_key=True)
        fileloc = sa.Column(sa.String(2000), nullable=False)
        fileloc_hash = sa.Column(sa.BigInteger, nullable=False)
    'Apply add source code table'
    op.create_table('dag_code', sa.Column('fileloc_hash', sa.BigInteger(), nullable=False, primary_key=True, autoincrement=False), sa.Column('fileloc', sa.String(length=2000), nullable=False), sa.Column('source_code', sa.UnicodeText(), nullable=False), sa.Column('last_updated', sa.TIMESTAMP(timezone=True), nullable=False))
    conn = op.get_bind()
    if conn.dialect.name != 'sqlite':
        if conn.dialect.name == 'mssql':
            op.drop_index(index_name='idx_fileloc_hash', table_name='serialized_dag')
        op.alter_column(table_name='serialized_dag', column_name='fileloc_hash', type_=sa.BigInteger(), nullable=False)
        if conn.dialect.name == 'mssql':
            op.create_index(index_name='idx_fileloc_hash', table_name='serialized_dag', columns=['fileloc_hash'])
    sessionmaker = sa.orm.sessionmaker()
    session = sessionmaker(bind=conn)
    serialized_dags = session.query(SerializedDagModel).all()
    for dag in serialized_dags:
        dag.fileloc_hash = DagCode.dag_fileloc_hash(dag.fileloc)
        session.merge(dag)
    session.commit()

def downgrade():
    if False:
        i = 10
        return i + 15
    'Unapply add source code table'
    op.drop_table('dag_code')