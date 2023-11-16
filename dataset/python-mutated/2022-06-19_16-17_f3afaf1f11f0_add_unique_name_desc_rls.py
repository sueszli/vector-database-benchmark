"""add_unique_name_desc_rls

Revision ID: f3afaf1f11f0
Revises: e09b4ae78457
Create Date: 2022-06-19 16:17:23.318618

"""
revision = 'f3afaf1f11f0'
down_revision = 'e09b4ae78457'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
Base = declarative_base()

class RowLevelSecurityFilter(Base):
    __tablename__ = 'row_level_security_filters'
    id = sa.Column(sa.Integer, primary_key=True)
    name = sa.Column(sa.String(255), unique=True, nullable=False)

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    session = Session(bind=bind)
    op.add_column('row_level_security_filters', sa.Column('name', sa.String(length=255)))
    op.add_column('row_level_security_filters', sa.Column('description', sa.Text(), nullable=True))
    all_rls = session.query(RowLevelSecurityFilter).all()
    for rls in all_rls:
        rls.name = f'rls-{rls.id}'
    session.commit()
    with op.batch_alter_table('row_level_security_filters') as batch_op:
        batch_op.alter_column('name', existing_type=sa.String(255), nullable=False)
        batch_op.create_unique_constraint('uq_rls_name', ['name'])

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('row_level_security_filters') as batch_op:
        batch_op.drop_constraint('uq_rls_name', type_='unique')
        batch_op.drop_column('description')
        batch_op.drop_column('name')