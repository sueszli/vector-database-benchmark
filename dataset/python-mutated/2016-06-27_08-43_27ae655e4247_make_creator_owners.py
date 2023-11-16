"""Make creator owners

Revision ID: 27ae655e4247
Revises: d8bc074f7aad
Create Date: 2016-06-27 08:43:52.592242

"""
revision = '27ae655e4247'
down_revision = 'd8bc074f7aad'
from alembic import op
from flask import g
from flask_appbuilder import Model
from sqlalchemy import Column, ForeignKey, Integer, Table
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import relationship
from superset import db
from superset.utils.core import get_user_id
Base = declarative_base()

class User(Base):
    """Declarative class to do query in upgrade"""
    __tablename__ = 'ab_user'
    id = Column(Integer, primary_key=True)
slice_user = Table('slice_user', Base.metadata, Column('id', Integer, primary_key=True), Column('user_id', Integer, ForeignKey('ab_user.id')), Column('slice_id', Integer, ForeignKey('slices.id')))
dashboard_user = Table('dashboard_user', Base.metadata, Column('id', Integer, primary_key=True), Column('user_id', Integer, ForeignKey('ab_user.id')), Column('dashboard_id', Integer, ForeignKey('dashboards.id')))

class AuditMixin:

    @declared_attr
    def created_by_fk(cls):
        if False:
            i = 10
            return i + 15
        return Column(Integer, ForeignKey('ab_user.id'), default=get_user_id, nullable=False)

    @declared_attr
    def created_by(cls):
        if False:
            i = 10
            return i + 15
        return relationship('User', primaryjoin=f'{cls.__name__}.created_by_fk == User.id', enable_typechecks=False)

class Slice(Base, AuditMixin):
    """Declarative class to do query in upgrade"""
    __tablename__ = 'slices'
    id = Column(Integer, primary_key=True)
    owners = relationship('User', secondary=slice_user)

class Dashboard(Base, AuditMixin):
    """Declarative class to do query in upgrade"""
    __tablename__ = 'dashboards'
    id = Column(Integer, primary_key=True)
    owners = relationship('User', secondary=dashboard_user)

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    objects = session.query(Slice).all()
    objects += session.query(Dashboard).all()
    for obj in objects:
        if obj.created_by and obj.created_by not in obj.owners:
            obj.owners.append(obj.created_by)
        session.commit()
    session.close()

def downgrade():
    if False:
        return 10
    pass