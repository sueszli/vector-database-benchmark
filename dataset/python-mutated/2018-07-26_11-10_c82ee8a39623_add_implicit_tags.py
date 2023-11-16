"""Add implicit tags

Revision ID: c82ee8a39623
Revises: c18bd4186f15
Create Date: 2018-07-26 11:10:23.653524

"""
revision = 'c82ee8a39623'
down_revision = 'c617da68de7d'
from datetime import datetime
from alembic import op
from flask_appbuilder.models.mixins import AuditMixin
from sqlalchemy import Column, DateTime, Enum, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from superset.tags.models import ObjectType, TagType
from superset.utils.core import get_user_id
Base = declarative_base()

class AuditMixinNullable(AuditMixin):
    """Altering the AuditMixin to use nullable fields

    Allows creating objects programmatically outside of CRUD
    """
    created_on = Column(DateTime, default=datetime.now, nullable=True)
    changed_on = Column(DateTime, default=datetime.now, onupdate=datetime.now, nullable=True)

    @declared_attr
    def created_by_fk(self) -> Column:
        if False:
            for i in range(10):
                print('nop')
        return Column(Integer, ForeignKey('ab_user.id'), default=get_user_id, nullable=True)

    @declared_attr
    def changed_by_fk(self) -> Column:
        if False:
            return 10
        return Column(Integer, ForeignKey('ab_user.id'), default=get_user_id, onupdate=get_user_id, nullable=True)

class Tag(Base, AuditMixinNullable):
    """A tag attached to an object (query, chart or dashboard)."""
    __tablename__ = 'tag'
    id = Column(Integer, primary_key=True)
    name = Column(String(250), unique=True)
    type = Column(Enum(TagType))

class TaggedObject(Base, AuditMixinNullable):
    __tablename__ = 'tagged_object'
    id = Column(Integer, primary_key=True)
    tag_id = Column(Integer, ForeignKey('tag.id'))
    object_id = Column(Integer)
    object_type = Column(Enum(ObjectType))

class User(Base):
    """Declarative class to do query in upgrade"""
    __tablename__ = 'ab_user'
    id = Column(Integer, primary_key=True)

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    Tag.__table__.create(bind)
    TaggedObject.__table__.create(bind)

def downgrade():
    if False:
        return 10
    op.drop_table('tagged_object')
    op.drop_table('tag')