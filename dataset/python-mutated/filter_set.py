from __future__ import annotations
import json
import logging
from typing import Any
from flask_appbuilder import Model
from sqlalchemy import Column, ForeignKey, Integer, MetaData, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy_utils import generic_relationship
from superset import app, db
from superset.models.helpers import AuditMixinNullable
metadata = Model.metadata
config = app.config
logger = logging.getLogger(__name__)

class FilterSet(Model, AuditMixinNullable):
    __tablename__ = 'filter_sets'
    id = Column(Integer, primary_key=True)
    name = Column(String(500), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    json_metadata = Column(Text, nullable=False)
    dashboard_id = Column(Integer, ForeignKey('dashboards.id'))
    dashboard = relationship('Dashboard', back_populates='_filter_sets')
    owner_id = Column(Integer, nullable=False)
    owner_type = Column(String(255), nullable=False)
    owner_object = generic_relationship(owner_type, owner_id)

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'FilterSet<{self.name or self.id}>'

    @property
    def url(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'/api/filtersets/{self.id}/'

    @property
    def sqla_metadata(self) -> None:
        if False:
            i = 10
            return i + 15
        with self.get_sqla_engine_with_context() as engine:
            meta = MetaData(bind=engine)
            meta.reflect()

    @property
    def changed_by_name(self) -> str:
        if False:
            while True:
                i = 10
        if not self.changed_by:
            return ''
        return str(self.changed_by)

    def to_dict(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        return {'id': self.id, 'name': self.name, 'description': self.description, 'params': self.params, 'dashboard_id': self.dashboard_id, 'owner_type': self.owner_type, 'owner_id': self.owner_id}

    @classmethod
    def get(cls, _id: int) -> FilterSet:
        if False:
            for i in range(10):
                print('nop')
        session = db.session()
        qry = session.query(FilterSet).filter(_id)
        return qry.one_or_none()

    @classmethod
    def get_by_name(cls, name: str) -> FilterSet:
        if False:
            return 10
        session = db.session()
        qry = session.query(FilterSet).filter(FilterSet.name == name)
        return qry.one_or_none()

    @classmethod
    def get_by_dashboard_id(cls, dashboard_id: int) -> FilterSet:
        if False:
            return 10
        session = db.session()
        qry = session.query(FilterSet).filter(FilterSet.dashboard_id == dashboard_id)
        return qry.all()

    @property
    def params(self) -> dict[str, Any]:
        if False:
            while True:
                i = 10
        if self.json_metadata:
            return json.loads(self.json_metadata)
        return {}