from pathlib import Path
from .routes import serialize, app
from . import models
from oso import Oso
from polar.data.adapter.sqlalchemy_adapter import SqlAlchemyAdapter
from sqlalchemy import Column, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
oso = Oso()
engine = create_engine('sqlite://')
Session = sessionmaker(bind=engine)
Base = declarative_base(bind=engine)

class Repository(Base):
    __tablename__ = 'repo'
    name = Column(String(128), primary_key=True)
    is_public = Column(Boolean)
Base.metadata.create_all()
oso.register_class(models.User)
oso.register_class(Repository, fields={'is_public': bool})
oso.set_data_filtering_adapter(SqlAlchemyAdapter(Session()))
oso.load_files([Path(__file__).parent / 'main.polar'])

class User:

    @staticmethod
    def get_current_user():
        if False:
            return 10
        return models.User(roles=[{'name': 'admin', 'repository': Repository(name='gmail')}])

@app.route('/repos')
def repo_list():
    if False:
        return 10
    repositories = oso.authorized_resources(User.get_current_user(), 'read', Repository)
    return serialize(repositories)