from __future__ import annotations
import requests
from sqlalchemy import Column, Integer, String
from superagi.models.base_model import DBBaseModel
marketplace_url = 'https://app.superagi.com/api'

class Vectordbs(DBBaseModel):
    """
    Represents an vector db entity.
    Attributes:
        id (int): The unique identifier of the agent.
        name (str): The name of the database.
        db_type (str): The name of the db agent.
        organisation_id (int): The identifier of the associated organisation.
    """
    __tablename__ = 'vector_dbs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    db_type = Column(String)
    organisation_id = Column(Integer)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a string representation of the Vector db object.\n        Returns:\n            str: String representation of the Vector db.\n        '
        return f"Vector(id={self.id}, name='{self.name}', db_type='{self.db_type}' organisation_id={self.organisation_id}, updated_at={self.updated_at})"

    @classmethod
    def get_vector_db_from_id(cls, session, vector_db_id):
        if False:
            for i in range(10):
                print('nop')
        vector_db = session.query(Vectordbs).filter(Vectordbs.id == vector_db_id).first()
        return vector_db

    @classmethod
    def fetch_marketplace_list(cls):
        if False:
            print('Hello World!')
        headers = {'Content-Type': 'application/json'}
        response = requests.get(marketplace_url + f'/vector_dbs/marketplace/list', headers=headers, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return []

    @classmethod
    def get_vector_db_from_organisation(cls, session, organisation):
        if False:
            i = 10
            return i + 15
        vector_db_list = session.query(Vectordbs).filter(Vectordbs.organisation_id == organisation.id).all()
        return vector_db_list

    @classmethod
    def add_vector_db(cls, session, name, db_type, organisation):
        if False:
            print('Hello World!')
        vector_db = Vectordbs(name=name, db_type=db_type, organisation_id=organisation.id)
        session.add(vector_db)
        session.commit()
        return vector_db

    @classmethod
    def delete_vector_db(cls, session, vector_db_id):
        if False:
            print('Hello World!')
        session.query(Vectordbs).filter(Vectordbs.id == vector_db_id).delete()
        session.commit()