from sqlalchemy import Column, Integer, String, Boolean
from superagi.models.base_model import DBBaseModel
from sqlalchemy import or_

class ApiKey(DBBaseModel):
    """

    Attributes:
        

    Methods:
    """
    __tablename__ = 'api_keys'
    id = Column(Integer, primary_key=True)
    org_id = Column(Integer)
    name = Column(String)
    key = Column(String)
    is_expired = Column(Boolean)

    @classmethod
    def get_by_org_id(cls, session, org_id: int):
        if False:
            i = 10
            return i + 15
        db_api_keys = session.query(ApiKey).filter(ApiKey.org_id == org_id, or_(ApiKey.is_expired == False, ApiKey.is_expired == None)).all()
        return db_api_keys

    @classmethod
    def get_by_id(cls, session, id: int):
        if False:
            for i in range(10):
                print('nop')
        db_api_key = session.query(ApiKey).filter(ApiKey.id == id, or_(ApiKey.is_expired == False, ApiKey.is_expired == None)).first()
        return db_api_key

    @classmethod
    def delete_by_id(cls, session, id: int):
        if False:
            i = 10
            return i + 15
        db_api_key = session.query(ApiKey).filter(ApiKey.id == id).first()
        db_api_key.is_expired = True
        session.commit()
        session.flush()

    @classmethod
    def update_api_key(cls, session, id: int, name: str):
        if False:
            print('Hello World!')
        db_api_key = session.query(ApiKey).filter(ApiKey.id == id).first()
        db_api_key.name = name
        session.commit()
        session.flush()