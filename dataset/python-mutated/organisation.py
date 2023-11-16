from sqlalchemy import Column, Integer, String
from superagi.helper.tool_helper import register_toolkits
from superagi.models.base_model import DBBaseModel

class Organisation(DBBaseModel):
    """
    Model representing an organization.

    Attributes:
        id (Integer): The primary key of the organization.
        name (String): The name of the organization.
        description (String): The description of the organization.
    """
    __tablename__ = 'organisations'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)

    def __repr__(self):
        if False:
            return 10
        '\n        Returns a string representation of the Organisation object.\n\n        Returns:\n            str: String representation of the Organisation object.\n        '
        return f"Organisation(id={self.id}, name='{self.name}')"

    @classmethod
    def find_or_create_organisation(cls, session, user):
        if False:
            print('Hello World!')
        '\n        Finds or creates an organization for the given user.\n\n        Args:\n            session: The database session.\n            user: The user object.\n\n        Returns:\n            Organisation: The found or created organization.\n        '
        if user.organisation_id is not None:
            organisation = session.query(Organisation).filter(Organisation.id == user.organisation_id).first()
            return organisation
        existing_organisation = session.query(Organisation).filter(Organisation.name == 'Default Organization - ' + str(user.id)).first()
        if existing_organisation is not None:
            user.organisation_id = existing_organisation.id
            session.commit()
            return existing_organisation
        new_organisation = Organisation(name='Default Organization - ' + str(user.id), description='New default organiztaion')
        session.add(new_organisation)
        session.commit()
        session.flush()
        user.organisation_id = new_organisation.id
        session.commit()
        register_toolkits(session=session, organisation=new_organisation)
        return new_organisation