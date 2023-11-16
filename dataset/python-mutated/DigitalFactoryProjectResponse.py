from datetime import datetime
from typing import Optional, List, Dict, Any
from .BaseModel import BaseModel
from .DigitalFactoryFileResponse import DIGITAL_FACTORY_RESPONSE_DATETIME_FORMAT
from UM.i18n import i18nCatalog
catalog = i18nCatalog('cura')

class DigitalFactoryProjectResponse(BaseModel):
    """Class representing a cloud project."""

    def __init__(self, library_project_id: str, display_name: str, username: str=catalog.i18nc('@text Placeholder for the username if it has been deleted', 'deleted user'), organization_shared: bool=False, last_updated: Optional[str]=None, created_at: Optional[str]=None, thumbnail_url: Optional[str]=None, organization_id: Optional[str]=None, created_by_user_id: Optional[str]=None, description: Optional[str]='', tags: Optional[List[str]]=None, team_ids: Optional[List[str]]=None, status: Optional[str]=None, technical_requirements: Optional[Dict[str, Any]]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Creates a new digital factory project response object\n        :param library_project_id:\n        :param display_name:\n        :param username:\n        :param organization_shared:\n        :param thumbnail_url:\n        :param created_by_user_id:\n        :param description:\n        :param tags:\n        :param kwargs:\n        '
        self.library_project_id = library_project_id
        self.display_name = display_name
        self.description = description
        self.username = username
        self.organization_shared = organization_shared
        self.organization_id = organization_id
        self.created_by_user_id = created_by_user_id
        self.thumbnail_url = thumbnail_url
        self.tags = tags
        self.team_ids = team_ids
        self.created_at = datetime.strptime(created_at, DIGITAL_FACTORY_RESPONSE_DATETIME_FORMAT) if created_at else None
        self.last_updated = datetime.strptime(last_updated, DIGITAL_FACTORY_RESPONSE_DATETIME_FORMAT) if last_updated else None
        self.status = status
        self.technical_requirements = technical_requirements
        super().__init__(**kwargs)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return 'Project: {}, Id: {}, from: {}'.format(self.display_name, self.library_project_id, self.username)

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        super().validate()
        if not self.library_project_id:
            raise ValueError('library_project_id is required on cloud project')