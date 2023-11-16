from .BaseModel import BaseModel
from typing import Optional

class DigitalFactoryFeatureBudgetResponse(BaseModel):
    """Class representing the capabilities of a user account for Digital Library.
       NOTE: For each max_..._projects fields, '-1' means unlimited!
    """

    def __init__(self, library_can_use_business_value: Optional[bool]=False, library_can_use_comments: Optional[bool]=False, library_can_use_status: Optional[bool]=False, library_can_use_tags: Optional[bool]=False, library_can_use_technical_requirements: Optional[bool]=False, library_max_organization_shared_projects: Optional[int]=None, library_max_private_projects: Optional[int]=None, library_max_team_shared_projects: Optional[int]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        self.library_can_use_business_value = library_can_use_business_value
        self.library_can_use_comments = library_can_use_comments
        self.library_can_use_status = library_can_use_status
        self.library_can_use_tags = library_can_use_tags
        self.library_can_use_technical_requirements = library_can_use_technical_requirements
        self.library_max_organization_shared_projects = library_max_organization_shared_projects
        self.library_max_private_projects = library_max_private_projects
        self.library_max_team_shared_projects = library_max_team_shared_projects
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'max private: {}, max org: {}, max team: {}'.format(self.library_max_private_projects, self.library_max_organization_shared_projects, self.library_max_team_shared_projects)

    def validate(self) -> None:
        if False:
            return 10
        super().validate()