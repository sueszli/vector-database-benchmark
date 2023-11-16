from datetime import datetime
from typing import Optional
from .BaseModel import BaseModel
DIGITAL_FACTORY_RESPONSE_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S.%fZ'

class DigitalFactoryFileResponse(BaseModel):
    """Class representing a file in a digital factory project."""

    def __init__(self, client_id: str, content_type: str, file_id: str, file_name: str, library_project_id: str, status: str, user_id: str, username: str, uploaded_at: str, download_url: Optional[str]='', status_description: Optional[str]='', file_size: Optional[int]=0, upload_url: Optional[str]='', **kwargs) -> None:
        if False:
            return 10
        '\n        Creates a new DF file response object\n\n        :param client_id:\n        :param content_type:\n        :param file_id:\n        :param file_name:\n        :param library_project_id:\n        :param status:\n        :param user_id:\n        :param username:\n        :param download_url:\n        :param status_description:\n        :param file_size:\n        :param upload_url:\n        :param kwargs:\n        '
        self.client_id = client_id
        self.content_type = content_type
        self.download_url = download_url
        self.file_id = file_id
        self.file_name = file_name
        self.file_size = file_size
        self.library_project_id = library_project_id
        self.status = status
        self.status_description = status_description
        self.upload_url = upload_url
        self.user_id = user_id
        self.username = username
        self.uploaded_at = datetime.strptime(uploaded_at, DIGITAL_FACTORY_RESPONSE_DATETIME_FORMAT)
        super().__init__(**kwargs)

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'File: {}, from: {}, File ID: {}, Project ID: {}, Download URL: {}'.format(self.file_name, self.username, self.file_id, self.library_project_id, self.download_url)

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        super().validate()
        if not self.file_id:
            raise ValueError('file_id is required in Digital Library file')