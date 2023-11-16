from datetime import datetime
from typing import Optional
from .BaseModel import BaseModel

class DFLibraryFileUploadResponse(BaseModel):
    """
    Model that represents the response received from the Digital Factory after requesting to upload a file in a Library project
    """

    def __init__(self, client_id: str, content_type: str, file_id: str, file_name: str, library_project_id: str, status: str, uploaded_at: str, user_id: str, username: str, download_url: Optional[str]=None, file_size: Optional[int]=None, status_description: Optional[str]=None, upload_url: Optional[str]=None, **kwargs) -> None:
        if False:
            return 10
        "\n        :param client_id: The ID of the OAuth2 client that uploaded this file\n        :param content_type: The content type of the Digital Library project file\n        :param file_id: The ID of the library project file\n        :param file_name: The name of the file\n        :param library_project_id: The ID of the library project, in which the file will be uploaded\n        :param status: The status of the Digital Library project file\n        :param uploaded_at: The time on which the file was uploaded\n        :param user_id: The ID of the user that uploaded this file\n        :param username: The user's unique username\n        :param download_url: A signed URL to download the resulting file. Only available when the job is finished\n        :param file_size: The size of the uploaded file (in bytes)\n        :param status_description: Contains more details about the status, e.g. the cause of failures\n        :param upload_url: The one-time use URL where the file must be uploaded to (only if status is uploading)\n        :param kwargs: Other keyword arguments that may be included in the response\n        "
        self.client_id = client_id
        self.content_type = content_type
        self.file_id = file_id
        self.file_name = file_name
        self.library_project_id = library_project_id
        self.status = status
        self.uploaded_at = self.parseDate(uploaded_at)
        self.user_id = user_id
        self.username = username
        self.download_url = download_url
        self.file_size = file_size
        self.status_description = status_description
        self.upload_url = upload_url
        super().__init__(**kwargs)