from .BaseModel import BaseModel

class DFLibraryFileUploadRequest(BaseModel):

    def __init__(self, content_type: str, file_name: str, file_size: int, library_project_id: str, **kwargs) -> None:
        if False:
            print('Hello World!')
        self.content_type = content_type
        self.file_name = file_name
        self.file_size = file_size
        self.library_project_id = library_project_id
        super().__init__(**kwargs)