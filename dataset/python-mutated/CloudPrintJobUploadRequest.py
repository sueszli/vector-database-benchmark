from ..BaseModel import BaseModel

class CloudPrintJobUploadRequest(BaseModel):

    def __init__(self, job_name: str, file_size: int, content_type: str, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Creates a new print job upload request.\n\n        :param job_name: The name of the print job.\n        :param file_size: The size of the file in bytes.\n        :param content_type: The content type of the print job (e.g. text/plain or application/gzip)\n        '
        self.job_name = job_name
        self.file_size = file_size
        self.content_type = content_type
        super().__init__(**kwargs)