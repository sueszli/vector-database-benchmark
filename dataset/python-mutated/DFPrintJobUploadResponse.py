from typing import Optional
from .BaseModel import BaseModel

class DFPrintJobUploadResponse(BaseModel):

    def __init__(self, job_id: str, status: str, download_url: Optional[str]=None, job_name: Optional[str]=None, upload_url: Optional[str]=None, content_type: Optional[str]=None, status_description: Optional[str]=None, slicing_details: Optional[dict]=None, **kwargs) -> None:
        if False:
            return 10
        "Creates a new print job response model.\n\n        :param job_id: The job unique ID, e.g. 'kBEeZWEifXbrXviO8mRYLx45P8k5lHVGs43XKvRniPg='.\n        :param status: The status of the print job.\n        :param status_description: Contains more details about the status, e.g. the cause of failures.\n        :param download_url: A signed URL to download the resulting status. Only available when the job is finished.\n        :param job_name: The name of the print job.\n        :param slicing_details: Model for slice information.\n        :param upload_url: The one-time use URL where the toolpath must be uploaded to (only if status is uploading).\n        :param content_type: The content type of the print job (e.g. text/plain or application/gzip)\n        :param generated_time: The datetime when the object was generated on the server-side.\n        "
        self.job_id = job_id
        self.status = status
        self.download_url = download_url
        self.job_name = job_name
        self.upload_url = upload_url
        self.content_type = content_type
        self.status_description = status_description
        self.slicing_details = slicing_details
        super().__init__(**kwargs)