"""
Contains Uploaders, a class to hold a S3Uploader and an ECRUploader
"""
from enum import Enum
from typing import Union
from samcli.lib.package.ecr_uploader import ECRUploader
from samcli.lib.package.s3_uploader import S3Uploader

class Destination(Enum):
    S3 = 's3'
    ECR = 'ecr'

class Uploaders:
    """
    Class to hold a S3Uploader and an ECRUploader
    """
    _s3_uploader: S3Uploader
    _ecr_uploader: ECRUploader

    def __init__(self, s3_uploader: S3Uploader, ecr_uploader: ECRUploader):
        if False:
            for i in range(10):
                print('nop')
        self._s3_uploader = s3_uploader
        self._ecr_uploader = ecr_uploader

    def get(self, destination: Destination) -> Union[S3Uploader, ECRUploader]:
        if False:
            while True:
                i = 10
        if destination == Destination.S3:
            return self._s3_uploader
        if destination == Destination.ECR:
            return self._ecr_uploader
        raise ValueError(f'destination has invalid value: {destination}')

    @property
    def s3(self):
        if False:
            for i in range(10):
                print('nop')
        return self._s3_uploader

    @property
    def ecr(self):
        if False:
            return 10
        return self._ecr_uploader