"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) to tag AWS HealthImaging data stores.
"""
import boto3
from medical_imaging_basics import MedicalImagingWrapper

def tagging_data_stores(medical_imaging_wrapper, data_store_arn):
    if False:
        print('Hello World!')
    '\n    Taggging a data store.\n\n    :param medical_imaging_wrapper: A MedicalImagingWrapper instance.\n    :param data_store_arn: The Amazon Resource Name (ARN) of the data store.\n        For example: arn:aws:medical-imaging:us-east-1:123456789012:datastore/12345678901234567890123456789012\n    '
    medical_imaging_wrapper.tag_resource(data_store_arn, {'Deployment': 'Development'})
    medical_imaging_wrapper.list_tags_for_resource(data_store_arn)
    medical_imaging_wrapper.untag_resource(data_store_arn, ['Deployment'])
if __name__ == '__main__':
    a_data_store_arn = 'arn:aws:medical-imaging:us-east-1:123456789012:datastore/12345678901234567890123456789012'
    a_data_store_arn = input(f'Enter the ARN of the data store to tag: ')
    client = boto3.client('medical-imaging')
    a_medical_imaging_wrapper = MedicalImagingWrapper(client)
    tagging_data_stores(a_medical_imaging_wrapper, a_data_store_arn)