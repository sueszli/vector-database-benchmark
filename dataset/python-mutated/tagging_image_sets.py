"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) to tag AWS HealthImaging image sets.
"""
import boto3
from medical_imaging_basics import MedicalImagingWrapper

def tagging_image_sets(medical_imaging_wrapper, image_set_arn):
    if False:
        for i in range(10):
            print('nop')
    "\n    Taggging an image set.\n\n    :param medical_imaging_wrapper: A MedicalImagingWrapper instance.\n    :param image_set_arn: The Amazon Resource Name (ARN) of the image set.\n        For example: 'arn:aws:medical-imaging:us-east-1:123456789012:datastore/12345678901234567890123456789012/'                                 'imageset/12345678901234567890123456789012'\n    "
    medical_imaging_wrapper.tag_resource(image_set_arn, {'Deployment': 'Development'})
    medical_imaging_wrapper.list_tags_for_resource(image_set_arn)
    medical_imaging_wrapper.untag_resource(image_set_arn, ['Deployment'])
if __name__ == '__main__':
    an_image_set_arn = 'arn:aws:medical-imaging:us-east-1:123456789012:datastore/12345678901234567890123456789012/imageset/12345678901234567890123456789012'
    an_image_set_arn = input(f'Enter the ARN of the image set to tag: ')
    client = boto3.client('medical-imaging')
    a_medical_imaging_wrapper = MedicalImagingWrapper(client)
    tagging_image_sets(a_medical_imaging_wrapper, an_image_set_arn)