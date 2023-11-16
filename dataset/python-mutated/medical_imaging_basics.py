"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) to manage and invoke AWS HealthImaging
functions.
"""
import datetime
import gzip
import json
import logging
import random
import time
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class MedicalImagingWrapper:

    def __init__(self, health_imaging_client):
        if False:
            while True:
                i = 10
        self.health_imaging_client = health_imaging_client

    def create_datastore(self, name):
        if False:
            return 10
        '\n        Create a data store.\n\n        :param name: The name of the data store to create.\n        :return: The data store ID.\n        '
        try:
            data_store = self.health_imaging_client.create_datastore(datastoreName=name)
        except ClientError as err:
            logger.error("Couldn't create data store %s. Here's why: %s: %s", name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return data_store['datastoreId']

    def get_datastore_properties(self, datastore_id):
        if False:
            return 10
        '\n        Get the properties of a data store.\n\n        :param datastore_id: The ID of the data store.\n        :return: The data store properties.\n        '
        try:
            data_store = self.health_imaging_client.get_datastore(datastoreId=datastore_id)
        except ClientError as err:
            logger.error("Couldn't get data store %s. Here's why: %s: %s", id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return data_store['datastoreProperties']

    def list_datastores(self):
        if False:
            return 10
        '\n        List the data stores.\n\n        :return: The list of data stores.\n        '
        try:
            paginator = self.health_imaging_client.get_paginator('list_datastores')
            page_iterator = paginator.paginate()
            datastore_summaries = []
            for page in page_iterator:
                datastore_summaries.extend(page['datastoreSummaries'])
        except ClientError as err:
            logger.error("Couldn't list data stores. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return datastore_summaries

    def delete_datastore(self, datastore_id):
        if False:
            print('Hello World!')
        '\n        Delete a data store.\n\n        :param datastore_id: The ID of the data store.\n        '
        try:
            self.health_imaging_client.delete_datastore(datastoreId=datastore_id)
        except ClientError as err:
            logger.error("Couldn't delete data store %s. Here's why: %s: %s", datastore_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def start_dicom_import_job(self, job_name, datastore_id, role_arn, input_s3_uri, output_s3_uri):
        if False:
            return 10
        '\n        Start a DICOM import job.\n\n        :param job_name: The name of the job.\n        :param datastore_id: The ID of the data store.\n        :param role_arn: The Amazon Resource Name (ARN) of the role to use for the job.\n        :param input_s3_uri: The S3 bucket input prefix path containing the DICOM files.\n        :param output_s3_uri: The S3 bucket output prefix path for the result.\n        :return: The job ID.\n        '
        try:
            job = self.health_imaging_client.start_dicom_import_job(jobName=job_name, datastoreId=datastore_id, dataAccessRoleArn=role_arn, inputS3Uri=input_s3_uri, outputS3Uri=output_s3_uri)
        except ClientError as err:
            logger.error("Couldn't start DICOM import job. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return job['jobId']

    def get_dicom_import_job(self, datastore_id, job_id):
        if False:
            while True:
                i = 10
        '\n        Get the properties of a DICOM import job.\n\n        :param datastore_id: The ID of the data store.\n        :param job_id: The ID of the job.\n        :return: The job properties.\n        '
        try:
            job = self.health_imaging_client.get_dicom_import_job(jobId=job_id, datastoreId=datastore_id)
        except ClientError as err:
            logger.error("Couldn't get DICOM import job. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return job['jobProperties']

    def list_dicom_import_jobs(self, datastore_id):
        if False:
            i = 10
            return i + 15
        '\n        List the DICOM import jobs.\n\n        :param datastore_id: The ID of the data store.\n        :return: The list of jobs.\n        '
        try:
            paginator = self.health_imaging_client.get_paginator('list_dicom_import_jobs')
            page_iterator = paginator.paginate(datastoreId=datastore_id)
            job_summaries = []
            for page in page_iterator:
                job_summaries.extend(page['jobSummaries'])
        except ClientError as err:
            logger.error("Couldn't list DICOM import jobs. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return job_summaries

    def search_image_sets(self, datastore_id, search_filter):
        if False:
            return 10
        '\n        Search for image sets.\n\n        :param datastore_id: The ID of the data store.\n        :param search_filter: The search filter.\n            For example: {"filters" : [{ "operator": "EQUAL", "values": [{"DICOMPatientId": "3524578"}]}]}.\n        :return: The list of image sets.\n        '
        try:
            paginator = self.health_imaging_client.get_paginator('search_image_sets')
            page_iterator = paginator.paginate(datastoreId=datastore_id, searchCriteria=search_filter)
            metadata_summaries = []
            for page in page_iterator:
                metadata_summaries.extend(page['imageSetsMetadataSummaries'])
        except ClientError as err:
            logger.error("Couldn't search image sets. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return metadata_summaries

    def get_image_set(self, datastore_id, image_set_id, version_id=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the properties of an image set.\n\n        :param datastore_id: The ID of the data store.\n        :param image_set_id: The ID of the image set.\n        :param version_id: The optional version of the image set.\n        :return: The image set properties.\n        '
        try:
            if version_id:
                image_set = self.health_imaging_client.get_image_set(imageSetId=image_set_id, datastoreId=datastore_id, versionId=version_id)
            else:
                image_set = self.health_imaging_client.get_image_set(imageSetId=image_set_id, datastoreId=datastore_id)
        except ClientError as err:
            logger.error("Couldn't get image set. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return image_set

    def get_image_set_metadata(self, metadata_file, datastore_id, image_set_id, version_id=None):
        if False:
            print('Hello World!')
        '\n        Get the metadata of an image set.\n\n        :param metadata_file: The file to store the JSON gzipped metadata.\n        :param datastore_id: The ID of the data store.\n        :param image_set_id: The ID of the image set.\n        :param version_id: The version of the image set.\n        '
        try:
            if version_id:
                image_set_metadata = self.health_imaging_client.get_image_set_metadata(imageSetId=image_set_id, datastoreId=datastore_id, versionId=version_id)
            else:
                image_set_metadata = self.health_imaging_client.get_image_set_metadata(imageSetId=image_set_id, datastoreId=datastore_id)
            print(image_set_metadata)
            with open(metadata_file, 'wb') as f:
                for chunk in image_set_metadata['imageSetMetadataBlob'].iter_chunks():
                    if chunk:
                        f.write(chunk)
        except ClientError as err:
            logger.error("Couldn't get image metadata. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def get_pixel_data(self, file_path_to_write, datastore_id, image_set_id, image_frame_id):
        if False:
            return 10
        "\n        Get an image frame's pixel data.\n\n        :param file_path_to_write: The path to write the image frame's HTJ2K encoded pixel data.\n        :param datastore_id: The ID of the data store.\n        :param image_set_id: The ID of the image set.\n        :param image_frame_id: The ID of the image frame.\n        "
        try:
            image_frame = self.health_imaging_client.get_image_frame(datastoreId=datastore_id, imageSetId=image_set_id, imageFrameInformation={'imageFrameId': image_frame_id})
            with open(file_path_to_write, 'wb') as f:
                for chunk in image_frame['imageFrameBlob'].iter_chunks():
                    if chunk:
                        f.write(chunk)
        except ClientError as err:
            logger.error("Couldn't get image frame. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def list_image_set_versions(self, datastore_id, image_set_id):
        if False:
            while True:
                i = 10
        '\n        List the image set versions.\n\n        :param datastore_id: The ID of the data store.\n        :param image_set_id: The ID of the image set.\n        :return: The list of image set versions.\n        '
        try:
            paginator = self.health_imaging_client.get_paginator('list_image_set_versions')
            page_iterator = paginator.paginate(imageSetId=image_set_id, datastoreId=datastore_id)
            image_set_properties_list = []
            for page in page_iterator:
                image_set_properties_list.extend(page['imageSetPropertiesList'])
        except ClientError as err:
            logger.error("Couldn't list image set versions. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return image_set_properties_list

    def update_image_set_metadata(self, datastore_id, image_set_id, version_id, metadata):
        if False:
            while True:
                i = 10
        '\n        Update the metadata of an image set.\n\n        :param datastore_id: The ID of the data store.\n        :param image_set_id: The ID of the image set.\n        :param version_id: The ID of the image set version.\n        :param metadata: The image set metadata as a dictionary.\n            For example {"DICOMUpdates": {"updatableAttributes":\n            "{"SchemaVersion":1.1,"Patient":{"DICOM":{"PatientName":"Garcia^Gloria"}}}"}}\n        :return: The updated image set metadata.\n        '
        try:
            updated_metadata = self.health_imaging_client.update_image_set_metadata(imageSetId=image_set_id, datastoreId=datastore_id, latestVersionId=version_id, updateImageSetMetadataUpdates=metadata)
        except ClientError as err:
            logger.error("Couldn't update image set metadata. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return updated_metadata

    def copy_image_set(self, datastore_id, image_set_id, version_id, destination_image_set_id=None, destination_version_id=None):
        if False:
            while True:
                i = 10
        '\n        Copy an image set.\n\n        :param datastore_id: The ID of the data store.\n        :param image_set_id: The ID of the image set.\n        :param version_id: The ID of the image set version.\n        :param destination_image_set_id: The ID of the optional destination image set.\n        :param destination_version_id: The ID of the optional destination image set version.\n        :return: The copied image set ID.\n        '
        try:
            copy_image_set_information = {'sourceImageSet': {'latestVersionId': version_id}}
            if destination_image_set_id and destination_version_id:
                copy_image_set_information['destinationImageSet'] = {'imageSetId': destination_image_set_id, 'latestVersionId': destination_version_id}
            copy_results = self.health_imaging_client.copy_image_set(datastoreId=datastore_id, sourceImageSetId=image_set_id, copyImageSetInformation=copy_image_set_information)
        except ClientError as err:
            logger.error("Couldn't copy image set. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return copy_results['destinationImageSetProperties']['imageSetId']

    def delete_image_set(self, datastore_id, image_set_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete an image set.\n\n        :param datastore_id: The ID of the data store.\n        :param image_set_id: The ID of the image set.\n        :return: The delete results.\n        '
        try:
            delete_results = self.health_imaging_client.delete_image_set(imageSetId=image_set_id, datastoreId=datastore_id)
        except ClientError as err:
            logger.error("Couldn't delete image set. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return delete_results

    def tag_resource(self, resource_arn, tags):
        if False:
            return 10
        '\n        Tag a resource.\n\n        :param resource_arn: The ARN of the resource.\n        :param tags: The tags to apply.\n        '
        try:
            self.health_imaging_client.tag_resource(resourceArn=resource_arn, tags=tags)
        except ClientError as err:
            logger.error("Couldn't tag resource. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def untag_resource(self, resource_arn, tag_keys):
        if False:
            while True:
                i = 10
        '\n        Untag a resource.\n\n        :param resource_arn: The ARN of the resource.\n        :param tag_keys: The tag keys to remove.\n        '
        try:
            self.health_imaging_client.untag_resource(resourceArn=resource_arn, tagKeys=tag_keys)
        except ClientError as err:
            logger.error("Couldn't untag resource. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def list_tags_for_resource(self, resource_arn):
        if False:
            print('Hello World!')
        '\n        List the tags for a resource.\n\n        :param resource_arn: The ARN of the resource.\n        :return: The list of tags.\n        '
        try:
            tags = self.health_imaging_client.list_tags_for_resource(resourceArn=resource_arn)
        except ClientError as err:
            logger.error("Couldn't list tags for resource. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return tags['tags']

    def usage_demo(self, source_s3_uri, dest_s3_uri, data_access_role_arn):
        if False:
            for i in range(10):
                print('nop')
        data_store_name = f'python_usage_demo_data_store_{random.randint(0, 200000)}'
        data_store_id = self.create_datastore(data_store_name)
        print(f'Data store created with id : {data_store_id}')
        while True:
            time.sleep(1)
            datastore_properties = self.get_datastore_properties(data_store_id)
            datastore_status = datastore_properties['datastoreStatus']
            print(f'data store status: "{datastore_status}"')
            if datastore_status == 'ACTIVE':
                break
            elif datastore_status == 'CREATE_FAILED':
                raise Exception('Create datastore job failed')
        datastores = self.list_datastores()
        print(f'datastores : {datastores}')
        job_name = 'python_usage_demo_job'
        job_id = self.start_dicom_import_job(job_name, data_store_id, data_access_role_arn, source_s3_uri, dest_s3_uri)
        print(f'Started import job with id: {job_id}')
        while True:
            time.sleep(1)
            job = self.get_dicom_import_job(data_store_id, job_id)
            job_status = job['jobStatus']
            print(f'Status of import job : "{job_status}"')
            if job_status == 'COMPLETED':
                break
            elif job_status == 'FAILED':
                raise Exception('DICOM import job failed')
        import_jobs = self.list_dicom_import_jobs(data_store_id)
        print(import_jobs)
        for job in import_jobs:
            print(job)
        filter = {'filters': [{'operator': 'EQUAL', 'values': [{'DICOMPatientId': '3524578'}]}]}
        image_sets = self.search_image_sets(data_store_id, filter)
        filter = {'filters': [{'operator': 'BETWEEN', 'values': [{'DICOMStudyDateAndTime': {'DICOMStudyDate': '19900101', 'DICOMStudyTime': '000000'}}, {'DICOMStudyDateAndTime': {'DICOMStudyDate': '20230101', 'DICOMStudyTime': '000000'}}]}]}
        image_sets = self.search_image_sets(data_store_id, filter)
        filter = {'filters': [{'values': [{'createdAt': datetime.datetime(2021, 8, 4, 14, 49, 54, 429000)}, {'createdAt': datetime.datetime.now() + datetime.timedelta(days=1)}], 'operator': 'BETWEEN'}]}
        image_sets = self.search_image_sets(data_store_id, filter)
        image_set_ids = [image_set['imageSetId'] for image_set in image_sets]
        for image_set in image_sets:
            print(image_set)
        image_set_id = image_sets[0]['imageSetId']
        version_id = image_sets[0]['version']
        returned_image_set = self.get_image_set(data_store_id, image_set_id, str(version_id))
        print(returned_image_set)
        image_metadata_file_name = 'metadata.json.gzip'
        self.get_image_set_metadata(image_metadata_file_name, data_store_id, image_set_id)
        image_frame_id = ''
        with gzip.open(image_metadata_file_name, 'rb') as f_in:
            data = json.load(f_in)
            series = data['Study']['Series']
            for value in series.values():
                for instance in value['Instances'].values():
                    image_frame_id = instance['ImageFrames'][0]['ID']
        if image_frame_id == '':
            raise Exception('Image frame id is empty')
        image_file_name = 'image_frame.jph'
        self.get_pixel_data(image_file_name, data_store_id, image_set_id, image_frame_id)
        returned_versions = self.list_image_set_versions(data_store_id, image_set_id)
        for version in returned_versions:
            print(version)
        copied_image_set_id = self.copy_image_set(data_store_id, image_set_id, str(version_id))
        print(f'Copied image set to new image set with ID : {copied_image_set_id}')
        image_set_ids.append(copied_image_set_id)
        while True:
            time.sleep(1)
            try:
                image_set_properties = self.get_image_set(data_store_id, copied_image_set_id)
            except ClientError as err:
                print(f"get_image_set raised an error {err.response['Error']['Message']}")
                break
            image_set_state = image_set_properties['imageSetState']
            print(f'Image set with id : "{copied_image_set_id}" has status: "{image_set_state}"')
            if image_set_state != 'LOCKED':
                break
        attributes = '{"SchemaVersion":1.1,"Patient":{"DICOM":{"PatientName":"Garcia^Gloria"}}}'
        metadata = {'DICOMUpdates': {'updatableAttributes': attributes}}
        self.update_image_set_metadata(data_store_id, copied_image_set_id, '1', metadata)
        print(f'Updated metadata for image set with id : {copied_image_set_id}')
        for image_set_id in image_set_ids:
            while True:
                time.sleep(1)
                try:
                    image_set_properties = self.get_image_set(data_store_id, image_set_id)
                except ClientError as err:
                    print(f"get_image_set raised an error {err.response['Error']['Message']}")
                    break
                image_set_state = image_set_properties['imageSetState']
                print(f'Image set with id : "{image_set_id}" has status: "{image_set_state}"')
                if image_set_state != 'LOCKED':
                    break
        for image_set_id in image_set_ids:
            self.delete_image_set(data_store_id, image_set_id)
            print(f'Deleted image set with id : {image_set_id}')
        for image_set_id in image_set_ids:
            while True:
                time.sleep(1)
                try:
                    image_set_properties = self.get_image_set(data_store_id, image_set_id)
                except ClientError as err:
                    print(f"get_image_set raised an error {err.response['Error']['Message']}")
                    break
                image_set_state = image_set_properties['imageSetState']
                print(f'Image set with id : "{image_set_id}" has status: "{image_set_state}"')
                if image_set_state == 'DELETED':
                    break
        self.delete_datastore(data_store_id)
        print(f'Data store deleted with id : {data_store_id}')
if __name__ == '__main__':
    source_s3_uri = 's3://medical-imaging-dicom-input/dicom_input/'
    dest_s3_uri = 's3://medical-imaging-output/job_output/'
    data_access_role_arn = 'arn:aws:iam::123456789012:role/ImportJobDataAccessRole'
    client = boto3.client('medical-imaging')
    medical_imaging_wrapper = MedicalImagingWrapper(client)
    medical_imaging_wrapper.usage_demo(source_s3_uri, dest_s3_uri, data_access_role_arn)