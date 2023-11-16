"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Comprehend to
run a topic modeling job. Topic modeling analyzes a set of documents and determines
common themes.
"""
from enum import Enum
import logging
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class JobInputFormat(Enum):
    per_file = 'ONE_DOC_PER_FILE'
    per_line = 'ONE_DOC_PER_LINE'

class ComprehendTopicModeler:
    """Encapsulates a Comprehend topic modeler."""

    def __init__(self, comprehend_client):
        if False:
            print('Hello World!')
        '\n        :param comprehend_client: A Boto3 Comprehend client.\n        '
        self.comprehend_client = comprehend_client

    def start_job(self, job_name, input_bucket, input_key, input_format, output_bucket, output_key, data_access_role_arn):
        if False:
            return 10
        '\n        Starts a topic modeling job. Input is read from the specified Amazon S3\n        input bucket and written to the specified output bucket. Output data is stored\n        in a tar archive compressed in gzip format. The job runs asynchronously, so you\n        can call `describe_topics_detection_job` to get job status until it\n        returns a status of SUCCEEDED.\n\n        :param job_name: The name of the job.\n        :param input_bucket: An Amazon S3 bucket that contains job input.\n        :param input_key: The prefix used to find input data in the input\n                             bucket. If multiple objects have the same prefix, all\n                             of them are used.\n        :param input_format: The format of the input data, either one document per\n                             file or one document per line.\n        :param output_bucket: The Amazon S3 bucket where output data is written.\n        :param output_key: The prefix prepended to the output data.\n        :param data_access_role_arn: The Amazon Resource Name (ARN) of a role that\n                                     grants Comprehend permission to read from the\n                                     input bucket and write to the output bucket.\n        :return: Information about the job, including the job ID.\n        '
        try:
            response = self.comprehend_client.start_topics_detection_job(JobName=job_name, DataAccessRoleArn=data_access_role_arn, InputDataConfig={'S3Uri': f's3://{input_bucket}/{input_key}', 'InputFormat': input_format.value}, OutputDataConfig={'S3Uri': f's3://{output_bucket}/{output_key}'})
            logger.info('Started topic modeling job %s.', response['JobId'])
        except ClientError:
            logger.exception("Couldn't start topic modeling job.")
            raise
        else:
            return response

    def describe_job(self, job_id):
        if False:
            print('Hello World!')
        '\n        Gets metadata about a topic modeling job.\n\n        :param job_id: The ID of the job to look up.\n        :return: Metadata about the job.\n        '
        try:
            response = self.comprehend_client.describe_topics_detection_job(JobId=job_id)
            job = response['TopicsDetectionJobProperties']
            logger.info('Got topic detection job %s.', job_id)
        except ClientError:
            logger.exception("Couldn't get topic detection job %s.", job_id)
            raise
        else:
            return job

    def list_jobs(self):
        if False:
            print('Hello World!')
        '\n        Lists topic modeling jobs for the current account.\n\n        :return: The list of jobs.\n        '
        try:
            response = self.comprehend_client.list_topics_detection_jobs()
            jobs = response['TopicsDetectionJobPropertiesList']
            logger.info('Got %s topic detection jobs.', len(jobs))
        except ClientError:
            logger.exception("Couldn't get topic detection jobs.")
            raise
        else:
            return jobs