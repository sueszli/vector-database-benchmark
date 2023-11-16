"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon Comprehend to run a
topic modeling job on sample data. After the job completes, the output is retrieved
from Amazon S3 and extracted from its compressed format.
"""
import logging
from pprint import pprint
import sys
import boto3
from comprehend_demo_resources import ComprehendDemoResources
from comprehend_topic_modeler import ComprehendTopicModeler, JobInputFormat
sys.path.append('../..')
from demo_tools.custom_waiter import CustomWaiter, WaitState
logger = logging.getLogger(__name__)

class JobCompleteWaiter(CustomWaiter):
    """Waits for a job to complete."""

    def __init__(self, client):
        if False:
            i = 10
            return i + 15
        super().__init__('JobComplete', 'DescribeTopicsDetectionJob', 'TopicsDetectionJobProperties.JobStatus', {'COMPLETED': WaitState.SUCCESS, 'FAILED': WaitState.FAILURE}, client, delay=60)

    def wait(self, job_id):
        if False:
            for i in range(10):
                print('nop')
        self._wait(JobId=job_id)

def usage_demo():
    if False:
        print('Hello World!')
    print('-' * 88)
    print('Welcome to the Amazon Comprehend topic modeling demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    input_prefix = 'input/'
    output_prefix = 'output/'
    demo_resources = ComprehendDemoResources(boto3.resource('s3'), boto3.resource('iam'))
    topic_modeler = ComprehendTopicModeler(boto3.client('comprehend'))
    print('Setting up storage and security resources needed for the demo.')
    demo_resources.setup('comprehend-topic-modeler-demo')
    print('Copying sample data from public bucket into input bucket.')
    demo_resources.bucket.copy({'Bucket': 'public-sample-us-west-2', 'Key': 'TopicModeling/Sample.txt'}, f'{input_prefix}sample.txt')
    print('Starting topic modeling job on sample data.')
    job_info = topic_modeler.start_job('demo-topic-modeling-job', demo_resources.bucket.name, input_prefix, JobInputFormat.per_line, demo_resources.bucket.name, output_prefix, demo_resources.data_access_role.arn)
    print(f"Waiting for job {job_info['JobId']} to complete. This typically takes 20 - 30 minutes.")
    job_waiter = JobCompleteWaiter(topic_modeler.comprehend_client)
    job_waiter.wait(job_info['JobId'])
    job = topic_modeler.describe_job(job_info['JobId'])
    print(f"Job {job['JobId']} complete:")
    pprint(job)
    print(f"Getting job output data from the output Amazon S3 bucket: {job['OutputDataConfig']['S3Uri']}.")
    job_output = demo_resources.extract_job_output(job)
    lines = 10
    print(f'First {lines} lines of document topics output:')
    pprint(job_output['doc-topics.csv']['data'][:lines])
    print(f'First {lines} lines of terms output:')
    pprint(job_output['topic-terms.csv']['data'][:lines])
    print('Cleaning up resources created for the demo.')
    demo_resources.cleanup()
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()