import time
import boto3
from botocore.exceptions import ClientError
from SqsQueueNotificationWorker import SqsWorker, JobStatus
pipeline_id = 'PIPELINE_ID'
input_file = 'FILE_TO_TRANSCODE'
output_file = 'TRANSCODED_FILE'
sqs_queue_name = 'ets-sample-queue'
preset_id = '1351620000001-000020'
output_file_prefix = 'elastic-transcoder-samples/output/'
monitor_sqs_messages = True

class JobMonitor:
    """Monitors the SQS notifications received for an Elastic Transcoder job"""

    def __init__(self, job_id, sqs_queue_name):
        if False:
            while True:
                i = 10
        'Initialize new JobMonitor\n\n        :param job_id: string; Elastic Transcoder job ID to monitor\n        :param sqs_queue_name: string; Name of SQS queue subscribed to receive\n        notifications\n        '
        self._sqs_worker = SqsWorker(job_id, sqs_queue_name)
        self._job_id = job_id

    def start(self):
        if False:
            i = 10
            return i + 15
        'Have the SqsWorker start monitoring notifications'
        self._sqs_worker.start()

    def stop(self):
        if False:
            return 10
        'Instruct the SqsWorker to stop monitoring notifications\n\n        If this occurs before the job has finished, the monitoring of\n        notifications is aborted, but the Elastic Transcoder job itself\n        continues.\n        '
        self._sqs_worker.stop()

    def finished(self):
        if False:
            print('Hello World!')
        return self._sqs_worker.finished()

    def status(self):
        if False:
            print('Hello World!')
        return self._sqs_worker.job_status()

    def wait_for_completion(self):
        if False:
            while True:
                i = 10
        'Block until the job finishes'
        while not self.finished():
            time.sleep(5)
        self.stop()

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'JobMonitor(Job ID: {self._job_id}, Status: {self.status().name})'

def create_elastic_transcoder_job():
    if False:
        print('Hello World!')
    'Create an Elastic Transcoder job\n\n    All Elastic Transcoder set up operations must be completed before calling\n    this function, such as defining the pipeline and specifying the S3 input\n    and output buckets, etc.\n\n    :return Dictionary containing information about the job\n            JobComplete Waiter object\n            None if job could not be created\n    '
    etc_client = boto3.client('elastictranscoder')
    try:
        response = etc_client.create_job(PipelineId=pipeline_id, Input={'Key': input_file}, Outputs=[{'Key': output_file, 'PresetId': preset_id}], OutputKeyPrefix=output_file_prefix)
    except ClientError as e:
        print(f'ERROR: {e}')
        return None
    else:
        return (response['Job'], etc_client.get_waiter('job_complete'))

def main():
    if False:
        i = 10
        return i + 15
    (job_info, job_waiter) = create_elastic_transcoder_job()
    if job_info is None:
        exit(1)
    job_id = job_info['Id']
    print(f'Waiting for job {job_id} to complete...')
    if monitor_sqs_messages:
        job_monitor = JobMonitor(job_id, sqs_queue_name)
        job_monitor.start()
        job_monitor.wait_for_completion()
        status = job_monitor.status()
        if status == JobStatus.SUCCESS:
            print('Job completed successfully')
        elif status == JobStatus.ERROR:
            print('Job terminated with error')
        else:
            print(f'Job status: {status.name}')
    else:
        job_waiter.wait(Id=job_id, WaiterConfig={'Delay': 5})
        print('Job completed')
if __name__ == '__main__':
    main()