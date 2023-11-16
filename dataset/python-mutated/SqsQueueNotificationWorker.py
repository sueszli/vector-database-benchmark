from ctypes import c_bool
from enum import Enum, auto
import json
import multiprocessing
import pprint
import boto3

class JobStatus(Enum):
    """Status of an Elastic Transcoder job"""
    SUCCESS = auto()
    ERROR = auto()
    RUNNING = auto()
    UNKNOWN = auto()

class ProcessStatus(Enum):
    """Status of an SqsWorker process"""
    READY = auto()
    IN_PROGRESS = auto()
    ABORTED = auto()
    FINISHED = auto()

class SqsWorker:
    """Monitors SQS notifications for an Elastic Transcoder job

    Each Elastic Transcoder job/JobMonitor must have its own SqsWorker
    object. The SqsWorker handles messages for the job. Messages for other
    jobs are ignored.

    The SysWorker performs its task in a separate process. The JobMonitor
    starts the process by calling SysWorker.start().

    While the SysWorker process is handling job notifications, the JobMonitor
    parent process can perform other tasks, including starting new jobs with
    new JobMonitor and SqsWorker objects.

    When the Transcoder job is finished, a SysWorker flag is set. The
    JobMonitor parent process must periodically retrieve the current setting
    of the flag by calling SysWorker.finished().

    When the Transcoder job has finished, the JobMonitor parent process must
    terminate the SysWorker process by calling SysWorker.stop().

    The final result of the completed job can be retrieved by calling
    SysWorker.job_status().
    """

    def __init__(self, job_id, sqs_queue_name):
        if False:
            while True:
                i = 10
        'Initialize an SqsWorker object to process SQS notification\n        messages for a particular Elastic Transcoder job.\n\n        :param job_id: string; Elastic Transcoder job ID to monitor\n        :param sqs_queue_name: string; Name of SQS queue subscribed to receive\n        notifications for job_id\n        '
        self._job_id = job_id
        self._finished = multiprocessing.Value(c_bool, False)
        self._job_status = multiprocessing.Value('i', JobStatus.RUNNING.value)
        self._process_status = multiprocessing.Value('i', ProcessStatus.READY.value)
        self._args = (job_id, sqs_queue_name, self._finished, self._job_status, self._process_status)
        self._process = None

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        "Start a new SqsWorker process to handle the job's notifications"
        if self._process is not None:
            raise RuntimeError('SqsQueueNotificationWorker already running.')
        self._process = multiprocessing.Process(target=poll_and_handle_messages, args=self._args)
        self._process.start()
        self._process_status.value = ProcessStatus.IN_PROGRESS.value

    def stop(self):
        if False:
            i = 10
            return i + 15
        'Stop the SqsWorker process'
        if self._process is None:
            raise RuntimeError('SqsQueueNotificationWorker already stopped.')
        if self._process.is_alive():
            self._process_status.value = ProcessStatus.ABORTED.value
            self._job_status.value = JobStatus.UNKNOWN.value
        self._finished.value = True
        self._process.join()

    def finished(self):
        if False:
            return 10
        'Finished = Job completed successfully or job terminated with error\n        or monitoring of notifications was aborted before receiving a\n        job-completed notification\n        '
        return self._finished.value

    def job_status(self):
        if False:
            i = 10
            return i + 15
        return JobStatus(self._job_status.value)

    def process_status(self):
        if False:
            print('Hello World!')
        return ProcessStatus(self._process_status.value)

    def __repr__(self):
        if False:
            return 10
        return f'SqsWorker(Job ID: {self._job_id}, Status: {ProcessStatus(self._process_status.value).name})'

def poll_and_handle_messages(job_id, sqs_queue_name, finished, job_status, process_status):
    if False:
        print('Hello World!')
    'Process SQS notifications for a particular Elastic Transcoder job\n\n    This method runs as a separate process.\n\n    :param job_id: string; Elastic Transcoder job ID to monitor\n    :param sqs_queue_name: string; Name of SQS queue\n    :param finished: boolean; Shared memory flag. While this method is running,\n    the flag might be set externally if the JobMonitor parent process instructs\n    us to stop before we receive notification that the job has finished.\n    Otherwise, this method sets the finished flag when the Transcoder job\n    finishes.\n    :param job_status: int/JobStatus enum; Shared memory variable containing\n    the Transcoder job status\n    :param process_status: int/ProcessStatus enum; Shared memory variable\n    containing the SysWorker process status\n    '
    sqs_client = boto3.client('sqs')
    response = sqs_client.get_queue_url(QueueName=sqs_queue_name)
    sqs_queue_url = response['QueueUrl']
    while not finished.value:
        response = sqs_client.receive_message(QueueUrl=sqs_queue_url, MaxNumberOfMessages=5, WaitTimeSeconds=5)
        if 'Messages' not in response:
            continue
        for message in response['Messages']:
            notification = json.loads(json.loads(message['Body'])['Message'])
            print('Notification:')
            pprint.pprint(notification)
            if notification['jobId'] == job_id:
                sqs_client.delete_message(QueueUrl=sqs_queue_url, ReceiptHandle=message['ReceiptHandle'])
                if notification['state'] == 'COMPLETED':
                    job_status.value = JobStatus.SUCCESS.value
                    process_status.value = ProcessStatus.FINISHED.value
                    finished.value = True
                elif notification['state'] == 'ERROR':
                    job_status.value = JobStatus.ERROR.value
                    process_status.value = ProcessStatus.FINISHED.value
                    finished.value = True