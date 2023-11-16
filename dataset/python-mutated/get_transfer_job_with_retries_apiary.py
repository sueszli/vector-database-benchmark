"""Command-line sample that gets a transfer job using retries
"""
import googleapiclient.discovery

def get_transfer_job_with_retries(project_id, job_name, retries):
    if False:
        for i in range(10):
            print('nop')
    'Check the latest transfer operation associated with a transfer job.'
    storagetransfer = googleapiclient.discovery.build('storagetransfer', 'v1')
    transferJob = storagetransfer.transferJobs().get(projectId=project_id, jobName=job_name).execute(num_retries=retries)
    print('Fetched transfer job: ' + transferJob.get('name') + ' using {} retries'.format(retries))