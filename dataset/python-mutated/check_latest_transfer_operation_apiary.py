"""Command-line sample that checks the latest operation of a transfer.
This sample is used on this page:
    https://cloud.google.com/storage/transfer/create-transfer
For more information, see README.md.
"""
import argparse
import json
import googleapiclient.discovery

def check_latest_transfer_operation(project_id, job_name):
    if False:
        for i in range(10):
            print('nop')
    'Check the latest transfer operation associated with a transfer job.'
    storagetransfer = googleapiclient.discovery.build('storagetransfer', 'v1')
    transferJob = storagetransfer.transferJobs().get(projectId=project_id, jobName=job_name).execute()
    latestOperationName = transferJob.get('latestOperationName')
    if latestOperationName:
        result = storagetransfer.transferOperations().get(name=latestOperationName).execute()
        print('The latest operation for job' + job_name + ' is: {}'.format(json.dumps(result, indent=4, sort_keys=True)))
    else:
        print('Transfer job ' + job_name + ' does not have an operation scheduled yet, ' + 'try again once the job starts running.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID.')
    parser.add_argument('job_name', help='Your job name.')
    args = parser.parse_args()
    check_latest_transfer_operation(args.project_id, args.job_name)