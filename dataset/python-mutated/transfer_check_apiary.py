"""Command-line sample that list operations for a transfer job.

This sample is used on this page:

    https://cloud.google.com/storage/transfer/create-transfer

For more information, see README.md.
"""
import argparse
import json
import googleapiclient.discovery

def main(project_id, job_name):
    if False:
        return 10
    'Review the transfer operations associated with a transfer job.'
    storagetransfer = googleapiclient.discovery.build('storagetransfer', 'v1')
    filterString = '{{"project_id": "{project_id}", "job_names": ["{job_name}"]}}'.format(project_id=project_id, job_name=job_name)
    result = storagetransfer.transferOperations().list(name='transferOperations', filter=filterString).execute()
    print('Result of transferOperations/list: {}'.format(json.dumps(result, indent=4, sort_keys=True)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='Your Google Cloud project ID.')
    parser.add_argument('job_name', help='Your job name.')
    args = parser.parse_args()
    main(args.project_id, args.job_name)