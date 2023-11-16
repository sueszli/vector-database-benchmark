"""Command-line sample that creates a daily transfer from a standard
GCS bucket to a Nearline GCS bucket for objects untouched for 30 days.

This sample is used on this page:

    https://cloud.google.com/storage/transfer/create-transfer

For more information, see README.md.
"""
import argparse
import datetime
import json
import googleapiclient.discovery

def main(description, project_id, start_date, start_time, source_bucket, sink_bucket):
    if False:
        for i in range(10):
            print('nop')
    'Create a daily transfer from Standard to Nearline Storage class.'
    storagetransfer = googleapiclient.discovery.build('storagetransfer', 'v1')
    transfer_job = {'description': description, 'status': 'ENABLED', 'projectId': project_id, 'schedule': {'scheduleStartDate': {'day': start_date.day, 'month': start_date.month, 'year': start_date.year}, 'startTimeOfDay': {'hours': start_time.hour, 'minutes': start_time.minute, 'seconds': start_time.second}}, 'transferSpec': {'gcsDataSource': {'bucketName': source_bucket}, 'gcsDataSink': {'bucketName': sink_bucket}, 'objectConditions': {'minTimeElapsedSinceLastModification': '2592000s'}, 'transferOptions': {'deleteObjectsFromSourceAfterTransfer': 'true'}}}
    result = storagetransfer.transferJobs().create(body=transfer_job).execute()
    print('Returned transferJob: {}'.format(json.dumps(result, indent=4)))
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('description', help='Transfer description.')
    parser.add_argument('project_id', help='Your Google Cloud project ID.')
    parser.add_argument('start_date', help='Date YYYY/MM/DD.')
    parser.add_argument('start_time', help='UTC Time (24hr) HH:MM:SS.')
    parser.add_argument('source_bucket', help='Standard GCS bucket name.')
    parser.add_argument('sink_bucket', help='Nearline GCS bucket name.')
    args = parser.parse_args()
    start_date = datetime.datetime.strptime(args.start_date, '%Y/%m/%d')
    start_time = datetime.datetime.strptime(args.start_time, '%H:%M:%S')
    main(args.description, args.project_id, start_date, start_time, args.source_bucket, args.sink_bucket)