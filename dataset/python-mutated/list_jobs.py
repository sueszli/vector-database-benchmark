"""Google Cloud Transcoder sample for listing jobs in a location.

Example usage:
    python list_jobs.py --project_id <project-id> --location <location>
"""
import argparse
from google.cloud.video.transcoder_v1.services.transcoder_service import pagers, TranscoderServiceClient

def list_jobs(project_id: str, location: str) -> pagers.ListJobsPager:
    if False:
        while True:
            i = 10
    'Lists all jobs in a location.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the jobs.\n\n    Returns:\n        An iterable object containing job resources.\n    '
    client = TranscoderServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    response = client.list_jobs(parent=parent)
    print('Jobs:')
    for job in response.jobs:
        print({job.name})
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the jobs.', required=True)
    args = parser.parse_args()
    list_jobs(args.project_id, args.location)