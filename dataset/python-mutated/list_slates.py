"""Google Cloud Video Stitcher sample for listing all slates in a location.
Example usage:
    python list_slates.py --project_id <project-id> --location <location>
"""
import argparse
from google.cloud.video.stitcher_v1.services.video_stitcher_service import pagers, VideoStitcherServiceClient

def list_slates(project_id: str, location: str) -> pagers.ListSlatesPager:
    if False:
        while True:
            i = 10
    'Lists all slates in a location.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the slates.\n\n    Returns:\n        An iterable object containing slate resources.\n    '
    client = VideoStitcherServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    response = client.list_slates(parent=parent)
    print('Slates:')
    for slate in response.slates:
        print({slate.name})
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the slates.', required=True)
    args = parser.parse_args()
    list_slates(args.project_id, args.location)