"""Google Cloud Video Stitcher sample for listing all CDN keys in a location.
Example usage:
    python list_cdn_keys.py --project_id <project-id> --location <location>
"""
import argparse
from google.cloud.video.stitcher_v1.services.video_stitcher_service import pagers, VideoStitcherServiceClient

def list_cdn_keys(project_id: str, location: str) -> pagers.ListCdnKeysPager:
    if False:
        while True:
            i = 10
    'Lists all CDN keys in a location.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the CDN keys.\n\n    Returns:\n        An iterable object containing CDN key resources.\n    '
    client = VideoStitcherServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    response = client.list_cdn_keys(parent=parent)
    print('CDN keys:')
    for cdn_key in response.cdn_keys:
        print({cdn_key.name})
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the CDN keys.', required=True)
    args = parser.parse_args()
    list_cdn_keys(args.project_id, args.location)