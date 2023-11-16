"""Google Cloud Video Stitcher sample for getting a CDN key.
Example usage:
    python get_cdn_key.py --project_id <project_id> --location <location>         --cdn_key_id <cdn_key_id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def get_cdn_key(project_id: str, location: str, cdn_key_id: str) -> stitcher_v1.types.CdnKey:
    if False:
        for i in range(10):
            print('nop')
    'Gets a CDN key.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the CDN key.\n        cdn_key_id: The user-defined CDN key ID.\n\n    Returns:\n        The CDN key resource.\n    '
    client = VideoStitcherServiceClient()
    name = f'projects/{project_id}/locations/{location}/cdnKeys/{cdn_key_id}'
    response = client.get_cdn_key(name=name)
    print(f'CDN key: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the CDN key.', required=True)
    parser.add_argument('--cdn_key_id', help='The user-defined CDN key ID.', required=True)
    args = parser.parse_args()
    get_cdn_key(args.project_id, args.location, args.cdn_key_id)