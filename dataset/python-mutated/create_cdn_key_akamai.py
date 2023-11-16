"""Google Cloud Video Stitcher sample for creating an Akamai CDN key. A CDN key is used
to retrieve protected media.
Example usage:
    python create_cdn_key_akamai.py --project_id <project-id> --location <location> --cdn_key_id <cdn_key_id>         --hostname <hostname> --akamai_token_key <token-key>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def create_cdn_key_akamai(project_id: str, location: str, cdn_key_id: str, hostname: str, akamai_token_key: str) -> stitcher_v1.types.CdnKey:
    if False:
        print('Hello World!')
    'Creates an Akamai CDN key.\n    Args:\n        project_id: The GCP project ID.\n        location: The location in which to create the CDN key.\n        cdn_key_id: The user-defined CDN key ID.\n        hostname: The hostname to which this CDN key applies.\n        akamai_token_key: A base64-encoded string token key.\n\n    Returns:\n        The CDN key resource.\n    '
    client = VideoStitcherServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    cdn_key = stitcher_v1.types.CdnKey(name=cdn_key_id, hostname=hostname, akamai_cdn_key=stitcher_v1.types.AkamaiCdnKey(token_key=akamai_token_key))
    operation = client.create_cdn_key(parent=parent, cdn_key_id=cdn_key_id, cdn_key=cdn_key)
    response = operation.result()
    print(f'CDN key: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location in which to create the CDN key.', default='us-central1')
    parser.add_argument('--cdn_key_id', help='The user-defined CDN key ID.', required=True)
    parser.add_argument('--hostname', help='The hostname to which this CDN key applies.', required=True)
    parser.add_argument('--akamai_token_key', help='The base64-encoded string token key.', required=True)
    args = parser.parse_args()
    create_cdn_key_akamai(args.project_id, args.location, args.cdn_key_id, args.hostname, args.akamai_token_key)