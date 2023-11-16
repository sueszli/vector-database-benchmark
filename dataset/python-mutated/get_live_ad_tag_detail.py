"""Google Cloud Video Stitcher sample for getting the specified ad tag detail
for a live stream session.
Example usage:
    python get_live_ad_tag_detail.py --project_id <project-id>         --location <location> --session_id <session-id>         --ad_tag_details_id <ad-tag-details-id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def get_live_ad_tag_detail(project_id: str, location: str, session_id: str, ad_tag_detail_id: str) -> stitcher_v1.types.LiveAdTagDetail:
    if False:
        print('Hello World!')
    'Gets the specified ad tag detail for a live session.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the session.\n        session_id: The ID of the live session.\n        ad_tag_detail_id: The ID of the ad tag details.\n\n    Returns:\n        The live ad tag detail resource.\n    '
    client = VideoStitcherServiceClient()
    name = client.live_ad_tag_detail_path(project_id, location, session_id, ad_tag_detail_id)
    response = client.get_live_ad_tag_detail(name=name)
    print(f'Live ad tag detail: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the live session.', required=True)
    parser.add_argument('--session_id', help='The ID of the live session.', required=True)
    parser.add_argument('--ad_tag_detail_id', help='The ID of the ad tag details.', required=True)
    args = parser.parse_args()
    get_live_ad_tag_detail(args.project_id, args.location, args.session_id, args.ad_tag_detail_id)