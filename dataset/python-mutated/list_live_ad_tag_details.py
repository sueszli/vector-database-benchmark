"""Google Cloud Video Stitcher sample for listing the ad tag details for a
live session.
Example usage:
    python list_live_ad_tag_details.py --project_id <project-id>         --location <location> --session_id <session-id>
"""
import argparse
from google.cloud.video.stitcher_v1.services.video_stitcher_service import pagers, VideoStitcherServiceClient

def list_live_ad_tag_details(project_id: str, location: str, session_id: str) -> pagers.ListLiveAdTagDetailsPager:
    if False:
        return 10
    'Lists the ad tag details for the specified live session.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the session.\n        session_id: The ID of the live session.\n\n    Returns:\n        An iterable object containing live ad tag details resources.\n    '
    client = VideoStitcherServiceClient()
    parent = client.live_session_path(project_id, location, session_id)
    page_result = client.list_live_ad_tag_details(parent=parent)
    print('Live ad tag details:')
    for response in page_result:
        print(response)
    return page_result
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the live session.', required=True)
    parser.add_argument('--session_id', help='The ID of the live session.', required=True)
    args = parser.parse_args()
    list_live_ad_tag_details(args.project_id, args.location, args.session_id)