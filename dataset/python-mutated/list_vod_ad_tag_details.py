"""Google Cloud Video Stitcher sample for listing the ad tag details for a video
on demand (VOD) session.
Example usage:
    python list_vod_ad_tag_details.py --project_id <project-id>         --location <location> --session_id <session-id>
"""
import argparse
from google.cloud.video.stitcher_v1.services.video_stitcher_service import pagers, VideoStitcherServiceClient

def list_vod_ad_tag_details(project_id: str, location: str, session_id: str) -> pagers.ListVodAdTagDetailsPager:
    if False:
        for i in range(10):
            print('nop')
    'Lists the ad tag details for the specified VOD session.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the session.\n        session_id: The ID of the VOD session.\n\n    Returns:\n        An iterable object containing VOD ad tag details resources.\n    '
    client = VideoStitcherServiceClient()
    parent = client.vod_session_path(project_id, location, session_id)
    page_result = client.list_vod_ad_tag_details(parent=parent)
    print('VOD ad tag details:')
    for response in page_result:
        print(response)
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the VOD session.', required=True)
    parser.add_argument('--session_id', help='The ID of the VOD session.', required=True)
    args = parser.parse_args()
    list_vod_ad_tag_details(args.project_id, args.location, args.session_id)