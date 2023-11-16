"""Google Cloud Video Stitcher sample for getting the specified stitch detail
for a video on demand (VOD) session.
Example usage:
    python get_vod_stitch_detail.py --project_id <project-id>         --location <location> --session_id <session-id>         --stitch_details_id <stitch-details-id>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def get_vod_stitch_detail(project_id: str, location: str, session_id: str, stitch_detail_id: str) -> stitcher_v1.types.VodStitchDetail:
    if False:
        i = 10
        return i + 15
    'Gets the specified stitch detail for a VOD session.\n    Args:\n        project_id: The GCP project ID.\n        location: The location of the session.\n        session_id: The ID of the VOD session.\n        stitch_detail_id: The ID of the stitch details.\n\n    Returns:\n        The VOD stitch detail resource.\n    '
    client = VideoStitcherServiceClient()
    name = client.vod_stitch_detail_path(project_id, location, session_id, stitch_detail_id)
    response = client.get_vod_stitch_detail(name=name)
    print(f'VOD stitch detail: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location of the VOD session.', required=True)
    parser.add_argument('--session_id', help='The ID of the VOD session.', required=True)
    parser.add_argument('--stitch_detail_id', help='The ID of the stitch details.', required=True)
    args = parser.parse_args()
    get_vod_stitch_detail(args.project_id, args.location, args.session_id, args.stitch_detail_id)