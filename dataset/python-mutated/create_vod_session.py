"""Google Cloud Video Stitcher sample for creating a video on demand (VOD)
session in which to insert ads.
Example usage:
    python create_vod_session.py --project_id <project-id>         --location <location> --source_uri <uri> --ad_tag_uri <uri>
"""
import argparse
from google.cloud.video import stitcher_v1
from google.cloud.video.stitcher_v1.services.video_stitcher_service import VideoStitcherServiceClient

def create_vod_session(project_id: str, location: str, source_uri: str, ad_tag_uri: str) -> stitcher_v1.types.VodSession:
    if False:
        return 10
    'Creates a VOD session. VOD sessions are ephemeral resources that expire\n    after a few hours.\n    Args:\n        project_id: The GCP project ID.\n        location: The location in which to create the session.\n        source_uri: Uri of the media to stitch; this URI must reference either an MPEG-DASH\n                    manifest (.mpd) file or an M3U playlist manifest (.m3u8) file.\n        ad_tag_uri: Uri of the ad tag.\n\n    Returns:\n        The VOD session resource.\n    '
    client = VideoStitcherServiceClient()
    parent = f'projects/{project_id}/locations/{location}'
    vod_session = stitcher_v1.types.VodSession(source_uri=source_uri, ad_tag_uri=ad_tag_uri, ad_tracking='SERVER')
    response = client.create_vod_session(parent=parent, vod_session=vod_session)
    print(f'VOD session: {response.name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', help='Your Cloud project ID.', required=True)
    parser.add_argument('--location', help='The location in which to create the VOD session.', default='us-central1')
    parser.add_argument('--source_uri', help='The Uri of the media to stitch (.mpd or .m3u8 file) in double quotes.', required=True)
    parser.add_argument('--ad_tag_uri', help='Uri of the ad tag in double quotes.', required=True)
    args = parser.parse_args()
    create_vod_session(args.project_id, args.location, args.source_uri, args.ad_tag_uri)