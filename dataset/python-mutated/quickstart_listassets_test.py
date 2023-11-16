import os
import quickstart_listassets
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_list_assets(capsys):
    if False:
        i = 10
        return i + 15
    from google.cloud import asset_v1
    quickstart_listassets.list_assets(project_id=PROJECT, asset_types=['iam.googleapis.com/Role'], page_size=10, content_type=asset_v1.ContentType.RESOURCE)
    (out, _) = capsys.readouterr()
    assert 'asset' in out