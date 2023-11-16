import google.auth
from list_locations import list_locations
PROJECT = google.auth.default()[1]

def test_locations_list():
    if False:
        return 10
    locations = list_locations(PROJECT)
    assert 'asia-northeast1' in locations