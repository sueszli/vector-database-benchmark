import os
import lookup_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
ids = ['c-024dcv3mk']
languages = ['en']

def test_lookup(capsys):
    if False:
        i = 10
        return i + 15
    lookup_sample.lookup_sample(project_id=project_id, location=location, ids=ids, languages=languages)
    (out, _) = capsys.readouterr()
    assert 'Name: Google' in out
    assert 'Types' in out
    assert 'Cloud MID' in out