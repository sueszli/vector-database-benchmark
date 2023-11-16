import os
import lookup_public_kg_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
ids = ['c-024dcv3mk']
languages = ['en']

def test_lookup_public_kg(capsys):
    if False:
        i = 10
        return i + 15
    lookup_public_kg_sample.lookup_public_kg_sample(project_id=project_id, location=location, ids=ids, languages=languages)
    (out, _) = capsys.readouterr()
    assert 'Name: Google' in out
    assert 'Types' in out
    assert 'Cloud MID' in out