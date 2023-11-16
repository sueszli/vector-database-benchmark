import os
import search_public_kg_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
search_query = 'Google'
languages = ['en']
types = ['Organization']
limit = 1

def test_search_public_kg(capsys):
    if False:
        for i in range(10):
            print('nop')
    search_public_kg_sample.search_public_kg_sample(project_id=project_id, location=location, search_query=search_query, languages=languages, types=types, limit=limit)
    (out, _) = capsys.readouterr()
    assert 'Name: Google' in out
    assert 'Types' in out
    assert 'Cloud MID' in out