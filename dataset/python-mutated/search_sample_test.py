import os
import search_sample
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'global'
search_query = 'Google'
languages = ['en']
types = ['Organization']
limit = 1

def test_search(capsys):
    if False:
        i = 10
        return i + 15
    search_sample.search_sample(project_id=project_id, location=location, search_query=search_query, languages=languages, types=types, limit=limit)
    (out, _) = capsys.readouterr()
    assert 'Name: Google' in out
    assert 'Types' in out
    assert 'Cloud MID' in out