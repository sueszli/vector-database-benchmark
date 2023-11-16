import os
import job_search_autocomplete_job_title
PROJECT_ID = os.environ['GOOGLE_CLOUD_PROJECT']

def test_autocomplete_job_title(capsys, tenant):
    if False:
        i = 10
        return i + 15
    job_search_autocomplete_job_title.complete_query(PROJECT_ID, tenant, 'Software')
    (out, _) = capsys.readouterr()
    assert 'Suggested title:' in out