import os
import quickstart_searchalliampolicies
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_search_all_iam_policies(capsys):
    if False:
        for i in range(10):
            print('nop')
    scope = f'projects/{PROJECT}'
    query = 'policy:roles/owner'
    quickstart_searchalliampolicies.search_all_iam_policies(scope, query=query)
    (out, _) = capsys.readouterr()
    assert 'roles/owner' in out