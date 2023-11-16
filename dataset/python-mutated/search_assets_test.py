import pytest
import search_assets

@pytest.mark.skip(reason='Needs fixing by CODEOWNER - issue #8541')
def test_search_assets(capsys, project_id, random_existing_tag_template_id):
    if False:
        for i in range(10):
            print('nop')
    override_values = {'project_id': project_id, 'tag_template_id': random_existing_tag_template_id}
    search_assets.search_assets(override_values)
    (out, err) = capsys.readouterr()
    assert 'Results in project:' in out
    assert random_existing_tag_template_id in out