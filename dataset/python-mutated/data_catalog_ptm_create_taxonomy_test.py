import data_catalog_ptm_create_taxonomy

def test_create_taxonomy(capsys, project_id: str, random_taxonomy_display_name: str):
    if False:
        return 10
    data_catalog_ptm_create_taxonomy.create_taxonomy(project_id=project_id, location_id='us', display_name=random_taxonomy_display_name)
    (out, _) = capsys.readouterr()
    assert f'Created taxonomy projects/{project_id}/locations/us/taxonomies/' in out