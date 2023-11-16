def search_assets(override_values):
    if False:
        return 10
    'Searches Data Catalog entries for a given project.'
    from google.cloud import datacatalog_v1
    datacatalog = datacatalog_v1.DataCatalogClient()
    project_id = 'project_id'
    search_string = 'type=dataset'
    project_id = override_values.get('project_id', project_id)
    tag_template_id = override_values.get('tag_template_id', search_string)
    search_string = f'name:{tag_template_id}'
    scope = datacatalog_v1.types.SearchCatalogRequest.Scope()
    scope.include_project_ids.append(project_id)
    search_results = datacatalog.search_catalog(scope=scope, query=search_string)
    print('Results in project:')
    for result in search_results:
        print(result)