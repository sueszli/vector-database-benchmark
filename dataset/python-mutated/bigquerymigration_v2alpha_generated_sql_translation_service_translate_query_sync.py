from google.cloud import bigquery_migration_v2alpha

def sample_translate_query():
    if False:
        for i in range(10):
            print('nop')
    client = bigquery_migration_v2alpha.SqlTranslationServiceClient()
    request = bigquery_migration_v2alpha.TranslateQueryRequest(parent='parent_value', source_dialect='TERADATA', query='query_value')
    response = client.translate_query(request=request)
    print(response)