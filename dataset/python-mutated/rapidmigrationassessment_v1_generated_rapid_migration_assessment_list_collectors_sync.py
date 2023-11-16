from google.cloud import rapidmigrationassessment_v1

def sample_list_collectors():
    if False:
        for i in range(10):
            print('nop')
    client = rapidmigrationassessment_v1.RapidMigrationAssessmentClient()
    request = rapidmigrationassessment_v1.ListCollectorsRequest(parent='parent_value')
    page_result = client.list_collectors(request=request)
    for response in page_result:
        print(response)