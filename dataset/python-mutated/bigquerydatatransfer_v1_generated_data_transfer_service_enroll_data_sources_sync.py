from google.cloud import bigquery_datatransfer_v1

def sample_enroll_data_sources():
    if False:
        i = 10
        return i + 15
    client = bigquery_datatransfer_v1.DataTransferServiceClient()
    request = bigquery_datatransfer_v1.EnrollDataSourcesRequest()
    client.enroll_data_sources(request=request)