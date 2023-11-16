from google.cloud import beyondcorp_appconnectors_v1

def sample_report_status():
    if False:
        i = 10
        return i + 15
    client = beyondcorp_appconnectors_v1.AppConnectorsServiceClient()
    resource_info = beyondcorp_appconnectors_v1.ResourceInfo()
    resource_info.id = 'id_value'
    request = beyondcorp_appconnectors_v1.ReportStatusRequest(app_connector='app_connector_value', resource_info=resource_info)
    operation = client.report_status(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)