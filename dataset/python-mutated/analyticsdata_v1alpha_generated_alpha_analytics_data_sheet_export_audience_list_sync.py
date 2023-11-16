from google.analytics import data_v1alpha

def sample_sheet_export_audience_list():
    if False:
        print('Hello World!')
    client = data_v1alpha.AlphaAnalyticsDataClient()
    request = data_v1alpha.SheetExportAudienceListRequest(name='name_value')
    response = client.sheet_export_audience_list(request=request)
    print(response)