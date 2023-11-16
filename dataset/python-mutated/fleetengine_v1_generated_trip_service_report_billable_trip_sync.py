from google.maps import fleetengine_v1

def sample_report_billable_trip():
    if False:
        return 10
    client = fleetengine_v1.TripServiceClient()
    request = fleetengine_v1.ReportBillableTripRequest(name='name_value', country_code='country_code_value')
    client.report_billable_trip(request=request)