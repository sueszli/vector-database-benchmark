from google.analytics import admin_v1beta

def sample_provision_account_ticket():
    if False:
        i = 10
        return i + 15
    client = admin_v1beta.AnalyticsAdminServiceClient()
    request = admin_v1beta.ProvisionAccountTicketRequest()
    response = client.provision_account_ticket(request=request)
    print(response)