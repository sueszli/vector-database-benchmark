from google.cloud import phishingprotection_v1beta1

def sample_report_phishing():
    if False:
        for i in range(10):
            print('nop')
    client = phishingprotection_v1beta1.PhishingProtectionServiceV1Beta1Client()
    request = phishingprotection_v1beta1.ReportPhishingRequest(parent='parent_value', uri='uri_value')
    response = client.report_phishing(request=request)
    print(response)