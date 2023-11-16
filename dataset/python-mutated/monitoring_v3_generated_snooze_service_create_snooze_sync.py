from google.cloud import monitoring_v3

def sample_create_snooze():
    if False:
        while True:
            i = 10
    client = monitoring_v3.SnoozeServiceClient()
    snooze = monitoring_v3.Snooze()
    snooze.name = 'name_value'
    snooze.display_name = 'display_name_value'
    request = monitoring_v3.CreateSnoozeRequest(parent='parent_value', snooze=snooze)
    response = client.create_snooze(request=request)
    print(response)