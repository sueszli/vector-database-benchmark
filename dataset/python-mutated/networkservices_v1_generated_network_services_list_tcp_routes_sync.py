from google.cloud import network_services_v1

def sample_list_tcp_routes():
    if False:
        for i in range(10):
            print('nop')
    client = network_services_v1.NetworkServicesClient()
    request = network_services_v1.ListTcpRoutesRequest(parent='parent_value')
    page_result = client.list_tcp_routes(request=request)
    for response in page_result:
        print(response)