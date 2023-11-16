from google.cloud import iap_v1

def sample_list_tunnel_dest_groups():
    if False:
        for i in range(10):
            print('nop')
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    request = iap_v1.ListTunnelDestGroupsRequest(parent='parent_value')
    page_result = client.list_tunnel_dest_groups(request=request)
    for response in page_result:
        print(response)