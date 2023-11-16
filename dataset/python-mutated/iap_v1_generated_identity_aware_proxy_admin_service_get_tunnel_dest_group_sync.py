from google.cloud import iap_v1

def sample_get_tunnel_dest_group():
    if False:
        print('Hello World!')
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    request = iap_v1.GetTunnelDestGroupRequest(name='name_value')
    response = client.get_tunnel_dest_group(request=request)
    print(response)