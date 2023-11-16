from google.cloud import iap_v1

def sample_create_tunnel_dest_group():
    if False:
        return 10
    client = iap_v1.IdentityAwareProxyAdminServiceClient()
    tunnel_dest_group = iap_v1.TunnelDestGroup()
    tunnel_dest_group.name = 'name_value'
    request = iap_v1.CreateTunnelDestGroupRequest(parent='parent_value', tunnel_dest_group=tunnel_dest_group, tunnel_dest_group_id='tunnel_dest_group_id_value')
    response = client.create_tunnel_dest_group(request=request)
    print(response)