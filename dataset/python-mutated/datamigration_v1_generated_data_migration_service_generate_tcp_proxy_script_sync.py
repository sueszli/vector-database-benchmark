from google.cloud import clouddms_v1

def sample_generate_tcp_proxy_script():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.GenerateTcpProxyScriptRequest(vm_name='vm_name_value', vm_machine_type='vm_machine_type_value', vm_subnet='vm_subnet_value')
    response = client.generate_tcp_proxy_script(request=request)
    print(response)