from google.cloud import clouddms_v1

def sample_generate_ssh_script():
    if False:
        return 10
    client = clouddms_v1.DataMigrationServiceClient()
    vm_creation_config = clouddms_v1.VmCreationConfig()
    vm_creation_config.vm_machine_type = 'vm_machine_type_value'
    request = clouddms_v1.GenerateSshScriptRequest(vm_creation_config=vm_creation_config, vm='vm_value')
    response = client.generate_ssh_script(request=request)
    print(response)