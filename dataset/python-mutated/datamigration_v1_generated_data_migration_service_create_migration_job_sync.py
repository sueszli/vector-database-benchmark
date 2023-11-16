from google.cloud import clouddms_v1

def sample_create_migration_job():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    migration_job = clouddms_v1.MigrationJob()
    migration_job.reverse_ssh_connectivity.vm_ip = 'vm_ip_value'
    migration_job.reverse_ssh_connectivity.vm_port = 775
    migration_job.type_ = 'CONTINUOUS'
    migration_job.source = 'source_value'
    migration_job.destination = 'destination_value'
    request = clouddms_v1.CreateMigrationJobRequest(parent='parent_value', migration_job_id='migration_job_id_value', migration_job=migration_job)
    operation = client.create_migration_job(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)