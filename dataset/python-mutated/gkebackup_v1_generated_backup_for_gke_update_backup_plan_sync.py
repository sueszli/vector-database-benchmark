from google.cloud import gke_backup_v1

def sample_update_backup_plan():
    if False:
        while True:
            i = 10
    client = gke_backup_v1.BackupForGKEClient()
    backup_plan = gke_backup_v1.BackupPlan()
    backup_plan.cluster = 'cluster_value'
    request = gke_backup_v1.UpdateBackupPlanRequest(backup_plan=backup_plan)
    operation = client.update_backup_plan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)