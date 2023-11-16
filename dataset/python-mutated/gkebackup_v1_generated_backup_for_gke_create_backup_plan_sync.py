from google.cloud import gke_backup_v1

def sample_create_backup_plan():
    if False:
        i = 10
        return i + 15
    client = gke_backup_v1.BackupForGKEClient()
    backup_plan = gke_backup_v1.BackupPlan()
    backup_plan.cluster = 'cluster_value'
    request = gke_backup_v1.CreateBackupPlanRequest(parent='parent_value', backup_plan=backup_plan, backup_plan_id='backup_plan_id_value')
    operation = client.create_backup_plan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)