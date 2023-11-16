from google.cloud import gke_backup_v1

def sample_update_restore_plan():
    if False:
        i = 10
        return i + 15
    client = gke_backup_v1.BackupForGKEClient()
    restore_plan = gke_backup_v1.RestorePlan()
    restore_plan.backup_plan = 'backup_plan_value'
    restore_plan.cluster = 'cluster_value'
    restore_plan.restore_config.all_namespaces = True
    request = gke_backup_v1.UpdateRestorePlanRequest(restore_plan=restore_plan)
    operation = client.update_restore_plan(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)