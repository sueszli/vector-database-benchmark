def scope_define_instance():
    if False:
        print('Hello World!')
    from dagster_dbt import DbtCloudClientResource
    from dagster import EnvVar
    dbt_cloud_instance = DbtCloudClientResource(auth_token=EnvVar('DBT_CLOUD_API_TOKEN'), account_id=EnvVar.int('DBT_CLOUD_ACCOUNT_ID'))
    return dbt_cloud_instance

def scope_load_assets_from_dbt_cloud_job():
    if False:
        return 10
    from dagster_dbt import DbtCloudClientResource
    from dagster import EnvVar
    dbt_cloud_instance = DbtCloudClientResource(auth_token=EnvVar('DBT_CLOUD_API_TOKEN'), account_id=EnvVar.int('DBT_CLOUD_ACCOUNT_ID'))
    from dagster_dbt import load_assets_from_dbt_cloud_job
    dbt_cloud_assets = load_assets_from_dbt_cloud_job(dbt_cloud=dbt_cloud_instance, job_id=33333)

def scope_schedule_dbt_cloud_assets(dbt_cloud_assets):
    if False:
        for i in range(10):
            print('nop')
    from dagster import ScheduleDefinition, define_asset_job, AssetSelection, Definitions
    run_everything_job = define_asset_job('run_everything_job', AssetSelection.all())
    defs = Definitions(assets=[dbt_cloud_assets], schedules=[ScheduleDefinition(job=run_everything_job, cron_schedule='@daily')])