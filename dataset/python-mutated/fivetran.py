def scope_define_instance():
    if False:
        i = 10
        return i + 15
    from dagster_fivetran import FivetranResource
    from dagster import EnvVar
    fivetran_instance = FivetranResource(api_key=EnvVar('FIVETRAN_API_KEY'), api_secret=EnvVar('FIVETRAN_API_SECRET'))

def scope_load_assets_from_fivetran_instance():
    if False:
        for i in range(10):
            print('nop')
    from dagster_fivetran import FivetranResource
    from dagster import EnvVar
    fivetran_instance = FivetranResource(api_key=EnvVar('FIVETRAN_API_KEY'), api_secret=EnvVar('FIVETRAN_API_SECRET'))
    from dagster_fivetran import load_assets_from_fivetran_instance
    fivetran_assets = load_assets_from_fivetran_instance(fivetran_instance)

def scope_manually_define_fivetran_assets():
    if False:
        print('Hello World!')
    from dagster_fivetran import build_fivetran_assets
    fivetran_assets = build_fivetran_assets(connector_id='omit_constitutional', destination_tables=['public.survey_responses', 'public.surveys'])

def scope_fivetran_manual_config():
    if False:
        while True:
            i = 10
    from dagster_fivetran import FivetranResource
    from dagster import EnvVar
    fivetran_instance = FivetranResource(api_key=EnvVar('FIVETRAN_API_KEY'), api_secret=EnvVar('FIVETRAN_API_SECRET'))
    from dagster_fivetran import build_fivetran_assets
    from dagster import with_resources
    fivetran_assets = with_resources(build_fivetran_assets(connector_id='omit_constitutional', destination_tables=['public.survey_responses', 'public.surveys']), {'fivetran': fivetran_instance})

def scope_schedule_assets():
    if False:
        for i in range(10):
            print('nop')
    from dagster_fivetran import FivetranResource, load_assets_from_fivetran_instance
    from dagster import ScheduleDefinition, define_asset_job, AssetSelection, EnvVar, Definitions
    fivetran_instance = FivetranResource(api_key=EnvVar('FIVETRAN_API_KEY'), api_secret=EnvVar('FIVETRAN_API_SECRET'))
    fivetran_assets = load_assets_from_fivetran_instance(fivetran_instance)
    run_everything_job = define_asset_job('run_everything', selection='*')
    my_etl_job = define_asset_job('my_etl_job', AssetSelection.groups('my_fivetran_connection').downstream())
    defs = Definitions(assets=[fivetran_assets], schedules=[ScheduleDefinition(job=my_etl_job, cron_schedule='@daily'), ScheduleDefinition(job=run_everything_job, cron_schedule='@weekly')])

def scope_add_downstream_assets():
    if False:
        return 10
    import mock
    with mock.patch('dagster_snowflake_pandas.SnowflakePandasIOManager'):
        import json
        from dagster_fivetran import FivetranResource, load_assets_from_fivetran_instance
        from dagster import ScheduleDefinition, define_asset_job, asset, AssetIn, AssetKey, Definitions, AssetSelection, EnvVar, Definitions
        from dagster_snowflake_pandas import SnowflakePandasIOManager
        fivetran_instance = FivetranResource(api_key=EnvVar('FIVETRAN_API_KEY'), api_secret=EnvVar('FIVETRAN_API_SECRET'))
        fivetran_assets = load_assets_from_fivetran_instance(fivetran_instance, io_manager_key='snowflake_io_manager')

        @asset(ins={'survey_responses': AssetIn(key=AssetKey(['public', 'survey_responses']))})
        def survey_responses_file(survey_responses):
            if False:
                return 10
            with open('survey_responses.json', 'w', encoding='utf8') as f:
                f.write(json.dumps(survey_responses, indent=2))
        my_upstream_job = define_asset_job('my_upstream_job', AssetSelection.keys('survey_responses_file').upstream().required_multi_asset_neighbors())
        defs = Definitions(jobs=[my_upstream_job], assets=[fivetran_assets, survey_responses_file], resources={'snowflake_io_manager': SnowflakePandasIOManager(...)})