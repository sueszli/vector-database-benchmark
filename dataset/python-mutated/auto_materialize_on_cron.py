from dagster import AutoMaterializePolicy, AutoMaterializeRule, asset
materialize_on_cron_policy = AutoMaterializePolicy.eager().with_rules(AutoMaterializeRule.materialize_on_cron('0 9 * * *', timezone='US/Central'))

@asset(auto_materialize_policy=materialize_on_cron_policy)
def root_asset():
    if False:
        return 10
    ...