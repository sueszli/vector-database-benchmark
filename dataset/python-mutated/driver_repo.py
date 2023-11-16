from datetime import timedelta
import pandas as pd
import yaml
from feast import Entity, FeatureService, FeatureView, Field, PushSource, RequestSource, SnowflakeSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64
driver = Entity(name='driver', join_keys=['driver_id'])
project_name = yaml.safe_load(open('feature_store.yaml'))['project']
driver_stats_source = SnowflakeSource(database=yaml.safe_load(open('feature_store.yaml'))['offline_store']['database'], table=f'{project_name}_feast_driver_hourly_stats', timestamp_field='event_timestamp', created_timestamp_column='created')
driver_stats_fv = FeatureView(name='driver_hourly_stats', entities=[driver], ttl=timedelta(weeks=52 * 10), schema=[Field(name='conv_rate', dtype=Float32), Field(name='acc_rate', dtype=Float32), Field(name='avg_daily_trips', dtype=Int64)], source=driver_stats_source, tags={'team': 'driver_performance'})
input_request = RequestSource(name='vals_to_add', schema=[Field(name='val_to_add', dtype=Int64), Field(name='val_to_add_2', dtype=Int64)])

@on_demand_feature_view(sources=[driver_stats_fv, input_request], schema=[Field(name='conv_rate_plus_val1', dtype=Float64), Field(name='conv_rate_plus_val2', dtype=Float64)])
def transformed_conv_rate(inputs: pd.DataFrame) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    df = pd.DataFrame()
    df['conv_rate_plus_val1'] = inputs['conv_rate'] + inputs['val_to_add']
    df['conv_rate_plus_val2'] = inputs['conv_rate'] + inputs['val_to_add_2']
    return df
driver_activity_v1 = FeatureService(name='driver_activity_v1', features=[driver_stats_fv[['conv_rate']], transformed_conv_rate])
driver_activity_v2 = FeatureService(name='driver_activity_v2', features=[driver_stats_fv, transformed_conv_rate])
driver_stats_push_source = PushSource(name='driver_stats_push_source', batch_source=driver_stats_source)
driver_stats_fresh_fv = FeatureView(name='driver_hourly_stats_fresh', entities=[driver], ttl=timedelta(weeks=52 * 10), schema=[Field(name='conv_rate', dtype=Float32), Field(name='acc_rate', dtype=Float32), Field(name='avg_daily_trips', dtype=Int64)], online=True, source=driver_stats_push_source, tags={'team': 'driver_performance'})

@on_demand_feature_view(sources=[driver_stats_fresh_fv, input_request], schema=[Field(name='conv_rate_plus_val1', dtype=Float64), Field(name='conv_rate_plus_val2', dtype=Float64)])
def transformed_conv_rate_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
    if False:
        print('Hello World!')
    df = pd.DataFrame()
    df['conv_rate_plus_val1'] = inputs['conv_rate'] + inputs['val_to_add']
    df['conv_rate_plus_val2'] = inputs['conv_rate'] + inputs['val_to_add_2']
    return df
driver_activity_v3 = FeatureService(name='driver_activity_v3', features=[driver_stats_fresh_fv, transformed_conv_rate_fresh])