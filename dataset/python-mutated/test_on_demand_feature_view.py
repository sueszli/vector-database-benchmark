import pandas as pd
from feast.feature_view import FeatureView
from feast.field import Field
from feast.infra.offline_stores.file_source import FileSource
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.types import Float32

def udf1(features_df: pd.DataFrame) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    df = pd.DataFrame()
    df['output1'] = features_df['feature1']
    df['output2'] = features_df['feature2']
    return df

def udf2(features_df: pd.DataFrame) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    df = pd.DataFrame()
    df['output1'] = features_df['feature1'] + 100
    df['output2'] = features_df['feature2'] + 100
    return df

def test_hash():
    if False:
        print('Hello World!')
    file_source = FileSource(name='my-file-source', path='test.parquet')
    feature_view = FeatureView(name='my-feature-view', entities=[], schema=[Field(name='feature1', dtype=Float32), Field(name='feature2', dtype=Float32)], source=file_source)
    sources = [feature_view]
    on_demand_feature_view_1 = OnDemandFeatureView(name='my-on-demand-feature-view', sources=sources, schema=[Field(name='output1', dtype=Float32), Field(name='output2', dtype=Float32)], udf=udf1, udf_string='udf1 source code')
    on_demand_feature_view_2 = OnDemandFeatureView(name='my-on-demand-feature-view', sources=sources, schema=[Field(name='output1', dtype=Float32), Field(name='output2', dtype=Float32)], udf=udf1, udf_string='udf1 source code')
    on_demand_feature_view_3 = OnDemandFeatureView(name='my-on-demand-feature-view', sources=sources, schema=[Field(name='output1', dtype=Float32), Field(name='output2', dtype=Float32)], udf=udf2, udf_string='udf2 source code')
    on_demand_feature_view_4 = OnDemandFeatureView(name='my-on-demand-feature-view', sources=sources, schema=[Field(name='output1', dtype=Float32), Field(name='output2', dtype=Float32)], udf=udf2, udf_string='udf2 source code', description='test')
    s1 = {on_demand_feature_view_1, on_demand_feature_view_2}
    assert len(s1) == 1
    s2 = {on_demand_feature_view_1, on_demand_feature_view_3}
    assert len(s2) == 2
    s3 = {on_demand_feature_view_3, on_demand_feature_view_4}
    assert len(s3) == 2
    s4 = {on_demand_feature_view_1, on_demand_feature_view_2, on_demand_feature_view_3, on_demand_feature_view_4}
    assert len(s4) == 3