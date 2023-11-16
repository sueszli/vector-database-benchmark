import re
import pandas as pd
import pytest
from qa_engine import enrichments

@pytest.fixture
def enriched_catalog(oss_catalog, cloud_catalog, adoption_metrics_per_connector_version) -> pd.DataFrame:
    if False:
        print('Hello World!')
    return enrichments.get_enriched_catalog(oss_catalog, cloud_catalog, adoption_metrics_per_connector_version)

@pytest.fixture
def enriched_catalog_columns(enriched_catalog: pd.DataFrame) -> set:
    if False:
        return 10
    return set(enriched_catalog.columns)

def test_merge_performed_correctly(enriched_catalog, oss_catalog):
    if False:
        for i in range(10):
            print('nop')
    assert len(enriched_catalog) == len(oss_catalog)

def test_new_columns_are_added(enriched_catalog_columns):
    if False:
        return 10
    expected_new_columns = {'is_on_cloud', 'connector_name', 'connector_technical_name', 'connector_version', 'number_of_connections', 'number_of_users', 'succeeded_syncs_count', 'failed_syncs_count', 'total_syncs_count', 'sync_success_rate'}
    assert expected_new_columns.issubset(enriched_catalog_columns)

def test_no_column_are_removed_and_lowercased(enriched_catalog_columns, oss_catalog):
    if False:
        print('Hello World!')
    for column in oss_catalog:
        assert re.sub('(?<!^)(?=[A-Z])', '_', column).lower() in enriched_catalog_columns

def test_support_level_not_null(enriched_catalog):
    if False:
        for i in range(10):
            print('nop')
    assert len(enriched_catalog['support_level'].dropna()) == len(enriched_catalog['support_level'])