import pandas as pd

def get_enriched_catalog(oss_catalog: pd.DataFrame, cloud_catalog: pd.DataFrame, adoption_metrics_per_connector_version: pd.DataFrame) -> pd.DataFrame:
    if False:
        print('Hello World!')
    'Merge OSS and Cloud catalog in a single dataframe on their definition id.\n    Transformations:\n      - Rename columns to snake case.\n      - Rename name column to connector_name.\n      - Rename docker_image_tag to connector_version.\n      - Replace null value for support_level with unknown.\n    Enrichments:\n      - is_on_cloud: determined by the merge operation results.\n      - connector_technical_name: built from the docker repository field. airbyte/source-pokeapi -> source-pokeapi.\n      - Adoptions metrics: add the columns from the adoption_metrics_per_connector_version dataframe.\n    Args:\n        oss_catalog (pd.DataFrame): The open source catalog dataframe.\n        cloud_catalog (pd.DataFrame): The cloud catalog dataframe.\n        adoption_metrics_per_connector_version (pd.DataFrame): The crowd sourced adoptions metrics.\n\n    Returns:\n        pd.DataFrame: The enriched catalog.\n    '
    enriched_catalog = pd.merge(oss_catalog, cloud_catalog, how='left', on='connector_definition_id', indicator=True, suffixes=('', '_cloud'))
    enriched_catalog.columns = enriched_catalog.columns.str.replace('(?<=[a-z])(?=[A-Z])', '_', regex=True).str.lower()
    enriched_catalog = enriched_catalog[[c for c in enriched_catalog.columns if '_cloud' not in c]]
    enriched_catalog['is_on_cloud'] = enriched_catalog['_merge'] == 'both'
    enriched_catalog = enriched_catalog.drop(columns='_merge')
    enriched_catalog['connector_name'] = enriched_catalog['name']
    enriched_catalog['connector_technical_name'] = enriched_catalog['docker_repository'].str.replace('airbyte/', '')
    enriched_catalog['connector_version'] = enriched_catalog['docker_image_tag']
    enriched_catalog['support_level'] = enriched_catalog['support_level'].fillna('unknown')
    enriched_catalog = enriched_catalog.merge(adoption_metrics_per_connector_version, how='left', on=['connector_definition_id', 'connector_version'])
    enriched_catalog = enriched_catalog.drop_duplicates(subset=['connector_definition_id', 'connector_version'])
    enriched_catalog[adoption_metrics_per_connector_version.columns] = enriched_catalog[adoption_metrics_per_connector_version.columns].fillna(0)
    return enriched_catalog