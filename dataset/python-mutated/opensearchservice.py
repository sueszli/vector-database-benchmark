import re

def validate_search_service_engine_version(engine_version):
    if False:
        i = 10
        return i + 15
    '\n    Validate Engine Version for OpenSearchServiceDomain.\n    The value must be in the format OpenSearch_X.Y or Elasticsearch_X.Y\n    Property: Domain.EngineVersion\n    '
    engine_version_check = re.compile('^(OpenSearch_|Elasticsearch_)\\d{1,5}.\\d{1,5}')
    if engine_version_check.match(engine_version) is None:
        raise ValueError('OpenSearch EngineVersion must be in the format OpenSearch_X.Y or Elasticsearch_X.Y')
    return engine_version