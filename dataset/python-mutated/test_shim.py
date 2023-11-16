"""Unit tests for versionless import."""

def test_shim():
    if False:
        while True:
            i = 10
    from google.cloud import bigquery_datatransfer, bigquery_datatransfer_v1
    assert sorted(bigquery_datatransfer.__all__) == sorted(bigquery_datatransfer_v1.__all__)
    for name in bigquery_datatransfer.__all__:
        found = getattr(bigquery_datatransfer, name)
        expected = getattr(bigquery_datatransfer_v1, name)
        assert found is expected