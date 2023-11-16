from __future__ import annotations
from unittest import mock
from airflow.providers.google.ads.transfers.ads_to_gcs import GoogleAdsToGcsOperator
from tests.providers.google.ads.operators.test_ads import BUCKET, CLIENT_IDS, FIELDS_TO_EXTRACT, GCS_OBJ_PATH, IMPERSONATION_CHAIN, QUERY, api_version, gcp_conn_id, google_ads_conn_id

class TestGoogleAdsToGcsOperator:

    @mock.patch('airflow.providers.google.ads.transfers.ads_to_gcs.GoogleAdsHook')
    @mock.patch('airflow.providers.google.ads.transfers.ads_to_gcs.GCSHook')
    def test_execute(self, mock_gcs_hook, mock_ads_hook):
        if False:
            while True:
                i = 10
        op = GoogleAdsToGcsOperator(gcp_conn_id=gcp_conn_id, google_ads_conn_id=google_ads_conn_id, client_ids=CLIENT_IDS, query=QUERY, attributes=FIELDS_TO_EXTRACT, obj=GCS_OBJ_PATH, bucket=BUCKET, task_id='run_operator', impersonation_chain=IMPERSONATION_CHAIN, api_version=api_version)
        op.execute({})
        mock_ads_hook.assert_called_once_with(gcp_conn_id=gcp_conn_id, google_ads_conn_id=google_ads_conn_id, api_version=api_version)
        mock_ads_hook.return_value.search.assert_called_once_with(client_ids=CLIENT_IDS, query=QUERY, page_size=10000)
        mock_gcs_hook.assert_called_once_with(gcp_conn_id=gcp_conn_id, impersonation_chain=IMPERSONATION_CHAIN)
        mock_gcs_hook.return_value.upload.assert_called_once_with(bucket_name=BUCKET, object_name=GCS_OBJ_PATH, filename=mock.ANY, gzip=False)