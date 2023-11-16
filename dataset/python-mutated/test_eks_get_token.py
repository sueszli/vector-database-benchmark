from __future__ import annotations
import contextlib
import json
import os
import runpy
from io import StringIO
from unittest import mock
from unittest.mock import ANY
import pytest
import time_machine
from tests.test_utils import AIRFLOW_MAIN_FOLDER

class TestGetEksToken:

    @mock.patch('airflow.providers.amazon.aws.hooks.eks.EksHook')
    @time_machine.travel('1995-02-14', tick=False)
    @pytest.mark.parametrize('args, expected_aws_conn_id, expected_region_name', [[['airflow.providers.amazon.aws.utils.eks_get_token', '--region-name', 'test-region', '--aws-conn-id', 'test-id', '--cluster-name', 'test-cluster'], 'test-id', 'test-region'], [['airflow.providers.amazon.aws.utils.eks_get_token', '--region-name', 'test-region', '--cluster-name', 'test-cluster'], None, 'test-region'], [['airflow.providers.amazon.aws.utils.eks_get_token', '--cluster-name', 'test-cluster'], None, None]])
    def test_run(self, mock_eks_hook, args, expected_aws_conn_id, expected_region_name):
        if False:
            print('Hello World!')
        mock_eks_hook.return_value.fetch_access_token_for_cluster.return_value = 'k8s-aws-v1.aHR0cDovL2V4YW1wbGUuY29t'
        with mock.patch('sys.argv', args), contextlib.redirect_stdout(StringIO()) as temp_stdout:
            os.chdir(AIRFLOW_MAIN_FOLDER)
            runpy.run_path('airflow/providers/amazon/aws/utils/eks_get_token.py', run_name='__main__')
        json_output = json.loads(temp_stdout.getvalue())
        assert {'apiVersion': 'client.authentication.k8s.io/v1alpha1', 'kind': 'ExecCredential', 'spec': {}, 'status': {'expirationTimestamp': ANY, 'token': 'k8s-aws-v1.aHR0cDovL2V4YW1wbGUuY29t'}} == json_output
        assert json_output['status']['expirationTimestamp'].startswith('1995-02-')
        mock_eks_hook.assert_called_once_with(aws_conn_id=expected_aws_conn_id, region_name=expected_region_name)
        mock_eks_hook.return_value.fetch_access_token_for_cluster.assert_called_once_with('test-cluster')