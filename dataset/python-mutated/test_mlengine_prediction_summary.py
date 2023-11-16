from __future__ import annotations
import base64
import binascii
from unittest import mock
import dill
import pytest
try:
    from airflow.providers.google.cloud.utils import mlengine_prediction_summary
except ImportError as e:
    if 'apache_beam' in str(e):
        pytestmark = pytest.mark.skip(f'package apache_beam not present. Skipping all tests in {__name__}')

class TestJsonCode:

    def test_encode(self):
        if False:
            while True:
                i = 10
        assert b'{"a": 1}' == mlengine_prediction_summary.JsonCoder.encode({'a': 1})

    def test_decode(self):
        if False:
            return 10
        assert {'a': 1} == mlengine_prediction_summary.JsonCoder.decode('{"a": 1}')

class TestMakeSummary:

    def test_make_summary(self):
        if False:
            while True:
                i = 10
        print(mlengine_prediction_summary.MakeSummary(1, lambda x: x, []))

    def test_run_without_all_arguments_should_raise_exception(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(SystemExit):
            mlengine_prediction_summary.run()
        with pytest.raises(SystemExit):
            mlengine_prediction_summary.run(['--prediction_path=some/path'])
        with pytest.raises(SystemExit):
            mlengine_prediction_summary.run(['--prediction_path=some/path', '--metric_fn_encoded=encoded_text'])

    def test_run_should_fail_for_invalid_encoded_fn(self):
        if False:
            return 10
        with pytest.raises(binascii.Error):
            mlengine_prediction_summary.run(['--prediction_path=some/path', '--metric_fn_encoded=invalid_encoded_text', '--metric_keys=a'])

    def test_run_should_fail_if_enc_fn_is_not_callable(self):
        if False:
            return 10
        non_callable_value = 1
        fn_enc = base64.b64encode(dill.dumps(non_callable_value)).decode('utf-8')
        with pytest.raises(ValueError):
            mlengine_prediction_summary.run(['--prediction_path=some/path', '--metric_fn_encoded=' + fn_enc, '--metric_keys=a'])

    @mock.patch.object(mlengine_prediction_summary.beam.pipeline, 'PipelineOptions')
    @mock.patch.object(mlengine_prediction_summary.beam, 'Pipeline')
    @mock.patch.object(mlengine_prediction_summary.beam.io, 'ReadFromText')
    def test_run_should_not_fail_with_valid_fn(self, io_mock, pipeline_obj_mock, pipeline_mock):
        if False:
            i = 10
            return i + 15

        def metric_function():
            if False:
                print('Hello World!')
            return 1
        fn_enc = base64.b64encode(dill.dumps(metric_function)).decode('utf-8')
        mlengine_prediction_summary.run(['--prediction_path=some/path', '--metric_fn_encoded=' + fn_enc, '--metric_keys=a'])
        pipeline_mock.assert_called_once_with([])
        pipeline_obj_mock.assert_called_once()
        io_mock.assert_called_once()