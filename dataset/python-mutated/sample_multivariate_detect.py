"""
FILE: sample_multivariate_detect.py

DESCRIPTION:
    This sample demonstrates how to use multivariate dataset to train a model and use the model to detect anomalies.

Prerequisites:
     * The Anomaly Detector client library for Python
     * A valid data feed

USAGE:
    python sample_multivariate_detect.py

    Set the environment variables with your own values before running the sample:
    1) ANOMALY_DETECTOR_KEY - your source Form Anomaly Detector API key.
    2) ANOMALY_DETECTOR_ENDPOINT - the endpoint to your source Anomaly Detector resource.
"""
import json
import os
import time
from datetime import datetime, timezone
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.ai.anomalydetector.models import *

class MultivariateSample:

    def __init__(self, subscription_key, anomaly_detector_endpoint):
        if False:
            return 10
        self.sub_key = subscription_key
        self.end_point = anomaly_detector_endpoint
        self.ad_client = AnomalyDetectorClient(self.end_point, AzureKeyCredential(self.sub_key))

    def list_models(self):
        if False:
            print('Hello World!')
        models = self.ad_client.list_multivariate_models(skip=0, top=10)
        return list(models)

    def train(self, body):
        if False:
            print('Hello World!')
        try:
            model_list = self.list_models()
            print('{:d} available models before training.'.format(len(model_list)))
            print('Training new model...(it may take a few minutes)')
            model = self.ad_client.train_multivariate_model(body)
            trained_model_id = model.model_id
            print('Training model id is {}'.format(trained_model_id))
            model_status = None
            model = None
            while model_status != ModelStatus.READY and model_status != ModelStatus.FAILED:
                model = self.ad_client.get_multivariate_model(trained_model_id)
                print(model)
                model_status = model.model_info.status
                print('Model is {}'.format(model_status))
                time.sleep(30)
            if model_status == ModelStatus.FAILED:
                print('Creating model failed.')
                print('Errors:')
                if len(model.model_info.errors) > 0:
                    print('Error code: {}. Message: {}'.format(model.model_info.errors[0].code, model.model_info.errors[0].message))
                else:
                    print('None')
            if model_status == ModelStatus.READY:
                model_list = self.list_models()
                print('Done.\n--------------------')
                print('{:d} available models after training.'.format(len(model_list)))
            return trained_model_id
        except HttpResponseError as e:
            print('Error code: {}'.format(e.error.code), 'Error message: {}'.format(e.error.message))
        except Exception as e:
            raise e
        return None

    def batch_detect(self, model_id, body):
        if False:
            print('Hello World!')
        try:
            result = self.ad_client.detect_multivariate_batch_anomaly(model_id, body)
            result_id = result.result_id
            r = self.ad_client.get_multivariate_batch_detection_result(result_id)
            print('Get detection result...(it may take a few seconds)')
            while r.summary.status != MultivariateBatchDetectionStatus.READY and r.summary.status != MultivariateBatchDetectionStatus.FAILED:
                r = self.ad_client.get_multivariate_batch_detection_result(result_id)
                print('Detection is {}'.format(r.summary.status))
                time.sleep(15)
            if r.summary.status == MultivariateBatchDetectionStatus.FAILED:
                print('Detection failed.')
                print('Errors:')
                if len(r.summary.errors) > 0:
                    print('Error code: {}. Message: {}'.format(r.summary.errors[0].code, r.summary.errors[0].message))
                else:
                    print('None')
                return None
            return r
        except HttpResponseError as e:
            print('Error code: {}'.format(e.error.code), 'Error message: {}'.format(e.error.message))
        except Exception as e:
            raise e
        return None

    def delete_model(self, model_id):
        if False:
            for i in range(10):
                print('nop')
        self.ad_client.delete_multivariate_model(model_id)
        model_list = self.list_models()
        print('{:d} available models after deletion.'.format(len(model_list)))

    def last_detect(self, model_id, variables):
        if False:
            print('Hello World!')
        r = self.ad_client.detect_multivariate_last_anomaly(model_id, variables)
        print('Get last detection result')
        return r
if __name__ == '__main__':
    SUBSCRIPTION_KEY = os.environ['ANOMALY_DETECTOR_KEY']
    ANOMALY_DETECTOR_ENDPOINT = os.environ['ANOMALY_DETECTOR_ENDPOINT']
    sample = MultivariateSample(SUBSCRIPTION_KEY, ANOMALY_DETECTOR_ENDPOINT)
    time_format = '%Y-%m-%dT%H:%M:%SZ'
    blob_url = '{Your Blob Url}'
    train_body = ModelInfo(data_source=blob_url, start_time=datetime.strptime('2021-01-02T00:00:00Z', time_format), end_time=datetime.strptime('2021-01-02T05:00:00Z', time_format), data_schema=DataSchema.MULTI_TABLE, display_name='sample', sliding_window=200, align_policy=AlignPolicy(align_mode=AlignMode.OUTER, fill_n_a_method=FillNAMethod.LINEAR, padding_value=0))
    model_id = sample.train(train_body)
    batch_inference_body = MultivariateBatchDetectionOptions(data_source=blob_url, top_contributor_count=10, start_time=datetime.strptime('2021-01-02T00:00:00Z', time_format), end_time=datetime.strptime('2021-01-02T05:00:00Z', time_format))
    result = sample.batch_detect(model_id, batch_inference_body)
    assert result is not None
    print('Result ID:\t', result.result_id)
    print('Result status:\t', result.summary.status)
    print('Result length:\t', len(result.results))
    for r in result.results:
        print('timestamp: {}, is_anomaly: {:<5}, anomaly score: {:.4f}, severity: {:.4f}, contributor count: {:<4d}'.format(r.timestamp, r.value.is_anomaly, r.value.score, r.value.severity, len(r.value.interpretation) if r.value.is_anomaly else 0))
        if r.value.interpretation:
            for contributor in r.value.interpretation:
                print('\tcontributor variable: {:<10}, contributor score: {:.4f}'.format(contributor.variable, contributor.contribution_score))
    with open('./sample_data/multivariate_sample_data.json') as f:
        variables_data = json.load(f)
    variables = []
    for item in variables_data['variables']:
        variables.append(VariableValues(variable=item['variable'], timestamps=item['timestamps'], values=item['values']))
    last_inference_body = MultivariateLastDetectionOptions(variables=variables, top_contributor_count=10)
    last_detect_result = sample.last_detect(model_id, last_inference_body)
    assert last_detect_result is not None
    print('Variable States:\t', last_detect_result.variable_states)
    print('Variable States length:\t', len(last_detect_result.variable_states))
    print('Results:\t', last_detect_result.results)
    print('Results length:\t', len(last_detect_result.results))
    sample.delete_model(model_id)