import logging
import time
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from google.api_core.exceptions import ServerError
from google.api_core.exceptions import TooManyRequests
from google.cloud import aiplatform
from apache_beam.io.components.adaptive_throttler import AdaptiveThrottler
from apache_beam.metrics.metric import Metrics
from apache_beam.ml.inference import utils
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import PredictionResult
from apache_beam.utils import retry
MSEC_TO_SEC = 1000
LOGGER = logging.getLogger('VertexAIModelHandlerJSON')

def _retry_on_appropriate_gcp_error(exception):
    if False:
        print('Hello World!')
    '\n  Retry filter that returns True if a returned HTTP error code is 5xx or 429.\n  This is used to retry remote requests that fail, most notably 429\n  (TooManyRequests.)\n\n  Args:\n    exception: the returned exception encountered during the request/response\n      loop.\n\n  Returns:\n    boolean indication whether or not the exception is a Server Error (5xx) or\n      a TooManyRequests (429) error.\n  '
    return isinstance(exception, (TooManyRequests, ServerError))

class VertexAIModelHandlerJSON(ModelHandler[Any, PredictionResult, aiplatform.Endpoint]):

    def __init__(self, endpoint_id: str, project: str, location: str, experiment: Optional[str]=None, network: Optional[str]=None, private: bool=False, **kwargs):
        if False:
            print('Hello World!')
        'Implementation of the ModelHandler interface for Vertex AI.\n    **NOTE:** This API and its implementation are under development and\n    do not provide backward compatibility guarantees.\n    Unlike other ModelHandler implementations, this does not load the model\n    being used onto the worker and instead makes remote queries to a\n    Vertex AI endpoint. In that way it functions more like a mid-pipeline\n    IO. Public Vertex AI endpoints have a maximum request size of 1.5 MB.\n    If you wish to make larger requests and use a private endpoint, provide\n    the Compute Engine network you wish to use and set `private=True`\n\n    Args:\n      endpoint_id: the numerical ID of the Vertex AI endpoint to query\n      project: the GCP project name where the endpoint is deployed\n      location: the GCP location where the endpoint is deployed\n      experiment: optional. experiment label to apply to the\n        queries. See\n        https://cloud.google.com/vertex-ai/docs/experiments/intro-vertex-ai-experiments\n        for more information.\n      network: optional. the full name of the Compute Engine\n        network the endpoint is deployed on; used for private\n        endpoints. The network or subnetwork Dataflow pipeline\n        option must be set and match this network for pipeline\n        execution.\n        Ex: "projects/12345/global/networks/myVPC"\n      private: optional. if the deployed Vertex AI endpoint is\n        private, set to true. Requires a network to be provided\n        as well.\n    '
        self._env_vars = kwargs.get('env_vars', {})
        if private and network is None:
            raise ValueError('A VPC network must be provided to use a private endpoint.')
        aiplatform.init(project=project, location=location, experiment=experiment, network=network)
        self.endpoint_name = endpoint_id
        self.is_private = private
        _ = self._retrieve_endpoint(self.endpoint_name, self.is_private)
        self.throttled_secs = Metrics.counter(VertexAIModelHandlerJSON, 'cumulativeThrottlingSeconds')
        self.throttler = AdaptiveThrottler(window_ms=1, bucket_ms=1, overload_ratio=2)

    def _retrieve_endpoint(self, endpoint_id: str, is_private: bool) -> aiplatform.Endpoint:
        if False:
            print('Hello World!')
        'Retrieves an AI Platform endpoint and queries it for liveness/deployed\n    models.\n\n    Args:\n      endpoint_id: the numerical ID of the Vertex AI endpoint to retrieve.\n      is_private: a boolean indicating if the Vertex AI endpoint is a private\n        endpoint\n    Returns:\n      An aiplatform.Endpoint object\n    Raises:\n      ValueError: if endpoint is inactive or has no models deployed to it.\n    '
        if is_private:
            endpoint: aiplatform.Endpoint = aiplatform.PrivateEndpoint(endpoint_name=endpoint_id)
            LOGGER.debug('Treating endpoint %s as private', endpoint_id)
        else:
            endpoint = aiplatform.Endpoint(endpoint_name=endpoint_id)
            LOGGER.debug('Treating endpoint %s as public', endpoint_id)
        try:
            mod_list = endpoint.list_models()
        except Exception as e:
            raise ValueError('Failed to contact endpoint %s, got exception: %s', endpoint_id, e)
        if len(mod_list) == 0:
            raise ValueError('Endpoint %s has no models deployed to it.', endpoint_id)
        return endpoint

    def load_model(self) -> aiplatform.Endpoint:
        if False:
            while True:
                i = 10
        'Loads the Endpoint object used to build and send prediction request to\n    Vertex AI.\n    '
        ep = self._retrieve_endpoint(self.endpoint_name, self.is_private)
        return ep

    @retry.with_exponential_backoff(num_retries=5, retry_filter=_retry_on_appropriate_gcp_error)
    def get_request(self, batch: Sequence[Any], model: aiplatform.Endpoint, throttle_delay_secs: int, inference_args: Optional[Dict[str, Any]]):
        if False:
            print('Hello World!')
        while self.throttler.throttle_request(time.time() * MSEC_TO_SEC):
            LOGGER.info('Delaying request for %d seconds due to previous failures', throttle_delay_secs)
            time.sleep(throttle_delay_secs)
            self.throttled_secs.inc(throttle_delay_secs)
        try:
            req_time = time.time()
            prediction = model.predict(instances=list(batch), parameters=inference_args)
            self.throttler.successful_request(req_time * MSEC_TO_SEC)
            return prediction
        except TooManyRequests as e:
            LOGGER.warning('request was limited by the service with code %i', e.code)
            raise
        except Exception as e:
            LOGGER.error('unexpected exception raised as part of request, got %s', e)
            raise

    def run_inference(self, batch: Sequence[Any], model: aiplatform.Endpoint, inference_args: Optional[Dict[str, Any]]=None) -> Iterable[PredictionResult]:
        if False:
            for i in range(10):
                print('nop')
        ' Sends a prediction request to a Vertex AI endpoint containing batch\n    of inputs and matches that input with the prediction response from\n    the endpoint as an iterable of PredictionResults.\n\n    Args:\n      batch: a sequence of any values to be passed to the Vertex AI endpoint.\n        Should be encoded as the model expects.\n      model: an aiplatform.Endpoint object configured to access the desired\n        model.\n      inference_args: any additional arguments to send as part of the\n        prediction request.\n\n    Returns:\n      An iterable of Predictions.\n    '
        prediction = self.get_request(batch, model, throttle_delay_secs=5, inference_args=inference_args)
        return utils._convert_to_result(batch, prediction.predictions, prediction.deployed_model_id)

    def validate_inference_args(self, inference_args: Optional[Dict[str, Any]]):
        if False:
            while True:
                i = 10
        pass