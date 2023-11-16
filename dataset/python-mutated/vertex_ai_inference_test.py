import unittest
try:
    from apache_beam.ml.inference.vertex_ai_inference import _retry_on_appropriate_gcp_error
    from apache_beam.ml.inference.vertex_ai_inference import VertexAIModelHandlerJSON
    from google.api_core.exceptions import TooManyRequests
except ImportError:
    raise unittest.SkipTest('VertexAI dependencies are not installed')

class RetryOnClientErrorTest(unittest.TestCase):

    def test_retry_on_client_error_positive(self):
        if False:
            while True:
                i = 10
        e = TooManyRequests(message='fake service rate limiting')
        self.assertTrue(_retry_on_appropriate_gcp_error(e))

    def test_retry_on_client_error_negative(self):
        if False:
            while True:
                i = 10
        e = ValueError()
        self.assertFalse(_retry_on_appropriate_gcp_error(e))

class ModelHandlerArgConditions(unittest.TestCase):

    def test_exception_on_private_without_network(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, VertexAIModelHandlerJSON, endpoint_id='1', project='testproject', location='us-central1', private=True)
if __name__ == '__main__':
    unittest.main()