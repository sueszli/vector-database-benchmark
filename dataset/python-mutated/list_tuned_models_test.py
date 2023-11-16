import os
import backoff
from google.api_core.exceptions import ResourceExhausted
from google.cloud import aiplatform
import list_tuned_models
_PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
_LOCATION = 'us-central1'

@backoff.on_exception(backoff.expo, ResourceExhausted, max_time=10)
def test_list_tuned_models() -> None:
    if False:
        i = 10
        return i + 15
    tuned_model_names = list_tuned_models.list_tuned_models(_PROJECT_ID, _LOCATION)
    filtered_models_counter = 0
    for tuned_model_name in tuned_model_names:
        model_registry = aiplatform.models.ModelRegistry(model=tuned_model_name)
        if 'Vertex LLM Test Fixture (list_tuned_models_test.py::test_list_tuned_models)' in model_registry.get_version_info('1').model_display_name:
            filtered_models_counter += 1
    assert filtered_models_counter == 0