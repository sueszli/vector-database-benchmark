from __future__ import annotations
from typing import Optional
from google.auth import default
from google.cloud import aiplatform
import pandas as pd
import vertexai
from vertexai.preview.language_models import CodeGenerationModel, TuningEvaluationSpec
(credentials, _) = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])

def tune_code_generation_model(project_id: str, location: str, training_data: pd.DataFrame | str, train_steps: int=300, evaluation_dataset: Optional[str]=None, tensorboard_instance_name: Optional[str]=None) -> None:
    if False:
        i = 10
        return i + 15
    'Tune a new model, based on a prompt-response data.\n\n    "training_data" can be either the GCS URI of a file formatted in JSONL format\n    (for example: training_data=f\'gs://{bucket}/{filename}.jsonl\'), or a pandas\n    DataFrame. Each training example should be JSONL record with two keys, for\n    example:\n      {\n        "input_text": <input prompt>,\n        "output_text": <associated output>\n      },\n    or the pandas DataFame should contain two columns:\n      [\'input_text\', \'output_text\']\n    with rows for each training example.\n\n    Args:\n      project_id: GCP Project ID, used to initialize vertexai\n      location: GCP Region, used to initialize vertexai\n      training_data: GCS URI of jsonl file or pandas dataframe of training data\n      train_steps: Number of training steps to use when tuning the model.\n      evaluation_dataset: GCS URI of jsonl file of evaluation data.\n      tensorboard_instance_name: The full name of the existing Vertex AI TensorBoard instance:\n        projects/PROJECT_ID/locations/LOCATION_ID/tensorboards/TENSORBOARD_INSTANCE_ID\n        Note that this instance must be in the same region as your tuning job.\n    '
    vertexai.init(project=project_id, location=location, credentials=credentials)
    eval_spec = TuningEvaluationSpec(evaluation_data=evaluation_dataset)
    eval_spec.tensorboard = aiplatform.Tensorboard(tensorboard_name=tensorboard_instance_name)
    model = CodeGenerationModel.from_pretrained('code-bison@001')
    model.tune_model(training_data=training_data, train_steps=train_steps, tuning_job_location='europe-west4', tuned_model_location=location, tuning_evaluation_spec=eval_spec)
    print(model._job.status)
    return model
if __name__ == '__main__':
    tune_code_generation_model()