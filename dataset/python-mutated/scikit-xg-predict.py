"""Examples of using AI Platform's online prediction service,
   modified for scikit-learn and XGBoost."""
import googleapiclient.discovery

def predict_json(project, model, instances, version=None):
    if False:
        return 10
    'Send json data to a deployed model for prediction.\n    Args:\n        project (str): project where the AI Platform Model is deployed.\n        model (str): model name.\n        instances ([[float]]): List of input instances, where each input\n                   instance is a list of floats.\n        version: str, version of the model to target.\n    Returns:\n        Mapping[str: any]: dictionary of prediction results defined by the\n            model.\n    '
    service = googleapiclient.discovery.build('ml', 'v1')
    name = f'projects/{project}/models/{model}'
    if version is not None:
        name += f'/versions/{version}'
    response = service.projects().predict(name=name, body={'instances': instances}).execute()
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['predictions']