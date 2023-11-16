import datetime
import os
from typing import Union
import uuid
import yaml
from azure.ai.resources.entities.models import Model
CHAT_SCORING_SCRIPT_TEMPLATE = '\nimport asyncio\nimport json\nimport os\nfrom pathlib import Path\nfrom inspect import iscoroutinefunction\nimport sys\nfrom azureml.contrib.services.aml_request import AMLRequest, rawhttp\nfrom azureml.contrib.services.aml_response import AMLResponse\nimport json\nimport importlib\n\n\ndef response_to_dict(response):\n    for resp in response:\n        yield json.dumps(resp) + "\\n"\n\ndef init():\n    """\n    This function is called when the container is initialized/started, typically after create/update of the deployment.\n    You can write the logic here to perform init operations like caching the model in memory\n    """\n    # AZUREML_MODEL_DIR is an environment variable created during deployment.\n    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)\n    # Please provide your model\'s folder name if there is one\n    print(os.getenv("AZUREML_MODEL_DIR"))\n    resolved_path = str(Path(os.getenv("AZUREML_MODEL_DIR")).resolve() / "{}")\n    sys.path.append(resolved_path)\n\n@rawhttp\ndef run(raw_data: AMLRequest):\n    """\n    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.\n    In the example we extract the data from the json input and call the scikit-learn model\'s predict()\n    method and return the result back\n    """\n    raw_data = json.loads(raw_data.data)\n    messages = raw_data["messages"]\n    messages = [\n        {{\n            "role": message["role"],\n            "content": message["content"], \n        }} \n        for message in messages if message.get("kind", "text") == "text"\n    ]\n    stream = raw_data.get("stream", False)\n    session_state = raw_data.get("sessionState", raw_data.get("session_state", None))\n    context = raw_data.get("context", {{}})\n    from {} import chat_completion\n    if iscoroutinefunction(chat_completion):\n        response = asyncio.run(\n            chat_completion(\n                messages,\n                stream,\n                session_state,\n                context,\n            )\n        )\n    else:\n        response = chat_completion(\n            messages,\n            stream,\n            session_state,\n            context,\n        )\n    if stream:\n        aml_response = AMLResponse(response_to_dict(response), 200)\n        aml_response.headers["Content-Type"] = "application/jsonl"\n        return aml_response\n    return json.loads(json.dumps(response))\n\n'

def create_chat_scoring_script(directory: Union[str, os.PathLike], chat_module: str, model_dir_name: str=None) -> None:
    if False:
        i = 10
        return i + 15
    score_file_path = f'{str(directory)}/score.py'
    with open(score_file_path, 'w+') as f:
        f.write(CHAT_SCORING_SCRIPT_TEMPLATE.format(model_dir_name if model_dir_name else '', chat_module))

def create_mlmodel_file(model: Model):
    if False:
        while True:
            i = 10
    with open(f'{model.path}/MLmodel', 'w+') as f:
        now = datetime.datetime.utcnow()
        mlmodel_dict = {'flavors': {'python_function': {'code': '.', 'data': '.', 'env': model.conda_file, 'loader_module': model.loader_module}}, 'model_uuid': str(uuid.uuid4()).replace('-', ''), 'utc_time_created': now.strftime('%Y-%m-%d %H:%M:%S.%f')}
        yaml.safe_dump(mlmodel_dict, f)