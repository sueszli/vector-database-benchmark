import os
from typing import Union
import shutil
from pathlib import Path
from azure.ai.resources.entities.models import Model
from azure.ai.resources._utils._scoring_script_utils import create_chat_scoring_script
CHAT_COMPLETION_DOCKERFILE = '\nFROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04\nENV AZUREML_MODEL_DIR=/var/azureml-package-resources/azureml-models/model/\nCOPY ./* /var/azureml-package-resources/azureml-models/model/\nRUN conda env create -n azureml_env -f ${{AZUREML_MODEL_DIR}}/{}\nENV PATH=/opt/miniconda/envs/azureml_env/bin:$PATH\nRUN pip install azureml-inference-server-http~=0.8.0\nCOPY ./scoring/score.py /var/azureml-package-resources/score.py\nENV AZUREML_ENTRY_SCRIPT=/var/azureml-package-resources/score.py\nCMD ["runsvdir", "/var/runit"]\n'
MLFLOW_MODEL_DOCKERFILE = '\nFROM mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:20230920.v2\nUSER root\nENV MLFLOW_MODEL_FOLDER=model_files\nENV AZUREML_MODEL_DIR=/var/azureml-package-resources/azureml-models/model\nRUN mkdir -p /var/azureml-package-resources/azureml-models/model\nCOPY ./model_files /var/azureml-package-resources/azureml-models/model/${{MLFLOW_MODEL_FOLDER}}/\nRUN conda env create -n azureml_env -f ${{AZUREML_MODEL_DIR}}/${{MLFLOW_MODEL_FOLDER}}/{}\nENV PATH=/opt/miniconda/envs/azureml_env/bin:$PATH\nRUN pip install azureml-inference-server-http~=0.8.0\nUSER dockeruser\nCMD ["runsvdir", "/var/runit"]\n'

def create_dockerfile(model: Model, output: Union[str, os.PathLike], type: str):
    if False:
        return 10
    if type == 'mlflow':
        to_copy_dir = output / 'model_files'
    else:
        to_copy_dir = output
        scoring_path = Path(output) / 'scoring'
        os.makedirs(str(scoring_path), exist_ok=True)
        create_chat_scoring_script(scoring_path, model.chat_module)
    shutil.copytree(model.path, str(to_copy_dir), dirs_exist_ok=True)
    with open(f'{str(output)}/Dockerfile', 'w+') as f:
        if type == 'chat':
            formatted_dockerfile = CHAT_COMPLETION_DOCKERFILE.format(model.conda_file)
        elif type == 'mlflow':
            formatted_dockerfile = MLFLOW_MODEL_DOCKERFILE.format(model.conda_file)
        f.write(formatted_dockerfile)