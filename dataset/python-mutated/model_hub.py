import os
import shutil
from typing import IO
import requests
OMNIHUB_HOME = 'OMNIHUB_HOME'
if os.environ.__contains__(OMNIHUB_HOME):
    omnihub_dir = os.environ[OMNIHUB_HOME]
else:
    omnihub_dir = os.path.join(os.path.expanduser('~'), '.omnihub')
if not os.path.exists(omnihub_dir):
    os.mkdir(omnihub_dir)

class ModelHub(object):

    def __init__(self, framework_name: str, base_url: str):
        if False:
            i = 10
            return i + 15
        self.framework_name = framework_name
        self.stage_model_dir = os.path.join(omnihub_dir, self.framework_name)
        if not os.path.exists(self.stage_model_dir):
            os.mkdir(self.stage_model_dir)
        self.base_url = base_url

    def _download_file(self, url: str, **kwargs):
        if False:
            return 10
        local_filename = os.path.join(self.stage_model_dir, url.split('/')[-1])
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename

    def download_model(self, model_path, **kwargs) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Meant to be overridden by sub classes.\n        Handles downloading a model with the target URL\n        at the path specified.\n        :param model_path:  the path to the model from the base URL of the web service\n        :return: the path to the original model\n        '
        model_path = self._download_file(f'{self.base_url}/{model_path}')
        return model_path

    def stage_model(self, model_path: str, model_name: str):
        if False:
            for i in range(10):
                print('nop')
        '\n        Copy the model from its original path to the target\n        directory under self.stage_model_dir\n        :param model_path: the original path to the model downloaded\n        by the underlying framework\n        :param model_name: the name of the model file to save as\n        :return:\n        '
        shutil.copy(model_path, os.path.join(self.stage_model_dir, model_name))

    def stage_model_stream(self, model_path: IO, model_name: str):
        if False:
            return 10
        '\n        Copy the model from its original path to the target\n        directory under self.stage_model_dir\n        :param model_path: the original path to the model downloaded\n        by the underlying framework\n        :param model_name: the name of the model file to save as\n        :return:\n        '
        with open(os.path.join(self.stage_model_dir, model_name), 'wb+') as f:
            shutil.copyfileobj(model_path, f)