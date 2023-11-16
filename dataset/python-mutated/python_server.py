import abc
import asyncio
import base64
import os
import platform
from typing import TYPE_CHECKING, Any, Dict, Optional
import requests
import uvicorn
from fastapi import FastAPI
from lightning_utilities.core.imports import compare_version, module_available
from pydantic import BaseModel
from lightning.app.core.work import LightningWork
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.imports import _is_torch_available, requires
if TYPE_CHECKING:
    from lightning.app.frontend.frontend import Frontend
logger = Logger(__name__)
if not module_available('lightning_api_access') or not _is_torch_available():
    __doctest_skip__ = ['PythonServer', 'PythonServer.*']

def _get_device():
    if False:
        return 10
    import operator
    import torch
    _TORCH_GREATER_EQUAL_1_12 = compare_version('torch', operator.ge, '1.12.0')
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    if _TORCH_GREATER_EQUAL_1_12 and torch.backends.mps.is_available() and (platform.processor() in ('arm', 'arm64')):
        return torch.device('mps', local_rank)
    return torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

class _DefaultInputData(BaseModel):
    payload: str

class _DefaultOutputData(BaseModel):
    prediction: str

class Image(BaseModel):
    image: Optional[str] = None

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        if False:
            for i in range(10):
                print('nop')
        url = 'https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png'
        img = requests.get(url).content
        img = base64.b64encode(img).decode('UTF-8')
        return {'image': img}

    @staticmethod
    def request_code_sample(url: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f"""\nimport base64\nfrom pathlib import Path\nimport requests\n\nimgurl = "https://raw.githubusercontent.com/Lightning-AI/LAI-Triton-Server-Component/main/catimage.png"\nimg = requests.get(imgurl).content\nimg = base64.b64encode(img).decode("UTF-8")\nresponse = requests.post('{url}', json={{"image": img}})\n# If you are using basic authentication for your app, you should add your credentials to the request:\n# auth = requests.auth.HTTPBasicAuth('your_username', 'your_password')\n# response = requests.post('{url}', json={{"image": img}}, auth=auth)\n"""

    @staticmethod
    def response_code_sample() -> str:
        if False:
            while True:
                i = 10
        return 'img = response.json()["image"]\nimg = base64.b64decode(img.encode("utf-8"))\nPath("response.png").write_bytes(img)\n'

class Category(BaseModel):
    category: Optional[int] = None

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        if False:
            i = 10
            return i + 15
        return {'category': 463}

    @staticmethod
    def response_code_sample() -> str:
        if False:
            return 10
        return 'print("Predicted category is: ", response.json()["category"])\n'

class Text(BaseModel):
    text: Optional[str] = None

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        if False:
            while True:
                i = 10
        return {'text': 'A portrait of a person looking away from the camera'}

    @staticmethod
    def request_code_sample(url: str) -> str:
        if False:
            return 10
        return f"""\nimport base64\nfrom pathlib import Path\nimport requests\n\nresponse = requests.post('{url}', json={{\n    "text": "A portrait of a person looking away from the camera"\n}})\n# If you are using basic authentication for your app, you should add your credentials to the request:\n# response = requests.post('{url}', json={{\n#     "text": "A portrait of a person looking away from the camera"\n# }}, auth=requests.auth.HTTPBasicAuth('your_username', 'your_password'))\n"""

class Number(BaseModel):
    prediction: Optional[int] = None

    @staticmethod
    def get_sample_data() -> Dict[Any, Any]:
        if False:
            i = 10
            return i + 15
        return {'prediction': 463}

class PythonServer(LightningWork, abc.ABC):
    _start_method = 'spawn'

    @requires(['torch'])
    def __init__(self, input_type: type=_DefaultInputData, output_type: type=_DefaultOutputData, **kwargs: Any):
        if False:
            while True:
                i = 10
        'The PythonServer Class enables to easily get your machine learning server up and running.\n\n        Arguments:\n            input_type: Optional `input_type` to be provided. This needs to be a pydantic BaseModel class.\n                The default data type is good enough for the basic usecases and it expects the data\n                to be a json object that has one key called `payload`\n\n                .. code-block:: python\n\n                    input_data = {"payload": "some data"}\n\n                and this can be accessed as `request.payload` in the `predict` method.\n\n                .. code-block:: python\n\n                    def predict(self, request):\n                        data = request.payload\n\n            output_type: Optional `output_type` to be provided. This needs to be a pydantic BaseModel class.\n                The default data type is good enough for the basic usecases. It expects the return value of\n                the `predict` method to be a dictionary with one key called `prediction`.\n\n                .. code-block:: python\n\n                    def predict(self, request):\n                        # some code\n                        return {"prediction": "some data"}\n\n                and this can be accessed as `response.json()["prediction"]` in the client if\n                you are using requests library\n\n        Example:\n\n            >>> from lightning.app.components.serve.python_server import PythonServer\n            >>> from lightning.app import LightningApp\n            ...\n            >>> class SimpleServer(PythonServer):\n            ...\n            ...     def setup(self):\n            ...         self._model = lambda x: x + " " + x\n            ...\n            ...     def predict(self, request):\n            ...         return {"prediction": self._model(request.image)}\n            ...\n            >>> app = LightningApp(SimpleServer())\n\n        '
        super().__init__(parallel=True, **kwargs)
        if not issubclass(input_type, BaseModel):
            raise TypeError('input_type must be a pydantic BaseModel class')
        if not issubclass(output_type, BaseModel):
            raise TypeError('output_type must be a pydantic BaseModel class')
        self._input_type = input_type
        self._output_type = output_type
        self.ready = False

    def setup(self, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'This method is called before the server starts. Override this if you need to download the model or\n        initialize the weights, setting up pipelines etc.\n\n        Note that this will be called exactly once on every work machines. So if you have multiple machines for serving,\n        this will be called on each of them.\n\n        '
        return

    def configure_input_type(self) -> type:
        if False:
            while True:
                i = 10
        return self._input_type

    def configure_output_type(self) -> type:
        if False:
            for i in range(10):
                print('nop')
        return self._output_type

    @abc.abstractmethod
    def predict(self, request: Any) -> Any:
        if False:
            print('Hello World!')
        'This method is called when a request is made to the server.\n\n        This method must be overriden by the user with the prediction logic. The pre/post processing, actual prediction\n        using the model(s) etc goes here\n\n        '
        pass

    @staticmethod
    def _get_sample_dict_from_datatype(datatype: Any) -> dict:
        if False:
            print('Hello World!')
        if hasattr(datatype, 'get_sample_data'):
            return datatype.get_sample_data()
        datatype_props = datatype.schema()['properties']
        out: Dict[str, Any] = {}
        for (k, v) in datatype_props.items():
            if v['type'] == 'string':
                out[k] = 'data string'
            elif v['type'] == 'number':
                out[k] = 0.0
            elif v['type'] == 'integer':
                out[k] = 0
            elif v['type'] == 'boolean':
                out[k] = False
            else:
                raise TypeError('Unsupported type')
        return out

    def _attach_predict_fn(self, fastapi_app: FastAPI) -> None:
        if False:
            return 10
        input_type: type = self.configure_input_type()
        output_type: type = self.configure_output_type()

        def predict_fn_sync(request: input_type):
            if False:
                while True:
                    i = 10
            return self.predict(request)

        async def async_predict_fn(request: input_type):
            return await self.predict(request)
        if asyncio.iscoroutinefunction(self.predict):
            fastapi_app.post('/predict', response_model=output_type)(async_predict_fn)
        else:
            fastapi_app.post('/predict', response_model=output_type)(predict_fn_sync)

    def get_code_sample(self, url: str) -> Optional[str]:
        if False:
            while True:
                i = 10
        input_type: Any = self.configure_input_type()
        output_type: Any = self.configure_output_type()
        if not (hasattr(input_type, 'request_code_sample') and hasattr(output_type, 'response_code_sample')):
            return None
        return f'{input_type.request_code_sample(url)}\n{output_type.response_code_sample()}'

    def configure_layout(self) -> Optional['Frontend']:
        if False:
            print('Hello World!')
        try:
            from lightning_api_access import APIAccessFrontend
        except ModuleNotFoundError:
            logger.warn('Some dependencies to run the UI are missing. To resolve, run `pip install lightning-api-access`')
            return None
        class_name = self.__class__.__name__
        url = f'{self.url}/predict'
        try:
            request = self._get_sample_dict_from_datatype(self.configure_input_type())
            response = self._get_sample_dict_from_datatype(self.configure_output_type())
        except TypeError:
            return None
        frontend_payload = {'name': class_name, 'url': url, 'method': 'POST', 'request': request, 'response': response}
        code_sample = self.get_code_sample(url)
        if code_sample:
            frontend_payload['code_sample'] = code_sample
        return APIAccessFrontend(apis=[frontend_payload])

    def run(self, *args: Any, **kwargs: Any) -> Any:
        if False:
            return 10
        "Run method takes care of configuring and setting up a FastAPI server behind the scenes.\n\n        Normally, you don't need to override this method.\n\n        "
        self.setup(*args, **kwargs)
        fastapi_app = FastAPI()
        self._attach_predict_fn(fastapi_app)
        self.ready = True
        logger.info(f'Your {self.__class__.__qualname__} has started. View it in your browser: http://{self.host}:{self.port}')
        uvicorn.run(app=fastapi_app, host=self.host, port=self.port, log_level='error')