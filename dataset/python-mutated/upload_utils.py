import logging
import os
from abc import ABC, abstractmethod
from typing import Optional
from huggingface_hub import HfApi, login
logger = logging.getLogger(__name__)

class BaseModelUpload(ABC):
    """Abstract base class for uploading trained model artifacts to different repositories.

    This class defines the interface for uploading trained model artifacts to various repositories such as Huggingface
    Hub, without specifying the concrete implementation for each repository. Subclasses of this base class must
    implement the 'login' and 'upload' methods.
    """

    @abstractmethod
    def login(self):
        if False:
            while True:
                i = 10
        'Abstract method to handle authentication with the target repository.\n\n        Subclasses must implement this method to provide the necessary authentication\n        mechanisms required by the repository where the model artifacts will be uploaded.\n\n        Raises:\n            NotImplementedError: If this method is not implemented in the subclass.\n        '
        raise NotImplementedError()

    @abstractmethod
    def upload(self, repo_id: str, model_path: str, repo_type: Optional[str]=None, private: Optional[bool]=False, commit_message: Optional[str]=None, commit_description: Optional[str]=None, dataset_file: Optional[str]=None, dataset_name: Optional[str]=None) -> bool:
        if False:
            return 10
        'Abstract method to upload trained model artifacts to the target repository.\n\n        Subclasses must implement this method to define the process of pushing model\n        artifacts to the respective repository. This may include creating a new model version,\n        uploading model files, and any other specific steps required by the model repository\n        service.\n\n        Returns:\n            bool: True if the model artifacts were successfully uploaded, False otherwise.\n\n        Raises:\n            NotImplementedError: If this method is not implemented in the subclass.\n        '
        raise NotImplementedError()

    @staticmethod
    def _validate_upload_parameters(repo_id: str, model_path: str, repo_type: Optional[str]=None, private: Optional[bool]=False, commit_message: Optional[str]=None, commit_description: Optional[str]=None):
        if False:
            while True:
                i = 10
        "Validate parameters before uploading trained model artifacts.\n\n        This method checks if the input parameters meet the necessary requirements before uploading\n        trained model artifacts to the target repository.\n\n        Args:\n            repo_id (str): The ID of the target repository. Each provider will verify their specific rules.\n            model_path (str): The path to the directory containing the trained model artifacts. It should contain\n                the model's weights, usually saved under 'model/model_weights'.\n            repo_type (str, optional): The type of the repository. Not used in the base class, but subclasses\n                may use it for specific repository implementations. Defaults to None.\n            private (bool, optional): Whether the repository should be private or not. Not used in the base class,\n                but subclasses may use it for specific repository implementations. Defaults to False.\n            commit_message (str, optional): A message to attach to the commit when uploading to version control\n                systems. Not used in the base class, but subclasses may use it for specific repository\n                implementations. Defaults to None.\n            commit_description (str, optional): A description of the commit when uploading to version control\n                systems. Not used in the base class, but subclasses may use it for specific repository\n                implementations. Defaults to None.\n\n        Raises:\n            FileNotFoundError: If the model_path does not exist.\n            Exception: If the trained model artifacts are not found at the expected location within model_path, or\n                if the artifacts are not in the required format (i.e., 'pytorch_model.bin' or 'adapter_model.bin').\n        "
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"The path '{model_path}' does not exist.")
        trained_model_artifacts_path = os.path.join(model_path, 'model', 'model_weights')
        if not os.path.exists(trained_model_artifacts_path):
            raise Exception(f"Model artifacts not found at {trained_model_artifacts_path}. It is possible that model at '{model_path}' hasn't been trained yet, or something wentwrong during training where the model's weights were not saved.")

class HuggingFaceHub(BaseModelUpload):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.api = None
        self.login()

    def login(self):
        if False:
            print('Hello World!')
        'Login to huggingface hub using the token stored in ~/.cache/huggingface/token and returns a HfApi client\n        object that can be used to interact with HF Hub.'
        cached_token_path = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'token')
        if not os.path.exists(cached_token_path):
            login(add_to_git_credential=True)
        with open(cached_token_path) as f:
            hf_token = f.read()
        hf_api = HfApi(token=hf_token)
        assert hf_api.token == hf_token
        self.api = hf_api

    @staticmethod
    def _validate_upload_parameters(repo_id: str, model_path: str, repo_type: Optional[str]=None, private: Optional[bool]=False, commit_message: Optional[str]=None, commit_description: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        "Validate parameters before uploading trained model artifacts.\n\n        This method checks if the input parameters meet the necessary requirements before uploading\n        trained model artifacts to the target repository.\n\n        Args:\n            repo_id (str): The ID of the target repository. It must be a namespace (user or an organization)\n                and a repository name separated by a '/'. For example, if your HF username is 'johndoe' and you\n                want to create a repository called 'test', the repo_id should be 'johndoe/test'.\n            model_path (str): The path to the directory containing the trained model artifacts. It should contain\n                the model's weights, usually saved under 'model/model_weights'.\n            repo_type (str, optional): The type of the repository. Not used in the base class, but subclasses\n                may use it for specific repository implementations. Defaults to None.\n            private (bool, optional): Whether the repository should be private or not. Not used in the base class,\n                but subclasses may use it for specific repository implementations. Defaults to False.\n            commit_message (str, optional): A message to attach to the commit when uploading to version control\n                systems. Not used in the base class, but subclasses may use it for specific repository\n                implementations. Defaults to None.\n            commit_description (str, optional): A description of the commit when uploading to version control\n                systems. Not used in the base class, but subclasses may use it for specific repository\n                implementations. Defaults to None.\n\n        Raises:\n            ValueError: If the repo_id does not have both a namespace and a repo name separated by a '/'.\n        "
        if '/' not in repo_id:
            raise ValueError('`repo_id` must be a namespace (user or an organization) and a repo name separated by a `/`. For example, if your HF username is `johndoe` and you want to create a repository called `test`, the repo_id should be johndoe/test')
        BaseModelUpload._validate_upload_parameters(repo_id, model_path, repo_type, private, commit_message, commit_description)
        trained_model_artifacts_path = os.path.join(model_path, 'model', 'model_weights')
        files = set(os.listdir(trained_model_artifacts_path))
        if 'pytorch_model.bin' not in files and 'adapter_model.bin' not in files:
            raise Exception(f"Can't find model weights at {trained_model_artifacts_path}. Trained model weights should either be saved as `pytorch_model.bin` for regular model training, or have `adapter_model.bin`if using parameter efficient fine-tuning methods like LoRA.")

    def upload(self, repo_id: str, model_path: str, repo_type: Optional[str]=None, private: Optional[bool]=False, commit_message: Optional[str]=None, commit_description: Optional[str]=None, **kwargs) -> bool:
        if False:
            print('Hello World!')
        'Create an empty repo on the HuggingFace Hub and upload trained model artifacts to that repo.\n\n        Args:\n            repo_id (`str`):\n                A namespace (user or an organization) and a repo name separated\n                by a `/`.\n            model_path (`str`):\n                The path of the saved model. This is the top level directory where\n                the models weights as well as other associated training artifacts\n                are saved.\n            repo_type (`str`, *optional*):\n                Set to `"dataset"` or `"space"` if uploading to a dataset or\n                space, `None` or `"model"` if uploading to a model. Default is\n                `None`.\n            private (`bool`, *optional*, defaults to `False`):\n                Whether the model repo should be private.\n            commit_message (`str`, *optional*):\n                The summary / title / first line of the generated commit. Defaults to:\n                `f"Upload {path_in_repo} with huggingface_hub"`\n            commit_description (`str` *optional*):\n                The description of the generated commit\n        '
        HuggingFaceHub._validate_upload_parameters(repo_id, model_path, repo_type, private, commit_message, commit_description)
        self.api.create_repo(repo_id=repo_id, private=private, repo_type=repo_type, exist_ok=True)
        upload_path = self.api.upload_folder(folder_path=os.path.join(model_path, 'model', 'model_weights'), repo_id=repo_id, repo_type=repo_type, commit_message=commit_message, commit_description=commit_description)
        if upload_path:
            logger.info(f'Model uploaded to `{upload_path}` with repository name `{repo_id}`')
            return True
        return False

class Predibase(BaseModelUpload):

    def __init__(self):
        if False:
            print('Hello World!')
        self.pc = None
        self.login()

    def login(self):
        if False:
            while True:
                i = 10
        'Login to Predibase using the token stored in the PREDIBASE_API_TOKEN environment variable and return a\n        PredibaseClient object that can be used to interact with Predibase.'
        from predibase import PredibaseClient
        token = os.environ.get('PREDIBASE_API_TOKEN')
        if token is None:
            raise ValueError('Unable to find PREDIBASE_API_TOKEN environment variable. Please log into Predibase, generate a token and use `export PREDIBASE_API_TOKEN=` to use Predibase')
        try:
            pc = PredibaseClient()
            self.pc = pc
        except Exception as e:
            raise Exception(f'Failed to login to Predibase: {e}')
            return False
        return True

    @staticmethod
    def _validate_upload_parameters(repo_id: str, model_path: str, repo_type: Optional[str]=None, private: Optional[bool]=False, commit_message: Optional[str]=None, commit_description: Optional[str]=None):
        if False:
            return 10
        "Validate parameters before uploading trained model artifacts.\n\n        This method checks if the input parameters meet the necessary requirements before uploading\n        trained model artifacts to the target repository.\n\n        Args:\n            repo_id (str): The ID of the target repository. It must be a less than 256 characters.\n            model_path (str): The path to the directory containing the trained model artifacts. It should contain\n                the model's weights, usually saved under 'model/model_weights'.\n            repo_type (str, optional): The type of the repository. Not used in the base class, but subclasses\n                may use it for specific repository implementations. Defaults to None.\n            private (bool, optional): Whether the repository should be private or not. Not used in the base class,\n                but subclasses may use it for specific repository implementations. Defaults to False.\n            commit_message (str, optional): A message to attach to the commit when uploading to version control\n                systems. Not used in the base class, but subclasses may use it for specific repository\n                implementations. Defaults to None.\n            commit_description (str, optional): A description of the commit when uploading to version control\n                systems. Not used in the base class, but subclasses may use it for specific repository\n                implementations. Defaults to None.\n\n        Raises:\n            ValueError: If the repo_id is too long.\n        "
        if len(repo_id) > 255:
            raise ValueError('`repo_id` must be 255 characters or less.')
        BaseModelUpload._validate_upload_parameters(repo_id, model_path, repo_type, private, commit_message, commit_description)

    def upload(self, repo_id: str, model_path: str, commit_description: Optional[str]=None, dataset_file: Optional[str]=None, dataset_name: Optional[str]=None, **kwargs) -> bool:
        if False:
            print('Hello World!')
        'Create an empty repo in Predibase and upload trained model artifacts to that repo.\n\n        Args:\n            model_path (`str`):\n                The path of the saved model. This is the top level directory where\n                the models weights as well as other associated training artifacts\n                are saved.\n            repo_name (`str`):\n                A repo name.\n            repo_description (`str` *optional*):\n                The description of the repo.\n            dataset_file (`str` *optional*):\n                The path to the dataset file. Required if `service` is set to\n                `"predibase"` for new model repos.\n            dataset_name (`str` *optional*):\n                The name of the dataset. Used by the `service`\n                `"predibase"`. Falls back to the filename.\n        '
        Predibase._validate_upload_parameters(repo_id, model_path, None, False, '', commit_description)
        try:
            dataset = self.pc.upload_dataset(file_path=dataset_file, name=dataset_name)
        except Exception as e:
            raise RuntimeError('Failed to upload dataset to Predibase') from e
        try:
            repo = self.pc.create_model_repo(name=repo_id, description=commit_description, exists_ok=True)
        except Exception as e:
            raise RuntimeError('Failed to create repo in Predibase') from e
        try:
            self.pc.upload_model(repo=repo, model_path=model_path, dataset=dataset)
        except Exception as e:
            raise RuntimeError('Failed to upload model to Predibase') from e
        logger.info(f'Model uploaded to Predibase with repository name `{repo_id}`')
        return True