import importlib
import os
from typing import Dict, Optional, Union
from packaging import version
from .hub import cached_file
from .import_utils import is_peft_available
ADAPTER_CONFIG_NAME = 'adapter_config.json'
ADAPTER_WEIGHTS_NAME = 'adapter_model.bin'
ADAPTER_SAFE_WEIGHTS_NAME = 'adapter_model.safetensors'

def find_adapter_config_file(model_id: str, cache_dir: Optional[Union[str, os.PathLike]]=None, force_download: bool=False, resume_download: bool=False, proxies: Optional[Dict[str, str]]=None, token: Optional[Union[bool, str]]=None, revision: Optional[str]=None, local_files_only: bool=False, subfolder: str='', _commit_hash: Optional[str]=None) -> Optional[str]:
    if False:
        return 10
    '\n    Simply checks if the model stored on the Hub or locally is an adapter model or not, return the path of the adapter\n    config file if it is, None otherwise.\n\n    Args:\n        model_id (`str`):\n            The identifier of the model to look for, can be either a local path or an id to the repository on the Hub.\n        cache_dir (`str` or `os.PathLike`, *optional*):\n            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard\n            cache should not be used.\n        force_download (`bool`, *optional*, defaults to `False`):\n            Whether or not to force to (re-)download the configuration files and override the cached versions if they\n            exist.\n        resume_download (`bool`, *optional*, defaults to `False`):\n            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.\n        proxies (`Dict[str, str]`, *optional*):\n            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{\'http\': \'foo.bar:3128\',\n            \'http://hostname\': \'foo.bar:4012\'}.` The proxies are used on each request.\n        token (`str` or *bool*, *optional*):\n            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated\n            when running `huggingface-cli login` (stored in `~/.huggingface`).\n        revision (`str`, *optional*, defaults to `"main"`):\n            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a\n            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any\n            identifier allowed by git.\n\n            <Tip>\n\n            To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>".\n\n            </Tip>\n\n        local_files_only (`bool`, *optional*, defaults to `False`):\n            If `True`, will only try to load the tokenizer configuration from local files.\n        subfolder (`str`, *optional*, defaults to `""`):\n            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can\n            specify the folder name here.\n    '
    adapter_cached_filename = None
    if model_id is None:
        return None
    elif os.path.isdir(model_id):
        list_remote_files = os.listdir(model_id)
        if ADAPTER_CONFIG_NAME in list_remote_files:
            adapter_cached_filename = os.path.join(model_id, ADAPTER_CONFIG_NAME)
    else:
        adapter_cached_filename = cached_file(model_id, ADAPTER_CONFIG_NAME, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, token=token, revision=revision, local_files_only=local_files_only, subfolder=subfolder, _commit_hash=_commit_hash, _raise_exceptions_for_missing_entries=False, _raise_exceptions_for_connection_errors=False)
    return adapter_cached_filename

def check_peft_version(min_version: str) -> None:
    if False:
        return 10
    '\n    Checks if the version of PEFT is compatible.\n\n    Args:\n        version (`str`):\n            The version of PEFT to check against.\n    '
    if not is_peft_available():
        raise ValueError('PEFT is not installed. Please install it with `pip install peft`')
    is_peft_version_compatible = version.parse(importlib.metadata.version('peft')) >= version.parse(min_version)
    if not is_peft_version_compatible:
        raise ValueError(f'The version of PEFT you are using is not compatible, please use a version that is greater than {min_version}')