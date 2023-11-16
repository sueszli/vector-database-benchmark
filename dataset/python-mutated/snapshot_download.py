import os
import re
import tempfile
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Dict, List, Optional, Union
from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
from modelscope.utils.logger import get_logger
from .constants import FILE_HASH, MODELSCOPE_DOWNLOAD_PARALLELS, MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB
from .file_download import get_file_download_url, http_get_file, parallel_download
from .utils.caching import ModelFileSystemCache
from .utils.utils import file_integrity_validation, get_cache_dir, model_id_to_group_owner_name
logger = get_logger()

def snapshot_download(model_id: str, revision: Optional[str]=DEFAULT_MODEL_REVISION, cache_dir: Union[str, Path, None]=None, user_agent: Optional[Union[Dict, str]]=None, local_files_only: Optional[bool]=False, cookies: Optional[CookieJar]=None, ignore_file_pattern: List=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Download all files of a repo.\n    Downloads a whole snapshot of a repo's files at the specified revision. This\n    is useful when you want all files from a repo, because you don't know which\n    ones you will need a priori. All files are nested inside a folder in order\n    to keep their actual filename relative to that folder.\n\n    An alternative would be to just clone a repo but this would require that the\n    user always has git and git-lfs installed, and properly configured.\n\n    Args:\n        model_id (str): A user or an organization name and a repo name separated by a `/`.\n        revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a\n            commit hash. NOTE: currently only branch and tag name is supported\n        cache_dir (str, Path, optional): Path to the folder where cached files are stored.\n        user_agent (str, dict, optional): The user-agent info in the form of a dictionary or a string.\n        local_files_only (bool, optional): If `True`, avoid downloading the file and return the path to the\n            local cached file if it exists.\n        cookies (CookieJar, optional): The cookie of the request, default None.\n        ignore_file_pattern (`str` or `List`, *optional*, default to `None`):\n            Any file pattern to be ignored in downloading, like exact file names or file extensions.\n    Raises:\n        ValueError: the value details.\n\n    Returns:\n        str: Local folder path (string) of repo snapshot\n\n    Note:\n        Raises the following errors:\n        - [`EnvironmentError`](https://docs.python.org/3/library/exceptions.html#EnvironmentError)\n        if `use_auth_token=True` and the token cannot be found.\n        - [`OSError`](https://docs.python.org/3/library/exceptions.html#OSError) if\n        ETag cannot be determined.\n        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)\n        if some parameter value is invalid\n    "
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    temporary_cache_dir = os.path.join(cache_dir, 'temp')
    os.makedirs(temporary_cache_dir, exist_ok=True)
    (group_or_owner, name) = model_id_to_group_owner_name(model_id)
    cache = ModelFileSystemCache(cache_dir, group_or_owner, name)
    if local_files_only:
        if len(cache.cached_files) == 0:
            raise ValueError("Cannot find the requested files in the cached path and outgoing traffic has been disabled. To enable model look-ups and downloads online, set 'local_files_only' to False.")
        logger.warning('We can not confirm the cached file is for revision: %s' % revision)
        return cache.get_root_location()
    else:
        headers = {'user-agent': ModelScopeConfig.get_user_agent(user_agent=user_agent)}
        _api = HubApi()
        if cookies is None:
            cookies = ModelScopeConfig.get_cookies()
        revision = _api.get_valid_revision(model_id, revision=revision, cookies=cookies)
        snapshot_header = headers if 'CI_TEST' in os.environ else {**headers, **{'Snapshot': 'True'}}
        model_files = _api.get_model_files(model_id=model_id, revision=revision, recursive=True, use_cookies=False if cookies is None else cookies, headers=snapshot_header)
        if ignore_file_pattern is None:
            ignore_file_pattern = []
        if isinstance(ignore_file_pattern, str):
            ignore_file_pattern = [ignore_file_pattern]
        with tempfile.TemporaryDirectory(dir=temporary_cache_dir) as temp_cache_dir:
            for model_file in model_files:
                if model_file['Type'] == 'tree' or any([re.search(pattern, model_file['Name']) is not None for pattern in ignore_file_pattern]):
                    continue
                if cache.exists(model_file):
                    file_name = os.path.basename(model_file['Name'])
                    logger.debug(f'File {file_name} already in cache, skip downloading!')
                    continue
                url = get_file_download_url(model_id=model_id, file_path=model_file['Path'], revision=revision)
                if MODELSCOPE_PARALLEL_DOWNLOAD_THRESHOLD_MB * 1000 * 1000 < model_file['Size'] and MODELSCOPE_DOWNLOAD_PARALLELS > 1:
                    parallel_download(url, temp_cache_dir, model_file['Name'], headers=headers, cookies=None if cookies is None else cookies.get_dict(), file_size=model_file['Size'])
                else:
                    http_get_file(url, temp_cache_dir, model_file['Name'], headers=headers, cookies=cookies)
                temp_file = os.path.join(temp_cache_dir, model_file['Name'])
                if FILE_HASH in model_file:
                    file_integrity_validation(temp_file, model_file[FILE_HASH])
                cache.put_file(model_file, temp_file)
        return os.path.join(cache.get_root_location())