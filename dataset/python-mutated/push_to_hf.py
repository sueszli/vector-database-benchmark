"""
Utilities for pushing models to the Hugging Face Hub ([hf.co](https://hf.co/)).
"""
import logging
import shutil
import tarfile
import tempfile
import zipfile
from os import PathLike
from pathlib import Path
from typing import Optional, Union
from huggingface_hub import HfApi, HfFolder, Repository
from allennlp.common.file_utils import cached_path
logger = logging.getLogger(__name__)
README_TEMPLATE = '---\ntags:\n- allennlp\n---\n\n# TODO: Fill this model card\n'

def _create_model_card(repo_dir: Path):
    if False:
        return 10
    'Creates a model card for the repository.\n\n    TODO: Add metrics to model-index\n    TODO: Use information from common model cards\n    '
    readme_path = repo_dir / 'README.md'
    prev_readme = ''
    if readme_path.exists():
        with readme_path.open('r', encoding='utf8') as f:
            prev_readme = f.read()
    with readme_path.open('w', encoding='utf-8') as f:
        f.write(README_TEMPLATE)
        f.write(prev_readme)
_ALLOWLIST_PATHS = ['vocabulary', 'config.json', 'weights.th', 'best.th', 'metrics.json', 'log']

def _copy_allowed_file(filepath: Path, dst_directory: Path):
    if False:
        i = 10
        return i + 15
    '\n    Copies files from allowlist to a directory, overriding existing\n    files or directories if any.\n    '
    if filepath.name not in _ALLOWLIST_PATHS:
        return
    dst = dst_directory / filepath.name
    if dst.is_dir():
        shutil.rmtree(str(dst))
    elif dst.is_file():
        dst.unlink()
    if filepath.is_dir():
        shutil.copytree(filepath, dst)
    elif filepath.is_file():
        if filepath.name in ['best.th', 'weights.th']:
            dst = dst_directory / 'weights.th'
        shutil.copy(str(filepath), str(dst))

def push_to_hf(repo_name: str, serialization_dir: Optional[Union[str, PathLike]]=None, archive_path: Optional[Union[str, PathLike]]=None, organization: Optional[str]=None, commit_message: str='Update repository', local_repo_path: Union[str, PathLike]='hub', use_auth_token: Union[bool, str]=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Pushes model and related files to the Hugging Face Hub ([hf.co](https://hf.co/))\n\n    # Parameters\n\n    repo_name: `str`\n        Name of the repository in the Hugging Face Hub.\n\n    serialization_dir : `Union[str, PathLike]`, optional (default = `None`)\n        Full path to a directory with the serialized model.\n\n    archive_path : `Union[str, PathLike]`, optional (default = `None`)\n        Full path to the zipped model (e.g. model/model.tar.gz). Use `serialization_dir` if possible.\n\n    organization : `Optional[str]`, optional (default = `None`)\n        Name of organization to which the model should be uploaded.\n\n    commit_message: `str` (default=`Update repository`)\n        Commit message to use for the push.\n\n    local_repo_path : `Union[str, Path]`, optional (default=`hub`)\n        Local directory where the repository will be saved.\n\n    use_auth_token (``str`` or ``bool``, `optional`, defaults ``True``):\n        huggingface_token can be extract from ``HfApi().login(username, password)`` and is used to authenticate\n        against the Hugging Face Hub (useful from Google Colab for instance). It's automatically retrieved\n        if you've done `huggingface-cli login` before.\n    "
    if serialization_dir is not None:
        working_dir = Path(serialization_dir)
        if archive_path is not None:
            raise ValueError('serialization_dir and archive_path are mutually exclusive, please just use one.')
        if not working_dir.exists() or not working_dir.is_dir():
            raise ValueError(f"Can't find path: {serialization_dir}, please pointto a directory with the serialized model.")
    elif archive_path is not None:
        working_dir = Path(archive_path)
        if not working_dir.exists() or (not zipfile.is_zipfile(working_dir) and (not tarfile.is_tarfile(working_dir))):
            raise ValueError(f"Can't find path: {archive_path}, please point to a .tar.gz archiveor to a directory with the serialized model.")
        else:
            logging.info('Using the archive_path is discouraged. Using the serialization_dirwill also upload metrics and TensorBoard traces to the Hugging Face Hub.')
    else:
        raise ValueError('please specify either serialization_dir or archive_path')
    info_msg = f"Preparing repository '{use_auth_token}'"
    if isinstance(use_auth_token, str):
        huggingface_token = use_auth_token
    elif use_auth_token:
        huggingface_token = HfFolder.get_token()
    api = HfApi()
    repo_url = api.create_repo(name=repo_name, token=huggingface_token, organization=organization, private=False, exist_ok=True)
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=use_auth_token)
    repo.git_pull(rebase=True)
    repo.lfs_track(['*.th'])
    info_msg = f"Preparing repository '{repo_name}'"
    if organization is not None:
        info_msg += f' ({organization})'
    logging.info(info_msg)
    if serialization_dir is not None:
        for filename in working_dir.iterdir():
            _copy_allowed_file(Path(filename), repo_local_path)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_dir = Path(cached_path(working_dir, temp_dir, extract_archive=True))
            for filename in extracted_dir.iterdir():
                _copy_allowed_file(Path(filename), repo_local_path)
    _create_model_card(repo_local_path)
    logging.info(f'Pushing repo {repo_name} to the Hugging Face Hub')
    repo.push_to_hub(commit_message=commit_message)
    logging.info(f'View your model in {repo_url}')
    return repo_url