import contextlib
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import torch
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Dict, Optional, Any
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request
from urllib.parse import urlparse
from torch.serialization import MAP_LOCATION

class _Faketqdm:

    def __init__(self, total=None, disable=False, unit=None, *args, **kwargs):
        if False:
            print('Hello World!')
        self.total = total
        self.disable = disable
        self.n = 0

    def update(self, n):
        if False:
            for i in range(10):
                print('nop')
        if self.disable:
            return
        self.n += n
        if self.total is None:
            sys.stderr.write(f'\r{self.n:.1f} bytes')
        else:
            sys.stderr.write(f'\r{100 * self.n / float(self.total):.1f}%')
        sys.stderr.flush()

    def set_description(self, *args, **kwargs):
        if False:
            return 10
        pass

    def write(self, s):
        if False:
            while True:
                i = 10
        sys.stderr.write(f'{s}\n')

    def close(self):
        if False:
            while True:
                i = 10
        self.disable = True

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        if self.disable:
            return
        sys.stderr.write('\n')
try:
    from tqdm import tqdm
except ImportError:
    tqdm = _Faketqdm
__all__ = ['download_url_to_file', 'get_dir', 'help', 'list', 'load', 'load_state_dict_from_url', 'set_dir']
HASH_REGEX = re.compile('-([a-f0-9]*)\\.')
_TRUSTED_REPO_OWNERS = ('facebookresearch', 'facebookincubator', 'pytorch', 'fairinternal')
ENV_GITHUB_TOKEN = 'GITHUB_TOKEN'
ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
VAR_DEPENDENCY = 'dependencies'
MODULE_HUBCONF = 'hubconf.py'
READ_DATA_CHUNK = 8192
_hub_dir = None

@contextlib.contextmanager
def _add_to_sys_path(path):
    if False:
        return 10
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)

def _import_module(name, path):
    if False:
        print('Hello World!')
    import importlib.util
    from importlib.abc import Loader
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert isinstance(spec.loader, Loader)
    spec.loader.exec_module(module)
    return module

def _remove_if_exists(path):
    if False:
        for i in range(10):
            print('nop')
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)

def _git_archive_link(repo_owner, repo_name, ref):
    if False:
        return 10
    return f'https://github.com/{repo_owner}/{repo_name}/zipball/{ref}'

def _load_attr_from_module(module, func_name):
    if False:
        i = 10
        return i + 15
    if func_name not in dir(module):
        return None
    return getattr(module, func_name)

def _get_torch_home():
    if False:
        return 10
    torch_home = os.path.expanduser(os.getenv(ENV_TORCH_HOME, os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home

def _parse_repo_info(github):
    if False:
        return 10
    if ':' in github:
        (repo_info, ref) = github.split(':')
    else:
        (repo_info, ref) = (github, None)
    (repo_owner, repo_name) = repo_info.split('/')
    if ref is None:
        try:
            with urlopen(f'https://github.com/{repo_owner}/{repo_name}/tree/main/'):
                ref = 'main'
        except HTTPError as e:
            if e.code == 404:
                ref = 'master'
            else:
                raise
        except URLError as e:
            for possible_ref in ('main', 'master'):
                if os.path.exists(f'{get_dir()}/{repo_owner}_{repo_name}_{possible_ref}'):
                    ref = possible_ref
                    break
            if ref is None:
                raise RuntimeError(f'It looks like there is no internet connection and the repo could not be found in the cache ({get_dir()})') from e
    return (repo_owner, repo_name, ref)

def _read_url(url):
    if False:
        print('Hello World!')
    with urlopen(url) as r:
        return r.read().decode(r.headers.get_content_charset('utf-8'))

def _validate_not_a_forked_repo(repo_owner, repo_name, ref):
    if False:
        i = 10
        return i + 15
    headers = {'Accept': 'application/vnd.github.v3+json'}
    token = os.environ.get(ENV_GITHUB_TOKEN)
    if token is not None:
        headers['Authorization'] = f'token {token}'
    for url_prefix in (f'https://api.github.com/repos/{repo_owner}/{repo_name}/branches', f'https://api.github.com/repos/{repo_owner}/{repo_name}/tags'):
        page = 0
        while True:
            page += 1
            url = f'{url_prefix}?per_page=100&page={page}'
            response = json.loads(_read_url(Request(url, headers=headers)))
            if not response:
                break
            for br in response:
                if br['name'] == ref or br['commit']['sha'].startswith(ref):
                    return
    raise ValueError(f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. If it's a commit from a forked repo, please call hub.load() with forked repo directly.")

def _get_cache_or_reload(github, force_reload, trust_repo, calling_fn, verbose=True, skip_validation=False):
    if False:
        for i in range(10):
            print('nop')
    hub_dir = get_dir()
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)
    (repo_owner, repo_name, ref) = _parse_repo_info(github)
    normalized_br = ref.replace('/', '_')
    owner_name_branch = '_'.join([repo_owner, repo_name, normalized_br])
    repo_dir = os.path.join(hub_dir, owner_name_branch)
    _check_repo_is_trusted(repo_owner, repo_name, owner_name_branch, trust_repo=trust_repo, calling_fn=calling_fn)
    use_cache = not force_reload and os.path.exists(repo_dir)
    if use_cache:
        if verbose:
            sys.stderr.write(f'Using cache found in {repo_dir}\n')
    else:
        if not skip_validation:
            _validate_not_a_forked_repo(repo_owner, repo_name, ref)
        cached_file = os.path.join(hub_dir, normalized_br + '.zip')
        _remove_if_exists(cached_file)
        try:
            url = _git_archive_link(repo_owner, repo_name, ref)
            sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
            download_url_to_file(url, cached_file, progress=False)
        except HTTPError as err:
            if err.code == 300:
                warnings.warn(f"The ref {ref} is ambiguous. Perhaps it is both a tag and a branch in the repo? Torchhub will now assume that it's a branch. You can disambiguate tags and branches by explicitly passing refs/heads/branch_name or refs/tags/tag_name as the ref. That might require using skip_validation=True.")
                disambiguated_branch_ref = f'refs/heads/{ref}'
                url = _git_archive_link(repo_owner, repo_name, ref=disambiguated_branch_ref)
                download_url_to_file(url, cached_file, progress=False)
            else:
                raise
        with zipfile.ZipFile(cached_file) as cached_zipfile:
            extraced_repo_name = cached_zipfile.infolist()[0].filename
            extracted_repo = os.path.join(hub_dir, extraced_repo_name)
            _remove_if_exists(extracted_repo)
            cached_zipfile.extractall(hub_dir)
        _remove_if_exists(cached_file)
        _remove_if_exists(repo_dir)
        shutil.move(extracted_repo, repo_dir)
    return repo_dir

def _check_repo_is_trusted(repo_owner, repo_name, owner_name_branch, trust_repo, calling_fn='load'):
    if False:
        print('Hello World!')
    hub_dir = get_dir()
    filepath = os.path.join(hub_dir, 'trusted_list')
    if not os.path.exists(filepath):
        Path(filepath).touch()
    with open(filepath) as file:
        trusted_repos = tuple((line.strip() for line in file))
    trusted_repos_legacy = next(os.walk(hub_dir))[1]
    owner_name = '_'.join([repo_owner, repo_name])
    is_trusted = owner_name in trusted_repos or owner_name_branch in trusted_repos_legacy or repo_owner in _TRUSTED_REPO_OWNERS
    if trust_repo is None:
        if not is_trusted:
            warnings.warn(f"You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to {{calling_fn}}(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or {calling_fn}(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use {calling_fn}(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour")
        return
    if trust_repo is False or (trust_repo == 'check' and (not is_trusted)):
        response = input(f'The repository {owner_name} does not belong to the list of trusted repositories and as such cannot be downloaded. Do you trust this repository and wish to add it to the trusted list of repositories (y/N)?')
        if response.lower() in ('y', 'yes'):
            if is_trusted:
                print('The repository is already trusted.')
        elif response.lower() in ('n', 'no', ''):
            raise Exception('Untrusted repository.')
        else:
            raise ValueError(f'Unrecognized response {response}.')
    if not is_trusted:
        with open(filepath, 'a') as file:
            file.write(owner_name + '\n')

def _check_module_exists(name):
    if False:
        for i in range(10):
            print('nop')
    import importlib.util
    return importlib.util.find_spec(name) is not None

def _check_dependencies(m):
    if False:
        print('Hello World!')
    dependencies = _load_attr_from_module(m, VAR_DEPENDENCY)
    if dependencies is not None:
        missing_deps = [pkg for pkg in dependencies if not _check_module_exists(pkg)]
        if len(missing_deps):
            raise RuntimeError(f"Missing dependencies: {', '.join(missing_deps)}")

def _load_entry_from_hubconf(m, model):
    if False:
        while True:
            i = 10
    if not isinstance(model, str):
        raise ValueError('Invalid input: model should be a string of function name')
    _check_dependencies(m)
    func = _load_attr_from_module(m, model)
    if func is None or not callable(func):
        raise RuntimeError(f'Cannot find callable {model} in hubconf')
    return func

def get_dir():
    if False:
        return 10
    '\n    Get the Torch Hub cache directory used for storing downloaded models & weights.\n\n    If :func:`~torch.hub.set_dir` is not called, default path is ``$TORCH_HOME/hub`` where\n    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.\n    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux\n    filesystem layout, with a default value ``~/.cache`` if the environment\n    variable is not set.\n    '
    if os.getenv('TORCH_HUB'):
        warnings.warn('TORCH_HUB is deprecated, please use env TORCH_HOME instead')
    if _hub_dir is not None:
        return _hub_dir
    return os.path.join(_get_torch_home(), 'hub')

def set_dir(d):
    if False:
        for i in range(10):
            print('nop')
    '\n    Optionally set the Torch Hub directory used to save downloaded models & weights.\n\n    Args:\n        d (str): path to a local folder to save downloaded models & weights.\n    '
    global _hub_dir
    _hub_dir = os.path.expanduser(d)

def list(github, force_reload=False, skip_validation=False, trust_repo=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    List all callable entrypoints available in the repo specified by ``github``.\n\n    Args:\n        github (str): a string with format "repo_owner/repo_name[:ref]" with an optional\n            ref (tag or branch). If ``ref`` is not specified, the default branch is assumed to be ``main`` if\n            it exists, and otherwise ``master``.\n            Example: \'pytorch/vision:0.10\'\n        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.\n            Default is ``False``.\n        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit\n            specified by the ``github`` argument properly belongs to the repo owner. This will make\n            requests to the GitHub API; you can specify a non-default GitHub token by setting the\n            ``GITHUB_TOKEN`` environment variable. Default is ``False``.\n        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.\n            This parameter was introduced in v1.12 and helps ensuring that users\n            only run code from repos that they trust.\n\n            - If ``False``, a prompt will ask the user whether the repo should\n              be trusted.\n            - If ``True``, the repo will be added to the trusted list and loaded\n              without requiring explicit confirmation.\n            - If ``"check"``, the repo will be checked against the list of\n              trusted repos in the cache. If it is not present in that list, the\n              behaviour will fall back onto the ``trust_repo=False`` option.\n            - If ``None``: this will raise a warning, inviting the user to set\n              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This\n              is only present for backward compatibility and will be removed in\n              v2.0.\n\n            Default is ``None`` and will eventually change to ``"check"`` in v2.0.\n\n    Returns:\n        list: The available callables entrypoint\n\n    Example:\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)\n        >>> entrypoints = torch.hub.list(\'pytorch/vision\', force_reload=True)\n    '
    repo_dir = _get_cache_or_reload(github, force_reload, trust_repo, 'list', verbose=True, skip_validation=skip_validation)
    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
    entrypoints = [f for f in dir(hub_module) if callable(getattr(hub_module, f)) and (not f.startswith('_'))]
    return entrypoints

def help(github, model, force_reload=False, skip_validation=False, trust_repo=None):
    if False:
        return 10
    '\n    Show the docstring of entrypoint ``model``.\n\n    Args:\n        github (str): a string with format <repo_owner/repo_name[:ref]> with an optional\n            ref (a tag or a branch). If ``ref`` is not specified, the default branch is assumed\n            to be ``main`` if it exists, and otherwise ``master``.\n            Example: \'pytorch/vision:0.10\'\n        model (str): a string of entrypoint name defined in repo\'s ``hubconf.py``\n        force_reload (bool, optional): whether to discard the existing cache and force a fresh download.\n            Default is ``False``.\n        skip_validation (bool, optional): if ``False``, torchhub will check that the ref\n            specified by the ``github`` argument properly belongs to the repo owner. This will make\n            requests to the GitHub API; you can specify a non-default GitHub token by setting the\n            ``GITHUB_TOKEN`` environment variable. Default is ``False``.\n        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.\n            This parameter was introduced in v1.12 and helps ensuring that users\n            only run code from repos that they trust.\n\n            - If ``False``, a prompt will ask the user whether the repo should\n              be trusted.\n            - If ``True``, the repo will be added to the trusted list and loaded\n              without requiring explicit confirmation.\n            - If ``"check"``, the repo will be checked against the list of\n              trusted repos in the cache. If it is not present in that list, the\n              behaviour will fall back onto the ``trust_repo=False`` option.\n            - If ``None``: this will raise a warning, inviting the user to set\n              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This\n              is only present for backward compatibility and will be removed in\n              v2.0.\n\n            Default is ``None`` and will eventually change to ``"check"`` in v2.0.\n    Example:\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)\n        >>> print(torch.hub.help(\'pytorch/vision\', \'resnet18\', force_reload=True))\n    '
    repo_dir = _get_cache_or_reload(github, force_reload, trust_repo, 'help', verbose=True, skip_validation=skip_validation)
    with _add_to_sys_path(repo_dir):
        hubconf_path = os.path.join(repo_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
    entry = _load_entry_from_hubconf(hub_module, model)
    return entry.__doc__

def load(repo_or_dir, model, *args, source='github', trust_repo=None, force_reload=False, verbose=True, skip_validation=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Load a model from a github repo or a local directory.\n\n    Note: Loading a model is the typical use case, but this can also be used to\n    for loading other objects such as tokenizers, loss functions, etc.\n\n    If ``source`` is \'github\', ``repo_or_dir`` is expected to be\n    of the form ``repo_owner/repo_name[:ref]`` with an optional\n    ref (a tag or a branch).\n\n    If ``source`` is \'local\', ``repo_or_dir`` is expected to be a\n    path to a local directory.\n\n    Args:\n        repo_or_dir (str): If ``source`` is \'github\',\n            this should correspond to a github repo with format ``repo_owner/repo_name[:ref]`` with\n            an optional ref (tag or branch), for example \'pytorch/vision:0.10\'. If ``ref`` is not specified,\n            the default branch is assumed to be ``main`` if it exists, and otherwise ``master``.\n            If ``source`` is \'local\'  then it should be a path to a local directory.\n        model (str): the name of a callable (entrypoint) defined in the\n            repo/dir\'s ``hubconf.py``.\n        *args (optional): the corresponding args for callable ``model``.\n        source (str, optional): \'github\' or \'local\'. Specifies how\n            ``repo_or_dir`` is to be interpreted. Default is \'github\'.\n        trust_repo (bool, str or None): ``"check"``, ``True``, ``False`` or ``None``.\n            This parameter was introduced in v1.12 and helps ensuring that users\n            only run code from repos that they trust.\n\n            - If ``False``, a prompt will ask the user whether the repo should\n              be trusted.\n            - If ``True``, the repo will be added to the trusted list and loaded\n              without requiring explicit confirmation.\n            - If ``"check"``, the repo will be checked against the list of\n              trusted repos in the cache. If it is not present in that list, the\n              behaviour will fall back onto the ``trust_repo=False`` option.\n            - If ``None``: this will raise a warning, inviting the user to set\n              ``trust_repo`` to either ``False``, ``True`` or ``"check"``. This\n              is only present for backward compatibility and will be removed in\n              v2.0.\n\n            Default is ``None`` and will eventually change to ``"check"`` in v2.0.\n        force_reload (bool, optional): whether to force a fresh download of\n            the github repo unconditionally. Does not have any effect if\n            ``source = \'local\'``. Default is ``False``.\n        verbose (bool, optional): If ``False``, mute messages about hitting\n            local caches. Note that the message about first download cannot be\n            muted. Does not have any effect if ``source = \'local\'``.\n            Default is ``True``.\n        skip_validation (bool, optional): if ``False``, torchhub will check that the branch or commit\n            specified by the ``github`` argument properly belongs to the repo owner. This will make\n            requests to the GitHub API; you can specify a non-default GitHub token by setting the\n            ``GITHUB_TOKEN`` environment variable. Default is ``False``.\n        **kwargs (optional): the corresponding kwargs for callable ``model``.\n\n    Returns:\n        The output of the ``model`` callable when called with the given\n        ``*args`` and ``**kwargs``.\n\n    Example:\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)\n        >>> # from a github repo\n        >>> repo = \'pytorch/vision\'\n        >>> model = torch.hub.load(repo, \'resnet50\', weights=\'ResNet50_Weights.IMAGENET1K_V1\')\n        >>> # from a local directory\n        >>> path = \'/some/local/path/pytorch/vision\'\n        >>> # xdoctest: +SKIP\n        >>> model = torch.hub.load(path, \'resnet50\', weights=\'ResNet50_Weights.DEFAULT\')\n    '
    source = source.lower()
    if source not in ('github', 'local'):
        raise ValueError(f'Unknown source: "{source}". Allowed values: "github" | "local".')
    if source == 'github':
        repo_or_dir = _get_cache_or_reload(repo_or_dir, force_reload, trust_repo, 'load', verbose=verbose, skip_validation=skip_validation)
    model = _load_local(repo_or_dir, model, *args, **kwargs)
    return model

def _load_local(hubconf_dir, model, *args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Load a model from a local directory with a ``hubconf.py``.\n\n    Args:\n        hubconf_dir (str): path to a local directory that contains a\n            ``hubconf.py``.\n        model (str): name of an entrypoint defined in the directory\'s\n            ``hubconf.py``.\n        *args (optional): the corresponding args for callable ``model``.\n        **kwargs (optional): the corresponding kwargs for callable ``model``.\n\n    Returns:\n        a single model with corresponding pretrained weights.\n\n    Example:\n        >>> # xdoctest: +SKIP("stub local path")\n        >>> path = \'/some/local/path/pytorch/vision\'\n        >>> model = _load_local(path, \'resnet50\', weights=\'ResNet50_Weights.IMAGENET1K_V1\')\n    '
    with _add_to_sys_path(hubconf_dir):
        hubconf_path = os.path.join(hubconf_dir, MODULE_HUBCONF)
        hub_module = _import_module(MODULE_HUBCONF, hubconf_path)
        entry = _load_entry_from_hubconf(hub_module, model)
        model = entry(*args, **kwargs)
    return model

def download_url_to_file(url: str, dst: str, hash_prefix: Optional[str]=None, progress: bool=True) -> None:
    if False:
        print('Hello World!')
    "Download object at the given URL to a local path.\n\n    Args:\n        url (str): URL of the object to download\n        dst (str): Full path where object will be saved, e.g. ``/tmp/temporary_file``\n        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with ``hash_prefix``.\n            Default: None\n        progress (bool, optional): whether or not to display a progress bar to stderr\n            Default: True\n\n    Example:\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)\n        >>> # xdoctest: +REQUIRES(POSIX)\n        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')\n\n    "
    file_size = None
    req = Request(url, headers={'User-Agent': 'torch.hub'})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders('Content-Length')
    else:
        content_length = meta.get_all('Content-Length')
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    dst = os.path.expanduser(dst)
    for seq in range(tempfile.TMP_MAX):
        tmp_dst = dst + '.' + uuid.uuid4().hex + '.partial'
        try:
            f = open(tmp_dst, 'w+b')
        except FileExistsError:
            continue
        break
    else:
        raise FileExistsError(errno.EEXIST, 'No usable temporary file name found')
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError(f'invalid hash value (expected "{hash_prefix}", got "{digest}")')
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def _is_legacy_zip_format(filename: str) -> bool:
    if False:
        while True:
            i = 10
    if zipfile.is_zipfile(filename):
        infolist = zipfile.ZipFile(filename).infolist()
        return len(infolist) == 1 and (not infolist[0].is_dir())
    return False

def _legacy_zip_load(filename: str, model_dir: str, map_location: MAP_LOCATION, weights_only: bool) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    warnings.warn('Falling back to the old format < 1.6. This support will be deprecated in favor of default zipfile format introduced in 1.6. Please redo torch.save() to save it in the new zipfile format.')
    with zipfile.ZipFile(filename) as f:
        members = f.infolist()
        if len(members) != 1:
            raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
        f.extractall(model_dir)
        extraced_name = members[0].filename
        extracted_file = os.path.join(model_dir, extraced_name)
    return torch.load(extracted_file, map_location=map_location, weights_only=weights_only)

def load_state_dict_from_url(url: str, model_dir: Optional[str]=None, map_location: MAP_LOCATION=None, progress: bool=True, check_hash: bool=False, file_name: Optional[str]=None, weights_only: bool=False) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    "Loads the Torch serialized object at the given URL.\n\n    If downloaded file is a zip file, it will be automatically\n    decompressed.\n\n    If the object is already present in `model_dir`, it's deserialized and\n    returned.\n    The default value of ``model_dir`` is ``<hub_dir>/checkpoints`` where\n    ``hub_dir`` is the directory returned by :func:`~torch.hub.get_dir`.\n\n    Args:\n        url (str): URL of the object to download\n        model_dir (str, optional): directory in which to save the object\n        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)\n        progress (bool, optional): whether or not to display a progress bar to stderr.\n            Default: True\n        check_hash(bool, optional): If True, the filename part of the URL should follow the naming convention\n            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more\n            digits of the SHA256 hash of the contents of the file. The hash is used to\n            ensure unique names and to verify the contents of the file.\n            Default: False\n        file_name (str, optional): name for the downloaded file. Filename from ``url`` will be used if not set.\n        weights_only(bool, optional): If True, only weights will be loaded and no complex pickled objects.\n            Recommended for untrusted sources. See :func:`~torch.load` for more details.\n\n    Example:\n        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_HUB)\n        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')\n\n    "
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    if _is_legacy_zip_format(cached_file):
        return _legacy_zip_load(cached_file, model_dir, map_location, weights_only)
    return torch.load(cached_file, map_location=map_location, weights_only=weights_only)