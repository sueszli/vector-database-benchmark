"""
Module for running imgadm command on SmartOS
"""
import logging
import salt.utils.json
import salt.utils.path
import salt.utils.platform
log = logging.getLogger(__name__)
__func_alias__ = {'list_installed': 'list', 'update_installed': 'update', 'import_image': 'import'}
__virtualname__ = 'imgadm'

def __virtual__():
    if False:
        print('Hello World!')
    '\n    Provides imgadm only on SmartOS\n    '
    if salt.utils.platform.is_smartos_globalzone() and salt.utils.path.which('imgadm'):
        return __virtualname__
    return (False, f'{__virtualname__} module can only be loaded on SmartOS compute nodes')

def _exit_status(retcode, stderr=None):
    if False:
        print('Hello World!')
    '\n    Translate exit status of imgadm\n    '
    ret = {0: 'Successful completion.', 1: 'An error occurred.' if not stderr else stderr, 2: 'Usage error.', 3: 'Image not installed.'}[retcode]
    return ret

def _parse_image_meta(image=None, detail=False):
    if False:
        print('Hello World!')
    ret = None
    if image and 'Error' in image:
        ret = image
    elif image and 'manifest' in image and ('name' in image['manifest']):
        name = image['manifest']['name']
        version = image['manifest']['version']
        os = image['manifest']['os']
        description = image['manifest']['description']
        published = image['manifest']['published_at']
        source = image['source']
        if image['manifest']['name'] == 'docker-layer':
            name = None
            docker_repo = None
            docker_tag = None
            for tag in image['manifest']['tags']:
                if tag.startswith('docker:tag:') and image['manifest']['tags'][tag]:
                    docker_tag = tag.split(':')[-1]
                elif tag == 'docker:repo':
                    docker_repo = image['manifest']['tags'][tag]
            if docker_repo and docker_tag:
                name = f'{docker_repo}:{docker_tag}'
                description = 'Docker image imported from {repo}:{tag} on {date}.'.format(repo=docker_repo, tag=docker_tag, date=published)
        if name and detail:
            ret = {'name': name, 'version': version, 'os': os, 'description': description, 'published': published, 'source': source}
        elif name:
            ret = '{name}@{version} [{published}]'.format(name=name, version=version, published=published)
    else:
        log.debug('smartos_image - encountered invalid image payload: %s', image)
        ret = {'Error': 'This looks like an orphaned image, image payload was invalid.'}
    return ret

def _split_docker_uuid(uuid):
    if False:
        return 10
    '\n    Split a smartos docker uuid into repo and tag\n    '
    if uuid:
        uuid = uuid.split(':')
        if len(uuid) == 2:
            tag = uuid[1]
            repo = uuid[0]
            return (repo, tag)
    return (None, None)

def _is_uuid(uuid):
    if False:
        print('Hello World!')
    '\n    Check if uuid is a valid smartos uuid\n\n    Example: e69a0918-055d-11e5-8912-e3ceb6df4cf8\n    '
    if uuid and list((len(x) for x in uuid.split('-'))) == [8, 4, 4, 4, 12]:
        return True
    return False

def _is_docker_uuid(uuid):
    if False:
        return 10
    '\n    Check if uuid is a valid smartos docker uuid\n\n    Example plexinc/pms-docker:plexpass\n    '
    (repo, tag) = _split_docker_uuid(uuid)
    return not (not repo and (not tag))

def version():
    if False:
        print('Hello World!')
    "\n    Return imgadm version\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.version\n    "
    ret = {}
    cmd = 'imgadm --version'
    res = __salt__['cmd.run'](cmd).splitlines()
    ret = res[0].split()
    return ret[-1]

def docker_to_uuid(uuid):
    if False:
        while True:
            i = 10
    '\n    Get the image uuid from an imported docker image\n\n    .. versionadded:: 2019.2.0\n    '
    if _is_uuid(uuid):
        return uuid
    if _is_docker_uuid(uuid):
        images = list_installed(verbose=True)
        for image_uuid in images:
            if 'name' not in images[image_uuid]:
                continue
            if images[image_uuid]['name'] == uuid:
                return image_uuid
    return None

def update_installed(uuid=''):
    if False:
        return 10
    "\n    Gather info on unknown image(s) (locally installed)\n\n    uuid : string\n        optional uuid of image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.update [uuid]\n    "
    cmd = f'imgadm update {uuid}'.rstrip()
    __salt__['cmd.run'](cmd)
    return {}

def avail(search=None, verbose=False):
    if False:
        return 10
    "\n    Return a list of available images\n\n    search : string\n        search keyword\n    verbose : boolean (False)\n        toggle verbose output\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.avail [percona]\n        salt '*' imgadm.avail verbose=True\n    "
    ret = {}
    cmd = 'imgadm avail -j'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode)
        return ret
    for image in salt.utils.json.loads(res['stdout']):
        if image['manifest']['disabled'] or not image['manifest']['public']:
            continue
        if search and search not in image['manifest']['name']:
            continue
        uuid = image['manifest']['uuid']
        data = _parse_image_meta(image, verbose)
        if data:
            ret[uuid] = data
    return ret

def list_installed(verbose=False):
    if False:
        while True:
            i = 10
    "\n    Return a list of installed images\n\n    verbose : boolean (False)\n        toggle verbose output\n\n    .. versionchanged:: 2019.2.0\n\n        Docker images are now also listed\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.list\n        salt '*' imgadm.list docker=True\n        salt '*' imgadm.list verbose=True\n    "
    ret = {}
    cmd = 'imgadm list -j'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode)
        return ret
    for image in salt.utils.json.loads(res['stdout']):
        uuid = image['manifest']['uuid']
        data = _parse_image_meta(image, verbose)
        if data:
            ret[uuid] = data
    return ret

def show(uuid):
    if False:
        for i in range(10):
            print('nop')
    "\n    Show manifest of a given image\n\n    uuid : string\n        uuid of image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.show e42f8c84-bbea-11e2-b920-078fab2aab1f\n        salt '*' imgadm.show plexinc/pms-docker:plexpass\n    "
    ret = {}
    if _is_uuid(uuid) or _is_docker_uuid(uuid):
        cmd = f'imgadm show {uuid}'
        res = __salt__['cmd.run_all'](cmd, python_shell=False)
        retcode = res['retcode']
        if retcode != 0:
            ret['Error'] = _exit_status(retcode, res['stderr'])
        else:
            ret = salt.utils.json.loads(res['stdout'])
    else:
        ret['Error'] = f'{uuid} is not a valid uuid.'
    return ret

def get(uuid):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return info on an installed image\n\n    uuid : string\n        uuid of image\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.get e42f8c84-bbea-11e2-b920-078fab2aab1f\n        salt '*' imgadm.get plexinc/pms-docker:plexpass\n    "
    ret = {}
    if _is_docker_uuid(uuid):
        uuid = docker_to_uuid(uuid)
    if _is_uuid(uuid):
        cmd = f'imgadm get {uuid}'
        res = __salt__['cmd.run_all'](cmd, python_shell=False)
        retcode = res['retcode']
        if retcode != 0:
            ret['Error'] = _exit_status(retcode, res['stderr'])
        else:
            ret = salt.utils.json.loads(res['stdout'])
    else:
        ret['Error'] = f'{uuid} is not a valid uuid.'
    return ret

def import_image(uuid, verbose=False):
    if False:
        i = 10
        return i + 15
    "\n    Import an image from the repository\n\n    uuid : string\n        uuid to import\n    verbose : boolean (False)\n        toggle verbose output\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.import e42f8c84-bbea-11e2-b920-078fab2aab1f [verbose=True]\n    "
    ret = {}
    cmd = f'imgadm import {uuid}'
    res = __salt__['cmd.run_all'](cmd, python_shell=False)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode)
        return ret
    uuid = docker_to_uuid(uuid)
    data = _parse_image_meta(get(uuid), verbose)
    return {uuid: data}

def delete(uuid):
    if False:
        for i in range(10):
            print('nop')
    "\n    Remove an installed image\n\n    uuid : string\n        Specifies uuid to import\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.delete e42f8c84-bbea-11e2-b920-078fab2aab1f\n    "
    ret = {}
    cmd = f'imgadm delete {uuid}'
    res = __salt__['cmd.run_all'](cmd, python_shell=False)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode)
        return ret
    result = []
    for image in res['stdout'].splitlines():
        image = [var for var in image.split(' ') if var]
        result.append(image[2])
    return result

def vacuum(verbose=False):
    if False:
        while True:
            i = 10
    "\n    Remove unused images\n\n    verbose : boolean (False)\n        toggle verbose output\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.vacuum [verbose=True]\n    "
    ret = {}
    cmd = 'imgadm vacuum -f'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode)
        return ret
    result = {}
    for image in res['stdout'].splitlines():
        image = [var for var in image.split(' ') if var]
        result[image[2]] = {'name': image[3][1:image[3].index('@')], 'version': image[3][image[3].index('@') + 1:-1]}
    if verbose:
        return result
    else:
        return list(result.keys())

def sources(verbose=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return a list of available sources\n\n    verbose : boolean (False)\n        toggle verbose output\n\n    .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.sources\n    "
    ret = {}
    cmd = 'imgadm sources -j'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode)
        return ret
    for src in salt.utils.json.loads(res['stdout']):
        ret[src['url']] = src
        del src['url']
    if not verbose:
        ret = list(ret)
    return ret

def source_delete(source):
    if False:
        return 10
    "\n    Delete a source\n\n    source : string\n        source url to delete\n\n    .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.source_delete https://updates.joyent.com\n    "
    ret = {}
    cmd = f'imgadm sources -d {source}'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode, res['stderr'])
        return ret
    return sources(False)

def source_add(source, source_type='imgapi'):
    if False:
        print('Hello World!')
    "\n    Add a new source\n\n    source : string\n        source url to add\n    source_trype : string (imgapi)\n        source type, either imgapi or docker\n\n    .. versionadded:: 2019.2.0\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' imgadm.source_add https://updates.joyent.com\n        salt '*' imgadm.source_add https://docker.io docker\n    "
    ret = {}
    if source_type not in ['imgapi', 'docker']:
        log.warning('Possible unsupported imgage source type specified!')
    cmd = f'imgadm sources -a {source} -t {source_type}'
    res = __salt__['cmd.run_all'](cmd)
    retcode = res['retcode']
    if retcode != 0:
        ret['Error'] = _exit_status(retcode, res['stderr'])
        return ret
    return sources(False)