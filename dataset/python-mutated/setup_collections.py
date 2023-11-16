from __future__ import annotations
ANSIBLE_METADATA = {'metadata_version': '1.1', 'status': ['preview'], 'supported_by': 'community'}
DOCUMENTATION = "\n---\nmodule: setup_collections\nshort_description: Set up test collections based on the input\ndescription:\n- Builds and publishes a whole bunch of collections used for testing in bulk.\noptions:\n  server:\n    description:\n    - The Galaxy server to upload the collections to.\n    required: yes\n    type: str\n  token:\n    description:\n    - The token used to authenticate with the Galaxy server.\n    required: yes\n    type: str\n  collections:\n    description:\n    - A list of collection details to use for the build.\n    required: yes\n    type: list\n    elements: dict\n    options:\n      namespace:\n        description:\n        - The namespace of the collection.\n        required: yes\n        type: str\n      name:\n        description:\n        - The name of the collection.\n        required: yes\n        type: str\n      version:\n        description:\n        - The version of the collection.\n        type: str\n        default: '1.0.0'\n      dependencies:\n        description:\n        - The dependencies of the collection.\n        type: dict\n        default: '{}'\nauthor:\n- Jordan Borean (@jborean93)\n"
EXAMPLES = '\n- name: Build test collections\n  setup_collections:\n    path: ~/ansible/collections/ansible_collections\n    collections:\n    - namespace: namespace1\n      name: name1\n      version: 0.0.1\n    - namespace: namespace1\n      name: name1\n      version: 0.0.2\n'
RETURN = '\n#\n'
import datetime
import os
import subprocess
import tarfile
import tempfile
import yaml
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes
from functools import partial
from multiprocessing import dummy as threading
from multiprocessing import TimeoutError
COLLECTIONS_BUILD_AND_PUBLISH_TIMEOUT = 180

def publish_collection(module, collection):
    if False:
        while True:
            i = 10
    namespace = collection['namespace']
    name = collection['name']
    version = collection['version']
    dependencies = collection['dependencies']
    use_symlink = collection['use_symlink']
    result = {}
    collection_dir = os.path.join(module.tmpdir, '%s-%s-%s' % (namespace, name, version))
    b_collection_dir = to_bytes(collection_dir, errors='surrogate_or_strict')
    os.mkdir(b_collection_dir)
    os.mkdir(os.path.join(b_collection_dir, b'meta'))
    with open(os.path.join(b_collection_dir, b'README.md'), mode='wb') as fd:
        fd.write(b'Collection readme')
    galaxy_meta = {'namespace': namespace, 'name': name, 'version': version, 'readme': 'README.md', 'authors': ['Collection author <name@email.com'], 'dependencies': dependencies, 'license': ['GPL-3.0-or-later'], 'repository': 'https://ansible.com/'}
    with open(os.path.join(b_collection_dir, b'galaxy.yml'), mode='wb') as fd:
        fd.write(to_bytes(yaml.safe_dump(galaxy_meta), errors='surrogate_or_strict'))
    with open(os.path.join(b_collection_dir, b'meta/runtime.yml'), mode='wb') as fd:
        fd.write(b'requires_ansible: ">=1.0.0"')
    with tempfile.NamedTemporaryFile(mode='wb') as temp_fd:
        temp_fd.write(b'data')
        if use_symlink:
            os.mkdir(os.path.join(b_collection_dir, b'docs'))
            os.mkdir(os.path.join(b_collection_dir, b'plugins'))
            b_target_file = b'RE\xc3\x85DM\xc3\x88.md'
            with open(os.path.join(b_collection_dir, b_target_file), mode='wb') as fd:
                fd.write(b'data')
            os.symlink(b_target_file, os.path.join(b_collection_dir, b_target_file + b'-link'))
            os.symlink(temp_fd.name, os.path.join(b_collection_dir, b_target_file + b'-outside-link'))
            os.symlink(os.path.join(b'..', b_target_file), os.path.join(b_collection_dir, b'docs', b_target_file))
            os.symlink(os.path.join(b_collection_dir, b_target_file), os.path.join(b_collection_dir, b'plugins', b_target_file))
            os.symlink(b'docs', os.path.join(b_collection_dir, b'docs-link'))
        release_filename = '%s-%s-%s.tar.gz' % (namespace, name, version)
        collection_path = os.path.join(collection_dir, release_filename)
        (rc, stdout, stderr) = module.run_command(['ansible-galaxy', 'collection', 'build'], cwd=collection_dir)
        result['build'] = {'rc': rc, 'stdout': stdout, 'stderr': stderr}
        if module.params['signature_dir'] is not None:
            with tarfile.open(collection_path, mode='r') as collection_tar:
                if hasattr(tarfile, 'tar_filter'):
                    collection_tar.extractall(path=os.path.join(collection_dir, '%s-%s-%s' % (namespace, name, version)), filter='tar')
                else:
                    collection_tar.extractall(path=os.path.join(collection_dir, '%s-%s-%s' % (namespace, name, version)))
            manifest_path = os.path.join(collection_dir, '%s-%s-%s' % (namespace, name, version), 'MANIFEST.json')
            signature_path = os.path.join(module.params['signature_dir'], '%s-%s-%s-MANIFEST.json.asc' % (namespace, name, version))
            sign_manifest(signature_path, manifest_path, module, result)
            with tarfile.open(collection_path, 'w:gz') as tar:
                tar.add(os.path.join(collection_dir, '%s-%s-%s' % (namespace, name, version)), arcname=os.path.sep)
    publish_args = ['ansible-galaxy', 'collection', 'publish', collection_path, '--server', module.params['server']]
    if module.params['token']:
        publish_args.extend(['--token', module.params['token']])
    (rc, stdout, stderr) = module.run_command(publish_args)
    result['publish'] = {'rc': rc, 'stdout': stdout, 'stderr': stderr}
    return result

def sign_manifest(signature_path, manifest_path, module, collection_setup_result):
    if False:
        return 10
    collection_setup_result['gpg_detach_sign'] = {'signature_path': signature_path}
    (status_fd_read, status_fd_write) = os.pipe()
    gpg_cmd = ['gpg', '--batch', '--pinentry-mode', 'loopback', '--yes', '--homedir', module.params['signature_dir'], '--detach-sign', '--armor', '--output', signature_path, manifest_path]
    try:
        p = subprocess.Popen(gpg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=(status_fd_write,), encoding='utf8')
    except (FileNotFoundError, subprocess.SubprocessError) as err:
        collection_setup_result['gpg_detach_sign']['error'] = "Failed during GnuPG verification with command '{gpg_cmd}': {err}".format(gpg_cmd=gpg_cmd, err=err)
    else:
        (stdout, stderr) = p.communicate()
        collection_setup_result['gpg_detach_sign']['stdout'] = stdout
        if stderr:
            error = "Failed during GnuPG verification with command '{gpg_cmd}':\n{stderr}".format(gpg_cmd=gpg_cmd, stderr=stderr)
            collection_setup_result['gpg_detach_sign']['error'] = error
    finally:
        os.close(status_fd_write)

def run_module():
    if False:
        return 10
    module_args = dict(server=dict(type='str', required=True), token=dict(type='str'), collections=dict(type='list', elements='dict', required=True, options=dict(namespace=dict(type='str', required=True), name=dict(type='str', required=True), version=dict(type='str', default='1.0.0'), dependencies=dict(type='dict', default={}), use_symlink=dict(type='bool', default=False))), signature_dir=dict(type='path', default=None))
    module = AnsibleModule(argument_spec=module_args, supports_check_mode=False)
    start = datetime.datetime.now()
    result = dict(changed=True, results=[], start=str(start))
    pool = threading.Pool(4)
    publish_func = partial(publish_collection, module)
    try:
        result['results'] = pool.map_async(publish_func, module.params['collections']).get(timeout=COLLECTIONS_BUILD_AND_PUBLISH_TIMEOUT)
    except TimeoutError as timeout_err:
        module.fail_json('Timed out waiting for collections to be provisioned.')
    failed = bool(sum((r['build']['rc'] + r['publish']['rc'] for r in result['results'])))
    end = datetime.datetime.now()
    delta = end - start
    module.exit_json(failed=failed, end=str(end), delta=str(delta), **result)

def main():
    if False:
        for i in range(10):
            print('nop')
    run_module()
if __name__ == '__main__':
    main()