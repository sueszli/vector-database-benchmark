from __future__ import annotations
DOCUMENTATION = '\n---\nmodule: reset_pulp\nshort_description: Resets pulp back to the initial state\ndescription:\n- See short_description\noptions:\n  pulp_api:\n    description:\n    - The Pulp API endpoint.\n    required: yes\n    type: str\n  galaxy_ng_server:\n    description:\n    - The Galaxy NG API endpoint.\n    required: yes\n    type: str\n  url_username:\n    description:\n    - The username to use when authenticating against Pulp.\n    required: yes\n    type: str\n  url_password:\n    description:\n    - The password to use when authenticating against Pulp.\n    required: yes\n    type: str\n  repositories:\n    description:\n    - A list of pulp repositories to create.\n    - Galaxy NG expects a repository that matches C(GALAXY_API_DEFAULT_DISTRIBUTION_BASE_PATH) in\n      C(/etc/pulp/settings.py) or the default of C(published).\n    required: yes\n    type: list\n    elements: str\n  namespaces:\n    description:\n    - A list of namespaces to create for Galaxy NG.\n    required: yes\n    type: list\n    elements: str\nauthor:\n- Jordan Borean (@jborean93)\n'
EXAMPLES = '\n- name: reset pulp content\n  reset_pulp:\n    pulp_api: http://galaxy:24817\n    galaxy_ng_server: http://galaxy/api/galaxy/\n    url_username: username\n    url_password: password\n    repository: published\n    namespaces:\n    - namespace1\n    - namespace2\n'
RETURN = '\n#\n'
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text

def invoke_api(module, url, method='GET', data=None, status_codes=None):
    if False:
        return 10
    status_codes = status_codes or [200]
    headers = {}
    if data:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(data)
    (resp, info) = fetch_url(module, url, method=method, data=data, headers=headers)
    if info['status'] not in status_codes:
        info['url'] = url
        module.fail_json(**info)
    data = to_text(resp.read())
    if data:
        return json.loads(data)

def delete_galaxy_namespace(namespace, module):
    if False:
        return 10
    ' Deletes the galaxy ng namespace specified. '
    ns_uri = '%sv3/namespaces/%s/' % (module.params['galaxy_ng_server'], namespace)
    invoke_api(module, ns_uri, method='DELETE', status_codes=[204])

def delete_pulp_distribution(distribution, module):
    if False:
        print('Hello World!')
    ' Deletes the pulp distribution at the URI specified. '
    task_info = invoke_api(module, distribution, method='DELETE', status_codes=[202])
    wait_pulp_task(task_info['task'], module)

def delete_pulp_orphans(module):
    if False:
        i = 10
        return i + 15
    ' Deletes any orphaned pulp objects. '
    orphan_uri = module.params['galaxy_ng_server'] + 'pulp/api/v3/orphans/'
    task_info = invoke_api(module, orphan_uri, method='DELETE', status_codes=[202])
    wait_pulp_task(task_info['task'], module)

def delete_pulp_repository(repository, module):
    if False:
        for i in range(10):
            print('nop')
    ' Deletes the pulp repository at the URI specified. '
    task_info = invoke_api(module, repository, method='DELETE', status_codes=[202])
    wait_pulp_task(task_info['task'], module)

def get_galaxy_namespaces(module):
    if False:
        i = 10
        return i + 15
    ' Gets a list of galaxy namespaces. '
    namespace_uri = module.params['galaxy_ng_server'] + 'v3/namespaces/?limit=100&offset=0'
    ns_info = invoke_api(module, namespace_uri)
    return [n['name'] for n in ns_info['data']]

def get_pulp_distributions(module, distribution):
    if False:
        for i in range(10):
            print('nop')
    ' Gets a list of all the pulp distributions. '
    distro_uri = module.params['galaxy_ng_server'] + 'pulp/api/v3/distributions/ansible/ansible/'
    distro_info = invoke_api(module, distro_uri + '?name=' + distribution)
    return [module.params['pulp_api'] + r['pulp_href'] for r in distro_info['results']]

def get_pulp_repositories(module, repository):
    if False:
        for i in range(10):
            print('nop')
    ' Gets a list of all the pulp repositories. '
    repo_uri = module.params['galaxy_ng_server'] + 'pulp/api/v3/repositories/ansible/ansible/'
    repo_info = invoke_api(module, repo_uri + '?name=' + repository)
    return [module.params['pulp_api'] + r['pulp_href'] for r in repo_info['results']]

def get_repo_collections(repository, module):
    if False:
        while True:
            i = 10
    collections_uri = module.params['galaxy_ng_server'] + 'v3/plugin/ansible/content/' + repository + '/collections/index/'
    info = invoke_api(module, collections_uri + '?limit=100&offset=0', status_codes=[200, 500])
    if not info:
        return []
    return [module.params['pulp_api'] + c['href'] for c in info['data']]

def delete_repo_collection(collection, module):
    if False:
        while True:
            i = 10
    task_info = invoke_api(module, collection, method='DELETE', status_codes=[202])
    wait_pulp_task(task_info['task'], module)

def new_galaxy_namespace(name, module):
    if False:
        print('Hello World!')
    ' Creates a new namespace in Galaxy NG. '
    ns_uri = module.params['galaxy_ng_server'] + 'v3/namespaces/ '
    data = {'name': name, 'groups': []}
    ns_info = invoke_api(module, ns_uri, method='POST', data=data, status_codes=[201])
    return ns_info['id']

def new_pulp_repository(name, module):
    if False:
        for i in range(10):
            print('nop')
    ' Creates a new pulp repository. '
    repo_uri = module.params['galaxy_ng_server'] + 'pulp/api/v3/repositories/ansible/ansible/'
    data = {'name': name, 'retain_repo_versions': '1024'}
    repo_info = invoke_api(module, repo_uri, method='POST', data=data, status_codes=[201])
    return repo_info['pulp_href']

def new_pulp_distribution(name, base_path, repository, module):
    if False:
        i = 10
        return i + 15
    ' Creates a new pulp distribution for a repository. '
    distro_uri = module.params['galaxy_ng_server'] + 'pulp/api/v3/distributions/ansible/ansible/'
    data = {'name': name, 'base_path': base_path, 'repository': repository}
    task_info = invoke_api(module, distro_uri, method='POST', data=data, status_codes=[202])
    task_info = wait_pulp_task(task_info['task'], module)
    return module.params['pulp_api'] + task_info['created_resources'][0]

def wait_pulp_task(task, module):
    if False:
        for i in range(10):
            print('nop')
    ' Waits for a pulp import task to finish. '
    while True:
        task_info = invoke_api(module, module.params['pulp_api'] + task)
        if task_info['finished_at'] is not None:
            break
    return task_info

def main():
    if False:
        while True:
            i = 10
    module_args = dict(pulp_api=dict(type='str', required=True), galaxy_ng_server=dict(type='str', required=True), url_username=dict(type='str', required=True), url_password=dict(type='str', required=True, no_log=True), repositories=dict(type='list', elements='str', required=True), namespaces=dict(type='list', elements='str', required=True))
    module = AnsibleModule(argument_spec=module_args, supports_check_mode=False)
    module.params['force_basic_auth'] = True
    for repository in module.params['repositories']:
        [delete_repo_collection(c, module) for c in get_repo_collections(repository, module)]
    for repository in module.params['repositories']:
        [delete_pulp_distribution(d, module) for d in get_pulp_distributions(module, repository)]
        [delete_pulp_repository(r, module) for r in get_pulp_repositories(module, repository)]
    delete_pulp_orphans(module)
    [delete_galaxy_namespace(n, module) for n in get_galaxy_namespaces(module)]
    for repository in module.params['repositories']:
        repo_href = new_pulp_repository(repository, module)
        new_pulp_distribution(repository, repository, repo_href, module)
    [new_galaxy_namespace(n, module) for n in module.params['namespaces']]
    module.exit_json(changed=True)
if __name__ == '__main__':
    main()