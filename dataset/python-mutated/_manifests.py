"""Collection of function to generate k8s manifests.

Note that pod labels are coupled with how we restart services, which
is done by deleting all pods with the given labels.
"""
import copy
import json
import os
import shlex
import traceback
from typing import Any, Dict, Optional, Tuple
from _orchest.internals import config as _config
from _orchest.internals.utils import get_userdir_relpath
from app import utils
from app.core import pod_scheduling
from app.types import SessionConfig, SessionType
from config import CONFIG_CLASS
logger = utils.get_logger()

def _get_common_volumes_and_volume_mounts(userdir_pvc: str, project_dir: str, pipeline_path: Optional[str]=None, container_project_dir: str=_config.PROJECT_DIR, container_pipeline_path: str=_config.PIPELINE_FILE, container_data_dir: str=_config.DATA_DIR, container_runtime_socket: str=_config.CONTAINER_RUNTIME_SOCKET) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    if False:
        for i in range(10):
            print('nop')
    volumes = {}
    volume_mounts = {}
    relative_project_dir = get_userdir_relpath(project_dir)
    volumes['userdir-pvc'] = {'name': 'userdir-pvc', 'persistentVolumeClaim': {'claimName': userdir_pvc, 'readOnly': False}}
    volumes['container-runtime-socket'] = {'name': 'container-runtime-socket', 'hostPath': {'path': container_runtime_socket, 'type': 'Socket'}}
    volume_mounts['data'] = {'name': 'userdir-pvc', 'mountPath': container_data_dir, 'subPath': 'data'}
    volume_mounts['project-dir'] = {'name': 'userdir-pvc', 'mountPath': container_project_dir, 'subPath': relative_project_dir}
    if pipeline_path is not None:
        relative_pipeline_path = os.path.join(relative_project_dir, pipeline_path)
        volume_mounts['pipeline-file'] = {'name': 'userdir-pvc', 'mountPath': container_pipeline_path, 'subPath': relative_pipeline_path}
    (known_hosts_vol, known_hosts_vol_mount) = utils.get_known_hosts_volume_and_mount()
    volumes['known-hosts'] = known_hosts_vol
    volume_mounts['known-hosts'] = known_hosts_vol_mount
    return (volumes, volume_mounts)

def _get_jupyter_volumes_and_volume_mounts(project_uuid: str, userdir_pvc: str, project_dir: str, project_relative_pipeline_path: str, container_project_dir: str=_config.PROJECT_DIR, container_pipeline_path: str=_config.PIPELINE_FILE, container_data_dir: str=_config.DATA_DIR) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    if False:
        while True:
            i = 10
    (volumes, volume_mounts) = _get_common_volumes_and_volume_mounts(userdir_pvc, project_dir, project_relative_pipeline_path, container_project_dir, container_pipeline_path, container_data_dir)
    source_kernelspecs = os.path.join(_config.KERNELSPECS_PATH.format(project_uuid=project_uuid))
    volume_mounts['kernelspec'] = {'name': 'userdir-pvc', 'mountPath': '/usr/local/share/jupyter/kernels', 'subPath': source_kernelspecs}
    volume_mounts['jupyterlab-lab'] = {'name': 'userdir-pvc', 'mountPath': '/usr/local/share/jupyter/lab', 'subPath': '.orchest/user-configurations/jupyterlab/lab'}
    volume_mounts['jupyterlab-user-settings'] = {'name': 'userdir-pvc', 'mountPath': '/root/.jupyter/lab/user-settings', 'subPath': '.orchest/user-configurations/jupyterlab/user-settings'}
    return (volumes, volume_mounts)

def _get_session_sidecar_rbac_manifests(session_uuid: str, session_config: SessionConfig) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if False:
        print('Hello World!')
    project_uuid = session_config['project_uuid']
    ns = _config.ORCHEST_NAMESPACE
    role_manifest = {'kind': 'Role', 'apiVersion': 'rbac.authorization.k8s.io/v1', 'metadata': {'name': f'session-sidecar-role-{session_uuid}', 'labels': {'app': 'session-sidecar', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}, 'rules': [{'apiGroups': [''], 'resources': ['pods', 'pods/log'], 'verbs': ['get', 'list', 'watch']}]}
    account_manifest = {'apiVersion': 'v1', 'kind': 'ServiceAccount', 'metadata': {'name': f'session-sidecar-sa-{session_uuid}', 'labels': {'app': 'session-sidecar', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}}
    rolebinding_manifest = {'kind': 'RoleBinding', 'apiVersion': 'rbac.authorization.k8s.io/v1', 'metadata': {'name': f'session-sidecar-rolebinding-{session_uuid}', 'labels': {'app': 'session-sidecar', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}, 'subjects': [{'kind': 'ServiceAccount', 'name': f'session-sidecar-sa-{session_uuid}', 'namespace': ns}], 'roleRef': {'kind': 'Role', 'name': f'session-sidecar-role-{session_uuid}', 'apiGroup': 'rbac.authorization.k8s.io'}}
    return (role_manifest, account_manifest, rolebinding_manifest)

def _get_session_sidecar_deployment_manifest(session_uuid: str, session_config: SessionConfig, session_type: SessionType) -> dict:
    if False:
        return 10
    project_uuid = session_config['project_uuid']
    pipeline_uuid = session_config['pipeline_uuid']
    pipeline_path = session_config['pipeline_path']
    project_dir = session_config['project_dir']
    userdir_pvc = session_config['userdir_pvc']
    metadata = {'name': f'session-sidecar-{session_uuid}', 'labels': {'app': 'session-sidecar', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}
    (volumes_dict, volume_mounts_dict) = _get_common_volumes_and_volume_mounts(userdir_pvc, project_dir, pipeline_path)
    deployment_manifest = {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': metadata, 'spec': {'replicas': 1, 'selector': {'matchLabels': metadata['labels']}, 'template': {'metadata': metadata, 'spec': {'securityContext': {'runAsUser': 0, 'runAsGroup': int(os.environ.get('ORCHEST_HOST_GID')), 'fsGroup': int(os.environ.get('ORCHEST_HOST_GID'))}, 'serviceAccount': f'session-sidecar-sa-{session_uuid}', 'serviceAccountName': f'session-sidecar-sa-{session_uuid}', 'dnsConfig': {'options': [{'name': 'timeout', 'value': '10'}, {'name': 'attempts', 'value': '5'}]}, 'volumes': [volumes_dict['userdir-pvc']], 'terminationGracePeriodSeconds': 1, 'containers': [{'name': metadata['name'], 'image': 'orchest/session-sidecar:' + CONFIG_CLASS.ORCHEST_VERSION, 'resources': {'requests': {'cpu': _config.USER_CONTAINERS_CPU_SHARES}}, 'imagePullPolicy': 'IfNotPresent', 'env': _get_orchest_sdk_vars(project_uuid, pipeline_uuid, _config.PIPELINE_FILE, session_uuid, session_type) + [{'name': 'ORCHEST_NAMESPACE', 'value': _config.ORCHEST_NAMESPACE}, {'name': 'ORCHEST_CLUSTER', 'value': _config.ORCHEST_CLUSTER}], 'volumeMounts': [volume_mounts_dict['project-dir'], volume_mounts_dict['pipeline-file']]}]}}}}
    pod_scheduling.modify_session_sidecar_scheduling_behaviour(session_type.value, deployment_manifest)
    return deployment_manifest

def _get_orchest_sdk_vars(project_uuid: str, pipeline_uuid: str, pipeline_file: str, session_uuid: str, session_type: SessionType):
    if False:
        while True:
            i = 10
    return [{'name': k, 'value': v} for (k, v) in {'ORCHEST_PROJECT_UUID': project_uuid, 'ORCHEST_PIPELINE_UUID': pipeline_uuid, 'ORCHEST_PIPELINE_PATH': pipeline_file, 'ORCHEST_SESSION_UUID': session_uuid, 'ORCHEST_SESSION_TYPE': session_type.value}.items()]

def _get_environment_shell_deployment_service_manifest(session_uuid: str, service_name: str, shell_uuid: str, project_uuid: str, pipeline_uuid: str, pipeline_path: str, userdir_pvc: str, project_dir: str, environment_image: str, auth_user_uuid: Optional[str]=None) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if False:
        print('Hello World!')
    '\n    This manifest generation is in core/sessions\n    since the environment shell is part of the\n    session, but has a detached lifecycle in the sense\n    that environment shells can be started/stopped\n    independently from the session start/stop.\n\n    Note that environment shells will always require\n    an interactive session to be available and\n    stopping an interactive session will always stop\n    all associated environment shells.\n    '
    metadata = {'name': service_name, 'labels': {'app': 'environment-shell', 'project_uuid': project_uuid, 'session_uuid': session_uuid, 'shell_uuid': shell_uuid}}
    (volumes_dict, volume_mounts_dict) = _get_common_volumes_and_volume_mounts(userdir_pvc, project_dir, pipeline_path)
    registry_ip = utils.get_registry_ip()
    registry_environment_image = f'{registry_ip}/{environment_image}'
    env = _get_orchest_sdk_vars(project_uuid, pipeline_uuid, _config.PIPELINE_FILE, session_uuid, SessionType.INTERACTIVE)
    args = '/orchest/bootscript.sh shell'
    if auth_user_uuid is not None:
        args = utils.get_auth_user_git_config_setup_script(auth_user_uuid) + args
    deployment_manifest = {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': metadata, 'spec': {'replicas': 1, 'selector': {'matchLabels': metadata['labels']}, 'template': {'metadata': metadata, 'spec': {'terminationGracePeriodSeconds': 5, 'securityContext': {'runAsUser': 0, 'runAsGroup': int(os.environ.get('ORCHEST_HOST_GID')), 'fsGroup': int(os.environ.get('ORCHEST_HOST_GID'))}, 'dnsConfig': {'options': [{'name': 'timeout', 'value': '10'}, {'name': 'attempts', 'value': '5'}]}, 'volumes': [volumes_dict['userdir-pvc'], volumes_dict['container-runtime-socket'], volumes_dict['known-hosts']], 'containers': [{'name': metadata['name'], 'image': registry_environment_image, 'imagePullPolicy': 'IfNotPresent', 'volumeMounts': [volume_mounts_dict['project-dir'], volume_mounts_dict['data'], volume_mounts_dict['pipeline-file'], volume_mounts_dict['known-hosts']], 'command': ['/bin/sh', '-c'], 'args': [args], 'env': env, 'resources': {'requests': {'cpu': _config.USER_CONTAINERS_CPU_SHARES}}, 'startupProbe': {'exec': {'command': ['echo', '1'], 'initialDelaySeconds': 1, 'periodSeconds': 1}}, 'ports': [{'containerPort': 22}]}]}}}}
    pod_scheduling.modify_env_shell_scheduling_behaviour(deployment_manifest)
    service_manifest = {'apiVersion': 'v1', 'kind': 'Service', 'metadata': metadata, 'spec': {'type': 'ClusterIP', 'selector': metadata['labels'], 'ports': [{'port': 22, 'targetPort': 22}]}}
    return (deployment_manifest, service_manifest)

def _get_jupyter_server_deployment_service_manifest(session_uuid: str, session_config: SessionConfig) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if False:
        return 10
    project_uuid = session_config['project_uuid']
    pipeline_path = session_config['pipeline_path']
    project_dir = session_config['project_dir']
    userdir_pvc = session_config['userdir_pvc']
    metadata = {'name': f'jupyter-server-{session_uuid}', 'labels': {'app': 'jupyter-server', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}
    (volumes_dict, volume_mounts_dict) = _get_jupyter_volumes_and_volume_mounts(project_uuid, userdir_pvc, project_dir, pipeline_path)
    volumes = [volumes_dict['userdir-pvc'], volumes_dict['container-runtime-socket'], volumes_dict['known-hosts']]
    volume_mounts = [volume_mounts_dict['project-dir'], volume_mounts_dict['data'], volume_mounts_dict['jupyterlab-lab'], volume_mounts_dict['jupyterlab-user-settings'], volume_mounts_dict['known-hosts']]
    env = [{'name': 'ORCHEST_PROJECT_UUID', 'value': project_uuid}, {'name': 'ORCHEST_PIPELINE_UUID', 'value': session_config['pipeline_uuid']}, {'name': 'ORCHEST_SESSION_PIPELINE_PATH', 'value': session_config['pipeline_path']}, {'name': 'ORCHEST_API_ADDRESS', 'value': CONFIG_CLASS.ORCHEST_API_ADDRESS}, {'name': 'ORCHEST_WEBSERVER_ADDRESS', 'value': CONFIG_CLASS.ORCHEST_WEBSERVER_ADDRESS}, {'name': 'ORCHEST_SESSION_UUID', 'value': session_uuid}]
    args = utils.get_add_ssh_secrets_script() + f"/start.sh --allow-root --port=8888 --no-browser --gateway-url=http://jupyter-eg-{session_uuid}:8888/{metadata['name']} --notebook-dir={_config.PROJECT_DIR} --ServerApp.base_url=/{metadata['name']}"
    if session_config.get('auth_user_uuid') is not None:
        (v, vm) = utils.get_user_ssh_keys_volumes_and_mounts(session_config.get('auth_user_uuid'))
        volumes.extend(v)
        volume_mounts.extend(vm)
        args = utils.get_auth_user_git_config_setup_script(session_config.get('auth_user_uuid')) + args
    deployment_manifest = {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': metadata, 'spec': {'replicas': 1, 'selector': {'matchLabels': metadata['labels']}, 'template': {'metadata': metadata, 'spec': {'terminationGracePeriodSeconds': 5, 'securityContext': {'runAsUser': 0, 'runAsGroup': int(os.environ.get('ORCHEST_HOST_GID')), 'fsGroup': int(os.environ.get('ORCHEST_HOST_GID'))}, 'volumes': volumes, 'containers': [{'name': metadata['name'], 'image': utils.get_jupyter_server_image_to_use(), 'imagePullPolicy': 'IfNotPresent', 'volumeMounts': volume_mounts, 'env': env, 'command': ['/bin/sh', '-c'], 'args': [args], 'resources': {'requests': {'cpu': _config.USER_CONTAINERS_CPU_SHARES}}, 'startupProbe': {'httpGet': {'path': f"/{metadata['name']}/api", 'port': 8888}, 'periodSeconds': 1, 'failureThreshold': 120}, 'ports': [{'containerPort': 8888}]}]}}}}
    pod_scheduling.modify_jupyter_server_scheduling_behaviour(deployment_manifest)
    service_manifest = {'apiVersion': 'v1', 'kind': 'Service', 'metadata': metadata, 'spec': {'type': 'ClusterIP', 'selector': metadata['labels'], 'ports': [{'port': 80, 'targetPort': 8888}]}}
    ingress_rule = {}
    if _config.ORCHEST_FQDN is not None:
        ingress_rule['host'] = _config.ORCHEST_FQDN
    ingress_rule['http'] = {'paths': [{'backend': {'service': {'name': f'jupyter-server-{session_uuid}', 'port': {'number': 80}}}, 'path': f'/jupyter-server-{session_uuid}', 'pathType': 'Prefix'}]}
    ingress_metadata = copy.deepcopy(metadata)
    ingress_metadata['annotations'] = {'nginx.ingress.kubernetes.io/proxy-body-size': '0'}
    ingress_manifest = {'apiVersion': 'networking.k8s.io/v1', 'kind': 'Ingress', 'metadata': ingress_metadata, 'spec': {'ingressClassName': _config.INGRESS_CLASS, 'rules': [ingress_rule]}}
    return (deployment_manifest, service_manifest, ingress_manifest)

def _get_jupyter_enterprise_gateway_rbac_manifests(session_uuid: str, session_config: SessionConfig) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    project_uuid = session_config['project_uuid']
    ns = _config.ORCHEST_NAMESPACE
    role_manifest = {'kind': 'Role', 'apiVersion': 'rbac.authorization.k8s.io/v1', 'metadata': {'name': f'jupyter-eg-role-{session_uuid}', 'labels': {'app': 'jupyter-eg', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}, 'rules': [{'apiGroups': [''], 'resources': ['pods'], 'verbs': ['get', 'list', 'watch', 'update', 'delete', 'patch']}]}
    account_manifest = {'apiVersion': 'v1', 'kind': 'ServiceAccount', 'metadata': {'name': f'jupyter-eg-sa-{session_uuid}', 'labels': {'app': 'jupyter-eg', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}}
    rolebinding_manifest = {'kind': 'RoleBinding', 'apiVersion': 'rbac.authorization.k8s.io/v1', 'metadata': {'name': f'jupyter-eg-rolebinding-{session_uuid}', 'labels': {'app': 'jupyter-eg', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}, 'subjects': [{'kind': 'ServiceAccount', 'name': f'jupyter-eg-sa-{session_uuid}', 'namespace': ns}], 'roleRef': {'kind': 'Role', 'name': f'jupyter-eg-role-{session_uuid}', 'apiGroup': 'rbac.authorization.k8s.io'}}
    return (role_manifest, account_manifest, rolebinding_manifest)

def _get_jupyter_enterprise_gateway_deployment_service_manifest(session_uuid: str, session_config: SessionConfig, session_type: SessionType) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if False:
        return 10
    project_uuid = session_config['project_uuid']
    pipeline_uuid = session_config['pipeline_uuid']
    pipeline_path = session_config['pipeline_path']
    project_dir = session_config['project_dir']
    userdir_pvc = session_config['userdir_pvc']
    metadata = {'name': f'jupyter-eg-{session_uuid}', 'labels': {'app': 'jupyter-eg', 'project_uuid': project_uuid, 'session_uuid': session_uuid}}
    process_env_whitelist = ['ORCHEST_PIPELINE_UUID', 'ORCHEST_PIPELINE_PATH', 'ORCHEST_PROJECT_UUID', 'ORCHEST_PROJECT_DIR', 'ORCHEST_PIPELINE_FILE']
    process_env_whitelist = ','.join(process_env_whitelist)
    environment = {'EG_MIRROR_WORKING_DIRS': 'True', 'EG_LIST_KERNELS': 'True', 'EG_KERNEL_WHITELIST': '[]', 'EG_PROHIBITED_UIDS': '[]', 'EG_UNAUTHORIZED_USERS': '["dummy"]', 'EG_UID_BLACKLIST': '["-1"]', 'EG_ALLOW_ORIGIN': '*', 'EG_BASE_URL': f'/jupyter-server-{session_uuid}', 'EG_KERNEL_LAUNCH_TIMEOUT': '600', 'EG_ENV_PROCESS_WHITELIST': process_env_whitelist, 'EG_LOG_LEVEL': '10', 'EG_NAMESPACE': _config.ORCHEST_NAMESPACE, 'EG_SHARED_NAMESPACE': 'True', 'ORCHEST_PROJECT_DIR': project_dir, 'ORCHEST_PIPELINE_FILE': pipeline_path}
    environment = [{'name': k, 'value': v} for (k, v) in environment.items()]
    environment.extend(_get_orchest_sdk_vars(project_uuid, pipeline_uuid, _config.PIPELINE_FILE, session_uuid, session_type))
    (volumes_dict, volume_mounts_dict) = _get_jupyter_volumes_and_volume_mounts(project_uuid, userdir_pvc, project_dir, pipeline_path)
    deployment_manifest = {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': metadata, 'spec': {'replicas': 1, 'selector': {'matchLabels': metadata['labels']}, 'template': {'metadata': metadata, 'spec': {'securityContext': {'runAsUser': 0, 'runAsGroup': int(os.environ.get('ORCHEST_HOST_GID')), 'fsGroup': int(os.environ.get('ORCHEST_HOST_GID'))}, 'serviceAccount': f'jupyter-eg-sa-{session_uuid}', 'serviceAccountName': f'jupyter-eg-sa-{session_uuid}', 'terminationGracePeriodSeconds': 5, 'volumes': [volumes_dict['userdir-pvc']], 'containers': [{'name': metadata['name'], 'image': 'orchest/jupyter-enterprise-gateway:' + CONFIG_CLASS.ORCHEST_VERSION, 'resources': {'requests': {'cpu': _config.USER_CONTAINERS_CPU_SHARES}}, 'imagePullPolicy': 'IfNotPresent', 'env': environment, 'volumeMounts': [volume_mounts_dict['kernelspec']], 'ports': [{'containerPort': 8888}]}]}}}}
    service_manifest = {'apiVersion': 'v1', 'kind': 'Service', 'metadata': metadata, 'spec': {'type': 'ClusterIP', 'selector': metadata['labels'], 'ports': [{'port': 8888}]}}
    return (deployment_manifest, service_manifest)

def _get_user_service_deployment_service_manifest(session_uuid: str, session_config: SessionConfig, service_config: Dict[str, Any], session_type: SessionType) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]]:
    if False:
        return 10
    'Get deployment and service manifest for a user service.\n\n    Args:\n        session_uuid:\n        session_config: See `Args` section in class :class:`Session`\n            __init__ method.\n        service_config: See `Args` section in class :class:`Session`\n            __init__ method.\n        session_type: Type of session: interactive, or\n            noninteractive.\n\n    Returns:\n        Tuple of k8s deployment, service and ingress manifests to deploy\n        this user service in the session. The ingress is None if\n        service.exposed is False.\n\n    '
    project_uuid = session_config['project_uuid']
    pipeline_uuid = session_config['pipeline_uuid']
    pipeline_path = session_config['pipeline_path']
    project_dir = session_config['project_dir']
    userdir_pvc = session_config['userdir_pvc']
    img_mappings = session_config['env_uuid_to_image']
    is_pbp_enabled = service_config.get('preserve_base_path', False)
    ingress_url = 'service-' + service_config['name'] + '-' + session_uuid
    if is_pbp_enabled:
        ingress_url = 'pbp-' + ingress_url
    service_str = json.dumps(service_config)
    service_str = service_str.replace('$BASE_PATH_PREFIX', ingress_url)
    service_config = json.loads(service_str)
    try:
        if session_type.value == 'noninteractive':
            user_env_variables = session_config['user_env_variables']
        else:
            user_env_variables = utils.get_proj_pip_env_variables(project_uuid, pipeline_uuid)
    except Exception as e:
        logger.error('Failed to fetch user_env_variables: %s [%s]' % (e, type(e)))
        traceback.print_exc()
        user_env_variables = {}
    environment = service_config.get('env_variables', {})
    for inherited_key in service_config.get('env_variables_inherit', []):
        if inherited_key in user_env_variables:
            environment[inherited_key] = user_env_variables[inherited_key]
    env = _get_orchest_sdk_vars(project_uuid, pipeline_uuid, _config.PIPELINE_FILE, session_uuid, session_type)
    for (k, v) in environment.items():
        env.append({'name': k, 'value': v})
    volume_mounts = []
    volumes = []
    sbinds = service_config.get('binds', {})
    (volumes_dict, volume_mounts_dict) = _get_common_volumes_and_volume_mounts(userdir_pvc, project_dir, pipeline_path, container_project_dir=sbinds.get('/project-dir', _config.PROJECT_DIR), container_data_dir=sbinds.get('/data', _config.DATA_DIR))
    if '/data' in sbinds:
        volume_mounts.append(volume_mounts_dict['data'])
    if '/project-dir' in sbinds:
        volume_mounts.append(volume_mounts_dict['project-dir'])
    if '/data' in sbinds or '/project-dir' in sbinds:
        volumes.append(volumes_dict['userdir-pvc'])
    volumes.append(volumes_dict['container-runtime-socket'])
    image = service_config['image']
    prefix = _config.ENVIRONMENT_AS_SERVICE_PREFIX
    if image.startswith(prefix):
        registry_ip = utils.get_registry_ip()
        image = image.replace(prefix, '')
        image = img_mappings[image]
        image = registry_ip + '/' + image
    metadata = {'name': service_config['name'] + '-' + session_uuid, 'labels': {'app': service_config['name'], 'project_uuid': project_uuid, 'session_uuid': session_uuid}}
    deployment_manifest = {'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': metadata, 'spec': {'replicas': 1, 'selector': {'matchLabels': metadata['labels']}, 'template': {'metadata': metadata, 'spec': {'terminationGracePeriodSeconds': 5, 'securityContext': {'runAsUser': 0, 'runAsGroup': int(os.environ.get('ORCHEST_HOST_GID')), 'fsGroup': int(os.environ.get('ORCHEST_HOST_GID'))}, 'dnsConfig': {'options': [{'name': 'timeout', 'value': '10'}, {'name': 'attempts', 'value': '5'}]}, 'volumes': volumes, 'containers': [{'name': metadata['name'], 'image': image, 'imagePullPolicy': 'IfNotPresent', 'env': env, 'resources': {'requests': {'cpu': _config.USER_CONTAINERS_CPU_SHARES}}, 'volumeMounts': volume_mounts, 'ports': [{'containerPort': port} for port in service_config['ports']]}]}}}}
    pod_scheduling.modify_user_service_scheduling_behaviour(session_type.value, deployment_manifest)
    if service_config.get('command', ''):
        deployment_manifest['spec']['template']['spec']['containers'][0]['command'] = [service_config['command']]
    if 'args' in service_config:
        deployment_manifest['spec']['template']['spec']['containers'][0]['args'] = shlex.split(service_config['args'])
    service_manifest = {'apiVersion': 'v1', 'kind': 'Service', 'metadata': metadata, 'spec': {'type': 'ClusterIP', 'selector': metadata['labels'], 'ports': [{'port': port, 'name': f'port-{port}'} for port in service_config['ports']]}}
    if service_config['exposed']:
        ingress_paths = []
        for port in service_config.get('ports', []):
            ingress_paths.append({'backend': {'service': {'name': metadata['name'], 'port': {'number': port}}}, 'path': f'/({ingress_url}_{port}.*)' if is_pbp_enabled else f'/{ingress_url}_{port}(/|$)(.*)', 'pathType': 'Prefix'})
        ingress_metadata = copy.deepcopy(metadata)
        ingress_metadata['annotations'] = {'nginx.ingress.kubernetes.io/rewrite-target': '/$1' if is_pbp_enabled else '/$2'}
        if service_config.get('requires_authentication', True):
            auth_url = f'http://auth-server.{_config.ORCHEST_NAMESPACE}.svc.cluster.local/auth'
            ingress_metadata['annotations']['nginx.ingress.kubernetes.io/auth-url'] = auth_url
            ingress_metadata['annotations']['nginx.ingress.kubernetes.io/auth-signin'] = '/login'
        ingress_rule = {}
        if _config.ORCHEST_FQDN is not None:
            ingress_rule['host'] = _config.ORCHEST_FQDN
        ingress_rule['http'] = {'paths': ingress_paths}
        ingress_manifest = {'apiVersion': 'networking.k8s.io/v1', 'kind': 'Ingress', 'metadata': ingress_metadata, 'spec': {'ingressClassName': _config.INGRESS_CLASS, 'rules': [ingress_rule]}}
    else:
        ingress_manifest = None
    return (deployment_manifest, service_manifest, ingress_manifest)