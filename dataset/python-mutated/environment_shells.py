import time
from typing import Optional
from _orchest.internals import config as _config
from app import utils
from app.connections import k8s_apps_api, k8s_core_api
from app.core.sessions import _manifests
logger = utils.get_logger()

def launch_environment_shell(session_uuid: str, service_name: str, shell_uuid: str, project_uuid: str, pipeline_uuid: str, pipeline_path: str, userdir_pvc: str, project_dir: str, environment_image: str, auth_user_uuid: Optional[str]=None) -> None:
    if False:
        i = 10
        return i + 15
    "Starts environment shell\n\n    Args:\n        session_uuid: UUID to identify the session k8s namespace with,\n            which is where all related resources will be deployed.\n        service_name: service name used for the k8s service for\n            host based communication.\n        shell_uuid: UUID to identify the shell.\n        project_uuid: UUID of the project.\n        pipeline_uuid: UUID of the pipeline.\n        pipeline_path: Relative path (from project directory root) to\n            the pipeline file e.g. 'abc/pipeline.orchest'.\n        userdir_pvc: Name of the k8s PVC e.g. 'userdir-pvc'.\n        project_dir: Name of the project directory e.g. 'my-project'\n            note this is always a single path component.\n        environment_image: The full image specification that can be\n            given directly as the image string to the container runtime.\n        auth_user_uuid: uuid of the auth user for which to inject the\n         git configuration if exists.\n\n\n    The resources created in k8s\n      deployments\n      services\n      ingresses\n      pods\n      service_accounts\n      role_bindings\n      roles\n\n    Will be cleaned up when the session is stopped.\n    "
    (environment_shell_deployment_manifest, environment_shell_service_manifest) = _manifests._get_environment_shell_deployment_service_manifest(session_uuid, service_name, shell_uuid, project_uuid, pipeline_uuid, pipeline_path, userdir_pvc, project_dir, environment_image, auth_user_uuid)
    ns = _config.ORCHEST_NAMESPACE
    logger.info('Creating deployment %s' % (environment_shell_deployment_manifest['metadata']['name'],))
    k8s_apps_api.create_namespaced_deployment(ns, environment_shell_deployment_manifest)
    logger.info(f"Creating service {environment_shell_service_manifest['metadata']['name']}")
    k8s_core_api.create_namespaced_service(ns, environment_shell_service_manifest)
    logger.info('Waiting for environment shell service deployment to be ready.')
    deployment_name = environment_shell_deployment_manifest['metadata']['name']
    deployment = k8s_apps_api.read_namespaced_deployment_status(deployment_name, ns)
    while deployment.status.available_replicas != deployment.spec.replicas:
        logger.info(f'Waiting for {deployment_name}.')
        time.sleep(1)
        deployment = k8s_apps_api.read_namespaced_deployment_status(deployment_name, ns)

def get_environment_shells(session_uuid: str):
    if False:
        i = 10
        return i + 15
    'Gets all related resources, idempotent.'
    ns = _config.ORCHEST_NAMESPACE
    label_selector = f'session_uuid={session_uuid},app=environment-shell'
    try:
        services = k8s_core_api.list_namespaced_service(ns, label_selector=label_selector)
        return [{'hostname': service.metadata.name, 'session_uuid': session_uuid, 'uuid': service.metadata.name.replace('environment-shell-', '')} for service in services.items]
    except Exception as e:
        logger.error('Failed to get environment shells for session UUID %s' % session_uuid)
        logger.error('Error %s [%s]' % (e, type(e)))
        return []

def stop_environment_shell(environment_shell_uuid: str):
    if False:
        for i in range(10):
            print('nop')
    'Deletes environment shell.'
    ns = _config.ORCHEST_NAMESPACE
    name = 'environment-shell-' + environment_shell_uuid
    try:
        k8s_apps_api.delete_namespaced_deployment(name, ns)
        k8s_core_api.delete_namespaced_service(name, ns)
    except Exception as e:
        logger.error('Failed to delete environment shell with UUID %s' % environment_shell_uuid)
        logger.error('Error %s [%s]' % (e, type(e)))