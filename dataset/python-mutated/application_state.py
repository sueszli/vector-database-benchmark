import logging
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import ray
from ray import cloudpickle
from ray._private.utils import import_attr
from ray.exceptions import RuntimeEnvSetupError
from ray.serve._private.common import ApplicationStatus, ApplicationStatusInfo, DeploymentID, DeploymentInfo, DeploymentStatus, DeploymentStatusInfo, EndpointInfo, EndpointTag
from ray.serve._private.config import DeploymentConfig
from ray.serve._private.constants import SERVE_LOGGER_NAME
from ray.serve._private.deploy_utils import deploy_args_to_deployment_info, get_app_code_version, get_deploy_args
from ray.serve._private.deployment_state import DeploymentStateManager
from ray.serve._private.endpoint_state import EndpointState
from ray.serve._private.storage.kv_store import KVStoreBase
from ray.serve._private.usage import ServeUsageTag
from ray.serve._private.utils import DEFAULT, check_obj_ref_ready_nowait, override_runtime_envs_except_env_vars
from ray.serve.exceptions import RayServeException
from ray.serve.schema import DeploymentDetails, ServeApplicationSchema
from ray.types import ObjectRef
logger = logging.getLogger(SERVE_LOGGER_NAME)
CHECKPOINT_KEY = 'serve-application-state-checkpoint'

class BuildAppStatus(Enum):
    """Status of the build application task."""
    NO_TASK_IN_PROGRESS = 1
    IN_PROGRESS = 2
    SUCCEEDED = 3
    FAILED = 4

@dataclass
class BuildAppTaskInfo:
    """Stores info on the current in-progress build app task.

    We use a class instead of only storing the task object ref because
    when a new config is deployed, there can be an outdated in-progress
    build app task. We attach the code version to the task info to
    distinguish outdated build app tasks.
    """
    obj_ref: ObjectRef
    code_version: str
    finished: bool

@dataclass
class ApplicationTargetState:
    """Defines target state of application.

    Target state can become inconsistent if the code version doesn't
    match that of the config. When that happens, a new build app task
    should be kicked off to reconcile the inconsistency.

    deployment_infos: Map of deployment name to deployment info. This is
      - None if a config was deployed but the app hasn't finished
        building yet
      - An empty dict if the app is deleting
    code_version: Code version of all deployments in target state. None
        if application was deployed through serve.run
    config: application config deployed by user. None if application was
        deployed through serve.run
    deleting: whether the application is being deleted.
    """
    deployment_infos: Optional[Dict[str, DeploymentInfo]]
    code_version: Optional[str]
    config: Optional[ServeApplicationSchema]
    deleting: bool

class ApplicationState:
    """Manage single application states with all operations"""

    def __init__(self, name: str, deployment_state_manager: DeploymentStateManager, endpoint_state: EndpointState, save_checkpoint_func: Callable):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            name: Application name.\n            deployment_state_manager: State manager for all deployments\n                in the cluster.\n            endpoint_state: State manager for endpoints in the system.\n            save_checkpoint_func: Function that can be called to write\n                a checkpoint of the application state. This should be\n                called in self._set_target_state() before actually\n                setting the target state so that the controller can\n                properly recover application states if it crashes.\n        '
        self._name = name
        self._status_msg = ''
        self._deployment_state_manager = deployment_state_manager
        self._endpoint_state = endpoint_state
        self._route_prefix: Optional[str] = None
        self._docs_path: Optional[str] = None
        self._ingress_deployment_name: str = None
        self._status: ApplicationStatus = ApplicationStatus.DEPLOYING
        self._deployment_timestamp = time.time()
        self._build_app_task_info: Optional[BuildAppTaskInfo] = None
        self._target_state: ApplicationTargetState = ApplicationTargetState(deployment_infos=None, code_version=None, config=None, deleting=False)
        self._save_checkpoint_func = save_checkpoint_func

    @property
    def route_prefix(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._route_prefix

    @property
    def docs_path(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._docs_path

    @property
    def status(self) -> ApplicationStatus:
        if False:
            return 10
        "Status of the application.\n\n        DEPLOYING: The build task is still running, or the deployments\n            have started deploying but aren't healthy yet.\n        RUNNING: All deployments are healthy.\n        DEPLOY_FAILED: The build task failed or one or more deployments\n            became unhealthy in the process of deploying\n        UNHEALTHY: While the application was running, one or more\n            deployments transition from healthy to unhealthy.\n        DELETING: Application and its deployments are being deleted.\n        "
        return self._status

    @property
    def deployment_timestamp(self) -> int:
        if False:
            while True:
                i = 10
        return self._deployment_timestamp

    @property
    def target_deployments(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        'List of target deployment names in application.'
        if self._target_state.deployment_infos is None:
            return []
        return list(self._target_state.deployment_infos.keys())

    @property
    def ingress_deployment(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._ingress_deployment_name

    def recover_target_state_from_checkpoint(self, checkpoint_data: ApplicationTargetState):
        if False:
            while True:
                i = 10
        logger.info(f"Recovering target state for application '{self._name}' from checkpoint.")
        self._set_target_state(checkpoint_data.deployment_infos, checkpoint_data.code_version, checkpoint_data.config, checkpoint_data.deleting)

    def _set_target_state(self, deployment_infos: Optional[Dict[str, DeploymentInfo]], code_version: str, target_config: Optional[ServeApplicationSchema], deleting: bool=False):
        if False:
            return 10
        'Set application target state.\n\n        While waiting for build task to finish, this should be\n            (None, False)\n        When build task has finished and during normal operation, this should be\n            (target_deployments, False)\n        When a request to delete the application has been received, this should be\n            ({}, True)\n        '
        if deleting:
            self._update_status(ApplicationStatus.DELETING)
        else:
            self._update_status(ApplicationStatus.DEPLOYING)
        if deployment_infos is None:
            self._ingress_deployment_name = None
        else:
            for (name, info) in deployment_infos.items():
                if info.ingress:
                    self._ingress_deployment_name = name
        target_state = ApplicationTargetState(deployment_infos, code_version, target_config, deleting)
        self._save_checkpoint_func(writeahead_checkpoints={self._name: target_state})
        self._target_state = target_state

    def _set_target_state_deployment_infos(self, deployment_infos: Optional[Dict[str, DeploymentInfo]]):
        if False:
            for i in range(10):
                print('nop')
        'Updates only the target deployment infos.'
        self._set_target_state(deployment_infos=deployment_infos, code_version=self._target_state.code_version, target_config=self._target_state.config)

    def _set_target_state_config(self, target_config: Optional[ServeApplicationSchema]):
        if False:
            return 10
        'Updates only the target config.'
        self._set_target_state(deployment_infos=self._target_state.deployment_infos, code_version=self._target_state.code_version, target_config=target_config)

    def _set_target_state_deleting(self):
        if False:
            i = 10
            return i + 15
        'Set target state to deleting.\n\n        Wipes the target deployment infos, code version, and config.\n        '
        self._set_target_state(dict(), None, None, True)

    def _delete_deployment(self, name):
        if False:
            return 10
        id = EndpointTag(name, self._name)
        self._endpoint_state.delete_endpoint(id)
        self._deployment_state_manager.delete_deployment(id)

    def delete(self):
        if False:
            print('Hello World!')
        'Delete the application'
        if self._status != ApplicationStatus.DELETING:
            logger.info(f"Deleting application '{self._name}'", extra={'log_to_stderr': False})
        self._set_target_state_deleting()

    def is_deleted(self) -> bool:
        if False:
            return 10
        'Check whether the application is already deleted.\n\n        For an application to be considered deleted, the target state has to be set to\n        deleting and all deployments have to be deleted.\n        '
        return self._target_state.deleting and len(self._get_live_deployments()) == 0

    def apply_deployment_info(self, deployment_name: str, deployment_info: DeploymentInfo) -> None:
        if False:
            return 10
        'Deploys a deployment in the application.'
        route_prefix = deployment_info.route_prefix
        if route_prefix is not None and (not route_prefix.startswith('/')):
            raise RayServeException(f'Invalid route prefix "{route_prefix}", it must start with "/"')
        deployment_id = DeploymentID(deployment_name, self._name)
        self._deployment_state_manager.deploy(deployment_id, deployment_info)
        if deployment_info.route_prefix is not None:
            config = deployment_info.deployment_config
            self._endpoint_state.update_endpoint(deployment_id, EndpointInfo(route=deployment_info.route_prefix, app_is_cross_language=config.is_cross_language))
        else:
            self._endpoint_state.delete_endpoint(deployment_id)

    def deploy(self, deployment_infos: Dict[str, DeploymentInfo]):
        if False:
            print('Hello World!')
        'Deploy application from list of deployment infos.\n\n        This function should only be called if the app is being deployed\n        through serve.run instead of from a config.\n\n        Raises: RayServeException if there is more than one route prefix\n            or docs path.\n        '
        (self._route_prefix, self._docs_path) = self._check_routes(deployment_infos)
        self._set_target_state(deployment_infos=deployment_infos, code_version=None, target_config=None)

    def deploy_config(self, config: ServeApplicationSchema, deployment_time: int) -> None:
        if False:
            print('Hello World!')
        "Deploys an application config.\n\n        If the code version matches that of the current live deployments\n        then it only applies the updated config to the deployment state\n        manager. If the code version doesn't match, this will re-build\n        the application.\n        "
        self._deployment_timestamp = deployment_time
        self._set_target_state_config(config)
        config_version = get_app_code_version(config)
        if config_version == self._target_state.code_version:
            try:
                overrided_infos = override_deployment_info(self._name, self._target_state.deployment_infos, self._target_state.config)
                self._check_routes(overrided_infos)
                self._set_target_state_deployment_infos(overrided_infos)
            except (TypeError, ValueError, RayServeException):
                self._set_target_state(deployment_infos=None, code_version=None, target_config=self._target_state.config)
                self._update_status(ApplicationStatus.DEPLOY_FAILED, traceback.format_exc())
            except Exception:
                self._set_target_state(deployment_infos=None, code_version=None, target_config=self._target_state.config)
                self._update_status(ApplicationStatus.DEPLOY_FAILED, f"Unexpected error occured while applying config for application '{self._name}': \n{traceback.format_exc()}")
        else:
            if self._build_app_task_info and (not self._build_app_task_info.finished):
                logger.info(f"Received new config for application '{self._name}'. Cancelling previous request.")
                ray.cancel(self._build_app_task_info.obj_ref)
            self._set_target_state(deployment_infos=None, code_version=None, target_config=self._target_state.config)
            logger.info(f"Building application '{self._name}'.")
            build_app_obj_ref = build_serve_application.options(runtime_env=self._target_state.config.runtime_env).remote(self._target_state.config.import_path, self._target_state.config.deployment_names, config_version, self._target_state.config.name, self._target_state.config.args)
            self._build_app_task_info = BuildAppTaskInfo(build_app_obj_ref, config_version, False)

    def _get_live_deployments(self) -> List[str]:
        if False:
            while True:
                i = 10
        return self._deployment_state_manager.get_deployments_in_application(self._name)

    def _determine_app_status(self) -> Tuple[ApplicationStatus, str]:
        if False:
            while True:
                i = 10
        'Check deployment statuses and target state, and determine the\n        corresponding application status.\n\n        Returns:\n            Status (ApplicationStatus):\n                RUNNING: all deployments are healthy.\n                DEPLOYING: there is one or more updating deployments,\n                    and there are no unhealthy deployments.\n                DEPLOY_FAILED: one or more deployments became unhealthy\n                    while the application was deploying.\n                UNHEALTHY: one or more deployments became unhealthy\n                    while the application was running.\n                DELETING: the application is being deleted.\n            Error message (str):\n                Non-empty string if status is DEPLOY_FAILED or UNHEALTHY\n        '
        if self._target_state.deleting:
            return (ApplicationStatus.DELETING, '')
        num_healthy_deployments = 0
        unhealthy_deployment_names = []
        for deployment_status in self.get_deployments_statuses():
            if deployment_status.status == DeploymentStatus.UNHEALTHY:
                unhealthy_deployment_names.append(deployment_status.name)
            if deployment_status.status == DeploymentStatus.HEALTHY:
                num_healthy_deployments += 1
        if num_healthy_deployments == len(self.target_deployments):
            return (ApplicationStatus.RUNNING, '')
        elif len(unhealthy_deployment_names):
            status_msg = f'The deployments {unhealthy_deployment_names} are UNHEALTHY.'
            if self._status in [ApplicationStatus.DEPLOYING, ApplicationStatus.DEPLOY_FAILED]:
                return (ApplicationStatus.DEPLOY_FAILED, status_msg)
            else:
                return (ApplicationStatus.UNHEALTHY, status_msg)
        else:
            return (ApplicationStatus.DEPLOYING, '')

    def _reconcile_build_app_task(self) -> Tuple[Tuple, BuildAppStatus, str]:
        if False:
            i = 10
            return i + 15
        "If necessary, reconcile the in-progress build task.\n\n        Returns:\n            Deploy arguments (Dict[str, DeploymentInfo]):\n                The deploy arguments returned from the build app task\n                and their code version.\n            Status (BuildAppStatus):\n                NO_TASK_IN_PROGRESS: There is no build task to reconcile.\n                SUCCEEDED: Task finished successfully.\n                FAILED: An error occurred during execution of build app task\n                IN_PROGRESS: Task hasn't finished yet.\n            Error message (str):\n                Non-empty string if status is DEPLOY_FAILED or UNHEALTHY\n        "
        if self._target_state.config is None or self._build_app_task_info is None or self._build_app_task_info.finished:
            return (None, BuildAppStatus.NO_TASK_IN_PROGRESS, '')
        if not check_obj_ref_ready_nowait(self._build_app_task_info.obj_ref):
            return (None, BuildAppStatus.IN_PROGRESS, '')
        self._build_app_task_info.finished = True
        try:
            (args, err) = ray.get(self._build_app_task_info.obj_ref)
            if err is None:
                logger.info(f"Built application '{self._name}' successfully.")
            else:
                return (None, BuildAppStatus.FAILED, f"Deploying app '{self._name}' failed with exception:\n{err}")
        except RuntimeEnvSetupError:
            error_msg = f"Runtime env setup for app '{self._name}' failed:\n" + traceback.format_exc()
            return (None, BuildAppStatus.FAILED, error_msg)
        except Exception:
            error_msg = f"Unexpected error occured while deploying application '{self._name}': \n{traceback.format_exc()}"
            return (None, BuildAppStatus.FAILED, error_msg)
        try:
            deployment_infos = {params['deployment_name']: deploy_args_to_deployment_info(**params, app_name=self._name) for params in args}
            overrided_infos = override_deployment_info(self._name, deployment_infos, self._target_state.config)
            (self._route_prefix, self._docs_path) = self._check_routes(overrided_infos)
            return (overrided_infos, BuildAppStatus.SUCCEEDED, '')
        except (TypeError, ValueError, RayServeException):
            return (None, BuildAppStatus.FAILED, traceback.format_exc())
        except Exception:
            error_msg = f"Unexpected error occured while applying config for application '{self._name}': \n{traceback.format_exc()}"
            return (None, BuildAppStatus.FAILED, error_msg)

    def _check_routes(self, deployment_infos: Dict[str, DeploymentInfo]) -> Tuple[str, str]:
        if False:
            print('Hello World!')
        'Check route prefixes and docs paths of deployments in app.\n\n        There should only be one non-null route prefix. If there is one,\n        set it as the application route prefix. This function must be\n        run every control loop iteration because the target config could\n        be updated without kicking off a new task.\n\n        Returns: tuple of route prefix, docs path.\n        Raises: RayServeException if more than one route prefix or docs\n            path is found among deployments.\n        '
        num_route_prefixes = 0
        num_docs_paths = 0
        route_prefix = None
        docs_path = None
        for info in deployment_infos.values():
            if info.route_prefix is not None:
                route_prefix = info.route_prefix
                num_route_prefixes += 1
            if info.docs_path is not None:
                docs_path = info.docs_path
                num_docs_paths += 1
        if num_route_prefixes > 1:
            raise RayServeException(f'Found multiple route prefixes from application "{self._name}", Please specify only one route prefix for the application to avoid this issue.')
        if num_docs_paths > 1:
            raise RayServeException(f'Found multiple deployments in application "{self._name}" that have a docs path. This may be due to using multiple FastAPI deployments in your application. Please only include one deployment with a docs path in your application to avoid this issue.')
        return (route_prefix, docs_path)

    def _reconcile_target_deployments(self) -> None:
        if False:
            while True:
                i = 10
        'Reconcile target deployments in application target state.\n\n        Ensure each deployment is running on up-to-date info, and\n        remove outdated deployments from the application.\n        '
        for (deployment_name, info) in self._target_state.deployment_infos.items():
            deploy_info = deepcopy(info)
            if self._target_state.config and self._target_state.config.logging_config and (deploy_info.deployment_config.logging_config is None):
                deploy_info.deployment_config.logging_config = self._target_state.config.logging_config
            self.apply_deployment_info(deployment_name, deploy_info)
        for deployment_name in self._get_live_deployments():
            if deployment_name not in self.target_deployments:
                self._delete_deployment(deployment_name)

    def update(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Attempts to reconcile this application to match its target state.\n\n        Updates the application status and status message based on the\n        current state of the system.\n\n        Returns:\n            A boolean indicating whether the application is ready to be\n            deleted.\n        '
        (infos, task_status, msg) = self._reconcile_build_app_task()
        if task_status == BuildAppStatus.SUCCEEDED:
            self._set_target_state(deployment_infos=infos, code_version=self._build_app_task_info.code_version, target_config=self._target_state.config)
        elif task_status == BuildAppStatus.FAILED:
            self._update_status(ApplicationStatus.DEPLOY_FAILED, msg)
        if self._target_state.deployment_infos is not None:
            self._reconcile_target_deployments()
            (status, status_msg) = self._determine_app_status()
            self._update_status(status, status_msg)
        if self._target_state.deleting:
            return self.is_deleted()
        return False

    def get_checkpoint_data(self) -> ApplicationTargetState:
        if False:
            print('Hello World!')
        return self._target_state

    def get_deployments_statuses(self) -> List[DeploymentStatusInfo]:
        if False:
            while True:
                i = 10
        'Return all deployment status information'
        deployments = [DeploymentID(deployment, self._name) for deployment in self.target_deployments]
        return self._deployment_state_manager.get_deployment_statuses(deployments)

    def get_application_status_info(self) -> ApplicationStatusInfo:
        if False:
            while True:
                i = 10
        'Return the application status information'
        return ApplicationStatusInfo(self._status, message=self._status_msg, deployment_timestamp=self._deployment_timestamp)

    def list_deployment_details(self) -> Dict[str, DeploymentDetails]:
        if False:
            i = 10
            return i + 15
        'Gets detailed info on all live deployments in this application.\n        (Does not include deleted deployments.)\n\n        Returns:\n            A dictionary of deployment infos. The set of deployment info returned\n            may not be the full list of deployments that are part of the application.\n            This can happen when the application is still deploying and bringing up\n            deployments, or when the application is deleting and some deployments have\n            been deleted.\n        '
        details = {deployment_name: self._deployment_state_manager.get_deployment_details(DeploymentID(deployment_name, self._name)) for deployment_name in self.target_deployments}
        return {k: v for (k, v) in details.items() if v is not None}

    def _update_status(self, status: ApplicationStatus, status_msg: str='') -> None:
        if False:
            for i in range(10):
                print('nop')
        if status_msg and status in [ApplicationStatus.DEPLOY_FAILED, ApplicationStatus.UNHEALTHY]:
            logger.warning(status_msg)
        self._status = status
        self._status_msg = status_msg

class ApplicationStateManager:

    def __init__(self, deployment_state_manager: DeploymentStateManager, endpoint_state: EndpointState, kv_store: KVStoreBase):
        if False:
            print('Hello World!')
        self._deployment_state_manager = deployment_state_manager
        self._endpoint_state = endpoint_state
        self._kv_store = kv_store
        self._application_states: Dict[str, ApplicationState] = dict()
        self._recover_from_checkpoint()

    def _recover_from_checkpoint(self):
        if False:
            while True:
                i = 10
        checkpoint = self._kv_store.get(CHECKPOINT_KEY)
        if checkpoint is not None:
            application_state_info = cloudpickle.loads(checkpoint)
            for (app_name, checkpoint_data) in application_state_info.items():
                app_state = ApplicationState(app_name, self._deployment_state_manager, self._endpoint_state, self._save_checkpoint_func)
                app_state.recover_target_state_from_checkpoint(checkpoint_data)
                self._application_states[app_name] = app_state

    def delete_application(self, name: str) -> None:
        if False:
            while True:
                i = 10
        'Delete application by name'
        if name not in self._application_states:
            return
        self._application_states[name].delete()

    def apply_deployment_args(self, name: str, deployment_args: List[Dict]) -> None:
        if False:
            while True:
                i = 10
        'Apply list of deployment arguments to application target state.\n\n        This function should only be called if the app is being deployed\n        through serve.run instead of from a config.\n\n        Args:\n            name: application name\n            deployment_args_list: arguments for deploying a list of deployments.\n\n        Raises:\n            RayServeException: If the list of deployments is trying to\n                use a route prefix that is already used by another application\n        '
        live_route_prefixes: Dict[str, str] = {self._application_states[app_name].route_prefix: app_name for (app_name, app_state) in self._application_states.items() if app_state.route_prefix is not None and (not app_state.status == ApplicationStatus.DELETING) and (name != app_name)}
        for deploy_param in deployment_args:
            deploy_app_prefix = deploy_param.get('route_prefix')
            if deploy_app_prefix in live_route_prefixes:
                raise RayServeException(f'Prefix {deploy_app_prefix} is being used by application "{live_route_prefixes[deploy_app_prefix]}". Failed to deploy application "{name}".')
        if name not in self._application_states:
            self._application_states[name] = ApplicationState(name, self._deployment_state_manager, self._endpoint_state, self._save_checkpoint_func)
        ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))
        deployment_infos = {params['deployment_name']: deploy_args_to_deployment_info(**params, app_name=name) for params in deployment_args}
        self._application_states[name].deploy(deployment_infos)

    def deploy_config(self, name: str, app_config: ServeApplicationSchema, deployment_time: float=0) -> None:
        if False:
            while True:
                i = 10
        'Deploy application from config.'
        if name not in self._application_states:
            self._application_states[name] = ApplicationState(name, self._deployment_state_manager, endpoint_state=self._endpoint_state, save_checkpoint_func=self._save_checkpoint_func)
        ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))
        self._application_states[name].deploy_config(app_config, deployment_time)

    def get_deployments(self, app_name: str) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return all deployment names by app name'
        if app_name not in self._application_states:
            return []
        return self._application_states[app_name].target_deployments

    def get_deployments_statuses(self, app_name: str) -> List[DeploymentStatusInfo]:
        if False:
            for i in range(10):
                print('nop')
        'Return all deployment statuses by app name'
        if app_name not in self._application_states:
            return []
        return self._application_states[app_name].get_deployments_statuses()

    def get_app_status(self, name: str) -> ApplicationStatus:
        if False:
            i = 10
            return i + 15
        if name not in self._application_states:
            return ApplicationStatus.NOT_STARTED
        return self._application_states[name].status

    def get_app_status_info(self, name: str) -> ApplicationStatusInfo:
        if False:
            return 10
        if name not in self._application_states:
            return ApplicationStatusInfo(ApplicationStatus.NOT_STARTED, message=f"Application {name} doesn't exist", deployment_timestamp=0)
        return self._application_states[name].get_application_status_info()

    def get_docs_path(self, app_name: str) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._application_states[app_name].docs_path

    def get_route_prefix(self, name: str) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._application_states[name].route_prefix

    def get_ingress_deployment_name(self, name: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        if name not in self._application_states:
            return None
        return self._application_states[name].ingress_deployment

    def list_app_statuses(self) -> Dict[str, ApplicationStatusInfo]:
        if False:
            i = 10
            return i + 15
        'Return a dictionary with {app name: application info}'
        return {name: self._application_states[name].get_application_status_info() for name in self._application_states}

    def list_deployment_details(self, name: str) -> Dict[str, DeploymentDetails]:
        if False:
            for i in range(10):
                print('nop')
        'Gets detailed info on all deployments in specified application.'
        if name not in self._application_states:
            return {}
        return self._application_states[name].list_deployment_details()

    def update(self):
        if False:
            i = 10
            return i + 15
        'Update each application state'
        apps_to_be_deleted = []
        for (name, app) in self._application_states.items():
            ready_to_be_deleted = app.update()
            if ready_to_be_deleted:
                apps_to_be_deleted.append(name)
                logger.debug(f"Application '{name}' deleted successfully.")
        if len(apps_to_be_deleted) > 0:
            for app_name in apps_to_be_deleted:
                del self._application_states[app_name]
            ServeUsageTag.NUM_APPS.record(str(len(self._application_states)))

    def shutdown(self) -> None:
        if False:
            return 10
        for app_state in self._application_states.values():
            app_state.delete()
        self._kv_store.delete(CHECKPOINT_KEY)

    def is_ready_for_shutdown(self) -> bool:
        if False:
            return 10
        'Return whether all applications have shut down.\n\n        Iterate through all application states and check if all their applications\n        are deleted.\n        '
        return all((app_state.is_deleted() for app_state in self._application_states.values()))

    def _save_checkpoint_func(self, *, writeahead_checkpoints: Optional[Dict[str, ApplicationTargetState]]) -> None:
        if False:
            i = 10
            return i + 15
        'Write a checkpoint of all application states.'
        application_state_info = {app_name: app_state.get_checkpoint_data() for (app_name, app_state) in self._application_states.items()}
        if writeahead_checkpoints is not None:
            application_state_info.update(writeahead_checkpoints)
        self._kv_store.put(CHECKPOINT_KEY, cloudpickle.dumps(application_state_info))

@ray.remote(num_cpus=0, max_calls=1)
def build_serve_application(import_path: str, config_deployments: List[str], code_version: str, name: str, args: Dict) -> Tuple[List[Dict], Optional[str]]:
    if False:
        return 10
    'Import and build a Serve application.\n\n    Args:\n        import_path: import path to top-level bound deployment.\n        config_deployments: list of deployment names specified in config\n            with deployment override options. This is used to check that\n            all deployments specified in the config are valid.\n        code_version: code version inferred from app config. All\n            deployment versions are set to this code version.\n        name: application name. If specified, application will be deployed\n            without removing existing applications.\n        args: Arguments to be passed to the application builder.\n        logging_config: The application logging config, if deployment logging\n            config is not set, application logging config will be applied to the\n            deployment logging config.\n    Returns:\n        Deploy arguments: a list of deployment arguments if application\n            was built successfully, otherwise None.\n        Error message: a string if an error was raised, otherwise None.\n    '
    try:
        from ray.serve._private.api import call_app_builder_with_args_if_necessary
        from ray.serve._private.deployment_graph_build import build as pipeline_build
        from ray.serve._private.deployment_graph_build import get_and_validate_ingress_deployment
        app = call_app_builder_with_args_if_necessary(import_attr(import_path), args)
        deployments = pipeline_build(app._get_internal_dag_node(), name)
        ingress = get_and_validate_ingress_deployment(deployments)
        deploy_args_list = []
        for deployment in deployments:
            is_ingress = deployment.name == ingress.name
            deploy_args_list.append(get_deploy_args(name=deployment._name, replica_config=deployment._replica_config, ingress=is_ingress, deployment_config=deployment._deployment_config, version=code_version, route_prefix=deployment.route_prefix, docs_path=deployment._docs_path))
        return (deploy_args_list, None)
    except KeyboardInterrupt:
        logger.info('Existing config deployment request terminated.')
        return (None, None)
    except Exception:
        return (None, traceback.format_exc())

def override_deployment_info(app_name: str, deployment_infos: Dict[str, DeploymentInfo], override_config: Optional[ServeApplicationSchema]) -> Dict[str, DeploymentInfo]:
    if False:
        for i in range(10):
            print('nop')
    'Override deployment infos with options from app config.\n\n    Args:\n        app_name: application name\n        deployment_infos: deployment info loaded from code\n        override_config: application config deployed by user with\n            options to override those loaded from code.\n\n    Returns: the updated deployment infos.\n\n    Raises:\n        ValueError: If config options have invalid values.\n        TypeError: If config options have invalid types.\n    '
    deployment_infos = deepcopy(deployment_infos)
    if override_config is None:
        return deployment_infos
    config_dict = override_config.dict(exclude_unset=True)
    deployment_override_options = config_dict.get('deployments', [])
    for options in deployment_override_options:
        deployment_name = options['name']
        info = deployment_infos[deployment_name]
        if info.deployment_config.autoscaling_config is not None and info.deployment_config.max_concurrent_queries < info.deployment_config.autoscaling_config.target_num_ongoing_requests_per_replica:
            logger.warning("Autoscaling will never happen, because 'max_concurrent_queries' is less than 'target_num_ongoing_requests_per_replica' now.")
        override_options = dict()
        deployment_route_prefix = options.pop('route_prefix', DEFAULT.VALUE)
        if deployment_route_prefix is not DEFAULT.VALUE:
            override_options['route_prefix'] = deployment_route_prefix
        replica_config = info.replica_config
        app_runtime_env = override_config.runtime_env
        if 'ray_actor_options' in options:
            override_actor_options = options.pop('ray_actor_options', {})
        else:
            override_actor_options = replica_config.ray_actor_options or {}
        override_placement_group_bundles = options.pop('placement_group_bundles', replica_config.placement_group_bundles)
        override_placement_group_strategy = options.pop('placement_group_strategy', replica_config.placement_group_strategy)
        override_max_replicas_per_node = options.pop('max_replicas_per_node', replica_config.max_replicas_per_node)
        merged_env = override_runtime_envs_except_env_vars(app_runtime_env, override_actor_options.get('runtime_env', {}))
        override_actor_options.update({'runtime_env': merged_env})
        replica_config.update_ray_actor_options(override_actor_options)
        replica_config.update_placement_group_options(override_placement_group_bundles, override_placement_group_strategy)
        replica_config.update_max_replicas_per_node(override_max_replicas_per_node)
        override_options['replica_config'] = replica_config
        original_options = info.deployment_config.dict()
        options.pop('name', None)
        original_options.update(options)
        override_options['deployment_config'] = DeploymentConfig(**original_options)
        deployment_infos[deployment_name] = info.update(**override_options)
    app_route_prefix = config_dict.get('route_prefix', DEFAULT.VALUE)
    for deployment in list(deployment_infos.values()):
        if app_route_prefix is not DEFAULT.VALUE and deployment.route_prefix is not None:
            deployment.route_prefix = app_route_prefix
    return deployment_infos