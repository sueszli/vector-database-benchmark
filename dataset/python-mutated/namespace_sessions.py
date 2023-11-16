import os
from typing import Any, Dict
from flask import request
from flask.globals import current_app
from flask_restx import Namespace, Resource, marshal
from sqlalchemy import desc
import app.models as models
from _orchest.internals import config as _config
from _orchest.internals import utils as _utils
from _orchest.internals.two_phase_executor import TwoPhaseExecutor, TwoPhaseFunction
from app import errors as self_errors
from app import schema, utils
from app.apis.namespace_runs import AbortInteractivePipelineRun
from app.connections import db, k8s_core_api
from app.core import environments, events, pod_scheduling, sessions
from app.errors import JupyterEnvironmentBuildInProgressException
from app.types import InteractiveSessionConfig, SessionType
api = Namespace('sessions', description='Manage interactive sessions')
api = schema.register_schema(api)

@api.route('/')
class SessionList(Resource):

    @api.doc('fetch_sessions')
    @api.marshal_with(schema.sessions)
    def get(self):
        if False:
            return 10
        'Fetches all sessions.'
        query = models.InteractiveSession.query
        if 'pipeline_uuid' in request.args and 'project_uuid' in request.args:
            query = query.filter_by(pipeline_uuid=request.args.get('pipeline_uuid')).filter_by(project_uuid=request.args.get('project_uuid'))
        elif 'project_uuid' in request.args:
            query = query.filter_by(project_uuid=request.args.get('project_uuid'))
        sessions = query.all()
        return ({'sessions': [session.as_dict() for session in sessions]}, 200)

    @api.doc('launch_session')
    @api.expect(schema.session_config)
    def post(self):
        if False:
            while True:
                i = 10
        'Launches an interactive session.'
        session_config = request.get_json()
        isess = models.InteractiveSession.query.filter_by(project_uuid=session_config['project_uuid'], pipeline_uuid=session_config['pipeline_uuid']).one_or_none()
        if isess is not None:
            return ({'message': 'Session already exists.'}, 409)
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                CreateInteractiveSession(tpe).transaction(session_config)
        except JupyterEnvironmentBuildInProgressException:
            return ({'message': 'JupyterEnvironmentBuildInProgress'}, 423)
        except Exception as e:
            current_app.logger.error(e)
            return ({'message': str(e)}, 500)
        isess = models.InteractiveSession.query.filter_by(project_uuid=session_config['project_uuid'], pipeline_uuid=session_config['pipeline_uuid']).one_or_none()
        if isess is None:
            return ({'message': 'Could not start session.'}, 500)
        return (marshal(isess.as_dict(), schema.session), 201)

@api.route('/<string:project_uuid>/<string:pipeline_uuid>')
@api.param('project_uuid', 'UUID of project')
@api.param('pipeline_uuid', 'UUID of pipeline')
@api.response(404, 'Session not found')
class Session(Resource):
    """Manages interactive sessions.

    There can only be 1 interactive session per pipeline. Interactive
    sessions are uniquely identified by the pipeline's UUID.
    """

    @api.doc('get_session')
    @api.marshal_with(schema.session)
    def get(self, project_uuid, pipeline_uuid):
        if False:
            print('Hello World!')
        'Fetch a session given the pipeline UUID.'
        session = models.InteractiveSession.query.get_or_404(ident=(project_uuid, pipeline_uuid), description='Session not found.')
        return session.as_dict()

    @api.doc('shutdown_session')
    @api.response(200, 'Session stopped')
    @api.response(404, 'Session not found')
    def delete(self, project_uuid, pipeline_uuid):
        if False:
            return 10
        'Shutdowns session.'
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                could_shutdown = StopInteractiveSession(tpe).transaction(project_uuid, pipeline_uuid, async_mode=True)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if could_shutdown:
            return ({'message': 'Session shutdown was successful.'}, 200)
        else:
            return ({'message': 'Session not found.'}, 404)

@api.route('/<string:project_uuid>/<string:pipeline_uuid>/kernels')
@api.param('project_uuid', 'UUID of project')
@api.param('pipeline_uuid', 'UUID of pipeline')
@api.response(404, 'Session not found')
class SessionKernelList(Resource):
    """To create kernels to be used by Jupyter EG in a session."""

    @api.doc('create_kernel')
    @api.expect(schema.kernel_spec)
    def post(self, project_uuid, pipeline_uuid):
        if False:
            while True:
                i = 10
        'Lock and get the environment image to use for the kernel.'
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                LaunchKernel(tpe).transaction(project_uuid, pipeline_uuid, request.get_json())
        except Exception as e:
            current_app.logger.error(e)
            return ({'message': str(e)}, 500)
        return ({}, 201)

class CreateInteractiveSession(TwoPhaseFunction):

    def _transaction(self, session_config: InteractiveSessionConfig):
        if False:
            i = 10
            return i + 15
        latest_jupyter_image_build = models.JupyterImageBuild.query.order_by(desc(models.JupyterImageBuild.requested_time)).first()
        if latest_jupyter_image_build is not None and latest_jupyter_image_build.status in ['PENDING', 'STARTED']:
            raise JupyterEnvironmentBuildInProgressException()
        env_as_services = set()
        prefix = _config.ENVIRONMENT_AS_SERVICE_PREFIX
        for service in session_config.get('services', {}).values():
            img = service['image']
            if img.startswith(prefix):
                env_as_services.add(img.replace(prefix, ''))
        interactive_session = {'project_uuid': session_config['project_uuid'], 'pipeline_uuid': session_config['pipeline_uuid'], 'status': 'LAUNCHING', 'user_services': session_config.get('services', {})}
        db.session.add(models.InteractiveSession(**interactive_session))
        try:
            env_uuid_to_image = environments.lock_environment_images_for_interactive_session(session_config['project_uuid'], session_config['pipeline_uuid'], env_as_services)
            for (env_uuid, image) in env_uuid_to_image.items():
                env_uuid_to_image[env_uuid] = _config.ENVIRONMENT_IMAGE_NAME.format(project_uuid=session_config['project_uuid'], environment_uuid=env_uuid) + f':{image.tag}'
        except self_errors.ImageNotFound as e:
            raise self_errors.ImageNotFound(f'Pipeline services were referencing environments for which an image does not exist, {e}.')
        except self_errors.PipelineDefinitionNotValid:
            msg = 'Please make sure every pipeline step is assigned an environment.'
            raise self_errors.PipelineDefinitionNotValid(msg)
        session_config['env_uuid_to_image'] = env_uuid_to_image
        session_uuid = session_config['project_uuid'][:18] + session_config['pipeline_uuid'][:18]
        events._register_interactive_session_started(session_config['project_uuid'], session_config['pipeline_uuid'])
        self.collateral_kwargs['session_uuid'] = session_uuid
        self.collateral_kwargs['session_config'] = session_config

    @classmethod
    def _should_abort_session_start(cls, project_uuid, pipeline_uuid) -> bool:
        if False:
            print('Hello World!')
        session_entry = models.InteractiveSession.query.filter_by(project_uuid=project_uuid, pipeline_uuid=pipeline_uuid).one_or_none()
        return session_entry is None or session_entry.status != 'LAUNCHING'

    @classmethod
    def _background_session_start(cls, app, session_uuid: str, session_config: InteractiveSessionConfig):
        if False:
            return 10
        with app.app_context():
            try:
                project_uuid = session_config['project_uuid']
                pipeline_uuid = session_config['pipeline_uuid']
                sessions.launch(session_uuid, SessionType.INTERACTIVE, session_config, should_abort=lambda : cls._should_abort_session_start(project_uuid, pipeline_uuid))
                session_entry = models.InteractiveSession.query.with_for_update().populate_existing().filter_by(project_uuid=project_uuid, pipeline_uuid=pipeline_uuid).one_or_none()
                if session_entry is None:
                    return
                if session_entry.status == 'LAUNCHING':
                    session_entry.status = 'RUNNING'
                events._register_interactive_session_service_succeeded(project_uuid, pipeline_uuid)
                db.session.commit()
            except Exception as e:
                current_app.logger.error(e)
                db.session.commit()
                try:
                    sessions.cleanup_resources(session_uuid, wait_for_completion=True)
                    events._register_interactive_session_failed(project_uuid, pipeline_uuid)
                except Exception:
                    pass
                models.InteractiveSession.query.filter_by(project_uuid=project_uuid, pipeline_uuid=pipeline_uuid).delete()
                db.session.commit()

    def _collateral(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        current_app.config['SCHEDULER'].add_job(CreateInteractiveSession._background_session_start, args=[current_app._get_current_object(), *args], kwargs=kwargs)

class StopInteractiveSession(TwoPhaseFunction):

    def _transaction(self, project_uuid: str, pipeline_uuid: str, async_mode: bool=False):
        if False:
            print('Hello World!')
        session = models.InteractiveSession.query.with_for_update().populate_existing().filter_by(project_uuid=project_uuid, pipeline_uuid=pipeline_uuid).one_or_none()
        if session is None:
            self.collateral_kwargs['session_uuid'] = None
            self.collateral_kwargs['project_uuid'] = None
            self.collateral_kwargs['pipeline_uuid'] = None
            self.collateral_kwargs['async_mode'] = async_mode
            return False
        else:
            run = models.InteractivePipelineRun.query.filter(models.InteractivePipelineRun.project_uuid == project_uuid, models.InteractivePipelineRun.pipeline_uuid == pipeline_uuid, models.InteractivePipelineRun.status.in_(['PENDING', 'STARTED'])).one_or_none()
            if run is not None:
                AbortInteractivePipelineRun(self.tpe).transaction(run.uuid)
            session.status = 'STOPPING'
            session_uuid = project_uuid[:18] + pipeline_uuid[:18]
            self.collateral_kwargs['session_uuid'] = session_uuid
            self.collateral_kwargs['project_uuid'] = project_uuid
            self.collateral_kwargs['pipeline_uuid'] = pipeline_uuid
            self.collateral_kwargs['async_mode'] = async_mode
            events._register_interactive_session_stopped(project_uuid, pipeline_uuid)
        return True

    @classmethod
    def _session_stop(cls, app, session_uuid: str, project_uuid: str, pipeline_uuid: str):
        if False:
            for i in range(10):
                print('nop')
        with app.app_context():
            try:
                sessions.shutdown(session_uuid, wait_for_completion=True)
            finally:
                session = models.InteractiveSession.query.filter_by(project_uuid=project_uuid, pipeline_uuid=pipeline_uuid).one()
                db.session.delete(session)
                db.session.commit()

    def _collateral(self, session_uuid: str, project_uuid: str, pipeline_uuid: str, async_mode: bool):
        if False:
            i = 10
            return i + 15
        if project_uuid is None or pipeline_uuid is None:
            return
        if async_mode:
            current_app.config['SCHEDULER'].add_job(StopInteractiveSession._session_stop, args=[current_app._get_current_object(), session_uuid, project_uuid, pipeline_uuid])
        else:
            StopInteractiveSession._session_stop(current_app._get_current_object(), session_uuid, project_uuid, pipeline_uuid)

class LaunchKernel(TwoPhaseFunction):

    def _transaction(self, project_uuid: str, pipeline_uuid: str, kernel_spec: Dict[str, Any]) -> None:
        if False:
            while True:
                i = 10
        models.InteractiveSession.query.get_or_404(ident=(project_uuid, pipeline_uuid), description='Session not found.')
        (_, env_uuid, _) = _utils.env_image_name_to_proj_uuid_env_uuid_tag(kernel_spec['kernel_image'])
        env_image = environments.lock_environment_images_for_interactive_session(project_uuid, pipeline_uuid, set([env_uuid]))[env_uuid]
        registry_ip = utils.get_registry_ip()
        image_name = f'{registry_ip}/' + _config.ENVIRONMENT_IMAGE_NAME.format(project_uuid=project_uuid, environment_uuid=env_uuid) + f':{env_image.tag}'
        pod_manifest = {}
        kernel_id = kernel_spec['kernel_id']
        if kernel_spec.get('kernel_username') is None:
            name = f'kernel-{kernel_id}'
        else:
            kernel_username = kernel_spec['kernel_username']
            name = f'kernel-{kernel_username}-{kernel_id}'
        session_uuid = project_uuid[:18] + pipeline_uuid[:18]
        metadata = {'name': name, 'labels': {'project_uuid': project_uuid, 'session_uuid': session_uuid, 'kernel_id': kernel_id, 'component': 'kernel', 'app': 'enterprise-gateway'}}
        (vols, vol_mounts) = _utils.get_step_and_kernel_volumes_and_volume_mounts(userdir_pvc='userdir-pvc', project_dir=kernel_spec['project_dir'], pipeline_file=kernel_spec['pipeline_file'], container_project_dir=_config.PROJECT_DIR, container_pipeline_file=_config.PIPELINE_FILE, container_runtime_socket=_config.CONTAINER_RUNTIME_SOCKET)
        environment = {'ORCHEST_PROJECT_UUID': project_uuid, 'ORCHEST_PROJECT_DIR': kernel_spec['project_dir'], 'ORCHEST_PIPELINE_UUID': pipeline_uuid, 'ORCHEST_PIPELINE_FILE': kernel_spec['pipeline_file'], 'ORCHEST_PIPELINE_PATH': kernel_spec['pipeline_path'], 'ORCHEST_HOST_GID': os.environ.get('ORCHEST_HOST_GID'), 'ORCHEST_SESSION_UUID': session_uuid, 'ORCHEST_SESSION_TYPE': 'interactive', 'ORCHEST_GPU_ENABLED_INSTANCE': 'False', 'ORCHEST_CLUSTER': _config.ORCHEST_CLUSTER, 'ORCHEST_NAMESPACE': _config.ORCHEST_NAMESPACE, 'KERNEL_ID': kernel_id}
        environment['EG_RESPONSE_ADDRESS'] = kernel_spec['eg_response_address']
        if kernel_spec.get('spark_context_init_mode') is not None:
            environment['KERNEL_SPARK_CONTEXT_INIT_MODE'] = kernel_spec['spark_context_init_mode']
        environment.update(utils.get_proj_pip_env_variables(project_uuid, pipeline_uuid))
        environment.pop('PATH', None)
        env = [{'name': k, 'value': v} for (k, v) in environment.items()]
        pod_manifest = {'apiVersion': 'v1', 'kind': 'Pod', 'metadata': metadata, 'spec': {'securityContext': {'runAsUser': 0, 'runAsGroup': int(os.environ.get('ORCHEST_HOST_GID')), 'fsGroup': int(os.environ.get('ORCHEST_HOST_GID'))}, 'restartPolicy': 'Never', 'volumes': vols, 'containers': [{'name': name, 'image': image_name, 'env': env, 'ports': [{'name': 'web', 'containerPort': 80, 'protocol': 'TCP'}], 'volumeMounts': vol_mounts}], 'resources': {'requests': {'cpu': _config.USER_CONTAINERS_CPU_SHARES}}}}
        if kernel_spec['kernel_working_dir'] is not None:
            pod_manifest['spec']['containers'][0]['workingDir'] = kernel_spec['kernel_working_dir']
        pod_scheduling.modify_kernel_scheduling_behaviour(pod_manifest)
        self.collateral_kwargs['pod_manifest'] = pod_manifest

    def _collateral(self, pod_manifest: Dict[str, Any]):
        if False:
            for i in range(10):
                print('nop')
        ns = _config.ORCHEST_NAMESPACE
        k8s_core_api.create_namespaced_pod(ns, pod_manifest)