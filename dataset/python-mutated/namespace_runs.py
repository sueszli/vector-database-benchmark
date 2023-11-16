"""API endpoint to manage runs.

Note: "run" is short for "interactive pipeline run".
"""
import uuid
from typing import Any, Dict, Optional
from celery.contrib.abortable import AbortableAsyncResult
from flask import abort, current_app, request
from flask_restx import Namespace, Resource, marshal
from sqlalchemy import nullslast
import app.models as models
from _orchest.internals.two_phase_executor import TwoPhaseExecutor, TwoPhaseFunction
from app import errors as self_errors
from app import schema
from app.connections import db
from app.core import environments, events
from app.core.pipelines import Pipeline, construct_pipeline
from app.utils import get_proj_pip_env_variables, update_status_db
api = Namespace('runs', description='Manages interactive pipeline runs')
api = schema.register_schema(api)

@api.route('/')
class RunList(Resource):

    @api.doc('get_runs')
    @api.marshal_with(schema.interactive_runs)
    def get(self):
        if False:
            print('Hello World!')
        'Fetches all (interactive) pipeline runs.\n\n        These pipeline runs are either pending, running or have already\n        completed. Runs are ordered by started time descending.\n        '
        query = models.InteractivePipelineRun.query
        if 'pipeline_uuid' in request.args and 'project_uuid' in request.args:
            query = query.filter_by(pipeline_uuid=request.args.get('pipeline_uuid')).filter_by(project_uuid=request.args.get('project_uuid'))
        elif 'project_uuid' in request.args:
            query = query.filter_by(project_uuid=request.args.get('project_uuid'))
        elif request.args['active'] == 'true':
            active_states = ['STARTED', 'PENDING']
            expression = models.InteractivePipelineRun.status.in_(active_states)
            query = query.filter(expression)
        runs = query.order_by(nullslast(models.PipelineRun.started_time.desc())).all()
        return ({'runs': [run.__dict__ for run in runs]}, 200)

    @api.doc('start_run')
    @api.expect(schema.interactive_run_spec)
    def post(self):
        if False:
            return 10
        'Starts a new (interactive) pipeline run.'
        post_data = request.get_json()
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                run = CreateInteractiveRun(tpe).transaction(post_data['project_uuid'], post_data['run_config'], construct_pipeline(**post_data))
        except Exception as e:
            return ({'message': str(e)}, 500)
        return (marshal(run, schema.interactive_run), 201)

@api.route('/<string:run_uuid>')
@api.param('run_uuid', 'UUID of Run')
@api.response(404, 'Run not found')
class Run(Resource):

    @api.doc('get_run')
    @api.marshal_with(schema.interactive_run, code=200)
    def get(self, run_uuid):
        if False:
            while True:
                i = 10
        'Fetches an interactive pipeline run given its UUID.'
        run = models.InteractivePipelineRun.query.filter_by(uuid=run_uuid).one_or_none()
        if run is None:
            abort(404, description='Run not found.')
        return run.__dict__

    @api.doc('delete_run')
    @api.response(200, 'Run terminated')
    def delete(self, run_uuid):
        if False:
            return 10
        'Stops a pipeline run given its UUID.'
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                could_abort = AbortInteractivePipelineRun(tpe).transaction(run_uuid)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if could_abort:
            return ({'message': 'Run termination was successful.'}, 200)
        else:
            return ({'message': 'Run does not exist or is not running.'}, 400)

@api.route('/<string:run_uuid>/<string:step_uuid>')
@api.param('run_uuid', 'UUID of Run')
@api.param('step_uuid', 'UUID of Pipeline Step')
@api.response(404, 'Pipeline step not found')
class StepStatus(Resource):

    @api.doc('get_step_status')
    @api.marshal_with(schema.pipeline_run_pipeline_step, code=200)
    def get(self, run_uuid, step_uuid):
        if False:
            while True:
                i = 10
        'Fetches the status of a pipeline step of a specific run.'
        step = models.PipelineRunStep.query.get_or_404(ident=(run_uuid, step_uuid), description='Run and step combination not found')
        return step.__dict__

class AbortPipelineRun(TwoPhaseFunction):
    """Stop a pipeline run.

    Sets its state in the db to ABORTED, revokes the celery task.
    """

    def _transaction(self, run_uuid):
        if False:
            print('Hello World!')
        'Abort a pipeline level at the db level.\n\n        Args:\n            run_uuid:\n\n        Returns:\n            True if the run state was set to ABORTED, false if the run\n            did not exist or was not PENDING/STARTED.\n        '
        filter_by = {'uuid': run_uuid}
        status_update = {'status': 'ABORTED'}
        can_abort = update_status_db(status_update, model=models.PipelineRun, filter_by=filter_by)
        if can_abort:
            filter_by = {'run_uuid': run_uuid}
            update_status_db(status_update, model=models.PipelineRunStep, filter_by=filter_by)
        self.collateral_kwargs['run_uuid'] = run_uuid if can_abort else None
        return can_abort

    def _collateral(self, run_uuid: Optional[str]):
        if False:
            return 10
        'Revoke the pipeline run celery task'
        if not run_uuid:
            return
        celery = current_app.config['CELERY']
        res = AbortableAsyncResult(run_uuid, app=celery)
        res.abort()
        celery.control.revoke(run_uuid)

class AbortInteractivePipelineRun(TwoPhaseFunction):
    """Aborts an interactive pipeline run."""

    def _transaction(self, run_uuid):
        if False:
            return 10
        could_abort = AbortPipelineRun(self.tpe).transaction(run_uuid)
        if not could_abort:
            return False
        run = models.PipelineRun.query.filter(models.PipelineRun.uuid == run_uuid).one()
        events.register_interactive_pipeline_run_cancelled(run.project_uuid, run.pipeline_uuid, run_uuid)
        return True

    def _collateral(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class CreateInteractiveRun(TwoPhaseFunction):

    def _transaction(self, project_uuid: str, run_config: Dict[str, Any], pipeline: Pipeline):
        if False:
            return 10
        task_id = str(uuid.uuid4())
        run = {'uuid': task_id, 'pipeline_uuid': pipeline.properties['uuid'], 'project_uuid': project_uuid, 'status': 'PENDING', 'pipeline_definition': pipeline.to_dict()}
        db.session.add(models.InteractivePipelineRun(**run))
        db.session.flush()
        step_uuids = [s.properties['uuid'] for s in pipeline.steps]
        pipeline_steps = []
        for step_uuid in step_uuids:
            pipeline_steps.append(models.PipelineRunStep(**{'run_uuid': task_id, 'step_uuid': step_uuid, 'status': 'PENDING'}))
        db.session.bulk_save_objects(pipeline_steps)
        run['pipeline_steps'] = pipeline_steps
        try:
            env_uuid_to_image = environments.lock_environment_images_for_run(task_id, project_uuid, pipeline.get_environments())
        except self_errors.PipelineDefinitionNotValid:
            msg = 'Please make sure every pipeline step is assigned an environment.'
            raise self_errors.PipelineDefinitionNotValid(msg)
        events.register_interactive_pipeline_run_created(project_uuid, pipeline.properties['uuid'], task_id)
        self.collateral_kwargs['project_uuid'] = project_uuid
        self.collateral_kwargs['task_id'] = task_id
        self.collateral_kwargs['pipeline'] = pipeline
        self.collateral_kwargs['run_config'] = run_config
        self.collateral_kwargs['env_variables'] = get_proj_pip_env_variables(project_uuid, pipeline.properties['uuid'])
        self.collateral_kwargs['env_uuid_to_image'] = env_uuid_to_image
        return run

    def _collateral(self, project_uuid: str, task_id: str, pipeline: Pipeline, run_config: Dict[str, Any], env_variables: Dict[str, Any], env_uuid_to_image: Dict[str, str], **kwargs):
        if False:
            print('Hello World!')
        celery = current_app.config['CELERY']
        run_config['env_uuid_to_image'] = env_uuid_to_image
        run_config['user_env_variables'] = env_variables
        run_config['session_uuid'] = project_uuid[:18] + pipeline.properties['uuid'][:18]
        run_config['session_type'] = 'interactive'
        celery_job_kwargs = {'pipeline_definition': pipeline.to_dict(), 'run_config': run_config, 'session_uuid': run_config['session_uuid']}
        res = celery.send_task('app.core.tasks.run_pipeline', kwargs=celery_job_kwargs, task_id=task_id)
        res.forget()

    def _revert(self):
        if False:
            return 10
        models.InteractivePipelineRun.query.filter_by(uuid=self.collateral_kwargs['task_id']).update({'status': 'FAILURE'})
        models.PipelineRunStep.query.filter_by(run_uuid=self.collateral_kwargs['task_id']).update({'status': 'FAILURE'})
        events.register_interactive_pipeline_run_failed(self.collateral_kwargs['project_uuid'], self.collateral_kwargs['pipeline'].properties['uuid'], self.collateral_kwargs['task_id'])
        db.session.commit()

class UpdateInteractivePipelineRun(TwoPhaseFunction):
    """Updates an interactive pipeline run."""

    def _transaction(self, run_uuid: str, status: str):
        if False:
            return 10
        filter_by = {'uuid': run_uuid}
        status_update = {'status': status}
        has_updated = update_status_db(status_update, model=models.PipelineRun, filter_by=filter_by)
        if has_updated:
            run = models.InteractivePipelineRun.query.filter(models.InteractivePipelineRun.uuid == run_uuid).one()
            if status_update['status'] == 'STARTED':
                events.register_interactive_pipeline_run_started(run.project_uuid, run.pipeline_uuid, run_uuid)
            elif status_update['status'] == 'FAILURE':
                events.register_interactive_pipeline_run_failed(run.project_uuid, run.pipeline_uuid, run_uuid)
            elif status_update['status'] == 'SUCCESS':
                events.register_interactive_pipeline_run_succeeded(run.project_uuid, run.pipeline_uuid, run_uuid)

    def _collateral(self):
        if False:
            while True:
                i = 10
        pass