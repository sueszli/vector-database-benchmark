import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Set, Tuple
import requests
from celery.contrib.abortable import AbortableAsyncResult
from croniter import croniter
from flask import abort, current_app, request
from flask_restx import Namespace, Resource, marshal, reqparse
from sqlalchemy import asc, desc, func, or_, tuple_
from sqlalchemy.orm import attributes, joinedload, load_only, noload, undefer
import app.models as models
from _orchest.internals import config as _config
from _orchest.internals import utils as _utils
from _orchest.internals.two_phase_executor import TwoPhaseExecutor, TwoPhaseFunction
from app import errors, schema
from app import types as app_types
from app.apis.namespace_runs import AbortPipelineRun
from app.connections import db
from app.core import environments, events
from app.core.pipelines import Pipeline, construct_pipeline
from app.utils import fuzzy_filter_non_interactive_pipeline_runs, get_env_vars_update, get_proj_pip_env_variables, page_to_pagination_data, update_status_db
api = Namespace('jobs', description='Managing jobs')
api = schema.register_schema(api)

@api.route('/')
class JobList(Resource):

    @api.doc('get_jobs')
    @api.marshal_with(schema.jobs)
    def get(self):
        if False:
            return 10
        'Fetches all jobs.\n\n        The jobs are either in queue, running or already\n        completed.\n\n        '
        jobs = models.Job.query
        if 'project_uuid' in request.args:
            jobs = jobs.filter_by(project_uuid=request.args['project_uuid'])
        if 'active' in request.args:
            should_select_active = request.args['active'] == 'true'
            active_states = ['STARTED', 'PENDING']
            expression = models.Job.status.in_(active_states) if should_select_active else models.Job.status.not_in(active_states)
            jobs = jobs.filter(expression)
        jobs = jobs.order_by(desc(models.Job.created_time)).all()
        jobs = [job.__dict__ for job in jobs]
        return {'jobs': jobs}

    @api.doc('start_job')
    @api.expect(schema.job_spec)
    def post(self):
        if False:
            print('Hello World!')
        'Drafts a new job. Locks environment images for all its runs.\n\n        The environment images used by a job across its entire lifetime,\n        and thus its runs, will be the same. This is done by locking the\n        actual resource (image) that is backing the environment, so that\n        a new build of the environment will not affect the job.  To\n        actually queue the job you need to issue a PUT request for the\n        DRAFT job you create here. The PUT needs to contain the\n        `confirm_draft` key.\n\n        '
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                job = CreateJob(tpe).transaction(request.get_json())
        except errors.PipelineDefinitionNotValid as e:
            return ({'message': str(e)}, 409)
        except errors.PipelinesHaveInvalidEnvironments as e:
            return ({'message': str(e), 'invalid_pipelines': e.uuids}, 409)
        except Exception as e:
            current_app.logger.error(e)
            return ({'message': str(e)}, 500)
        return (marshal(job, schema.job), 201)

@api.route('/next_scheduled_job')
class NextScheduledJob(Resource):

    @api.doc('get_next_scheduled_job')
    @api.marshal_with(schema.next_scheduled_job_data)
    def get(self):
        if False:
            i = 10
            return i + 15
        'Returns data about the next job to be scheduled.'
        next_job = models.Job.query.options(load_only('uuid', 'next_scheduled_time'))
        if 'project_uuid' in request.args:
            next_job = next_job.filter_by(project_uuid=request.args['project_uuid'])
        next_job = next_job.filter(models.Job.status.in_(['PENDING', 'STARTED'])).filter(models.Job.next_scheduled_time.isnot(None)).order_by(models.Job.next_scheduled_time).first()
        data = {'uuid': None, 'next_scheduled_time': None}
        if next_job is not None:
            data['uuid'] = next_job.uuid
            data['next_scheduled_time'] = next_job.next_scheduled_time
        return data

@api.route('/<string:job_uuid>')
@api.param('job_uuid', 'UUID of job')
@api.response(404, 'Job not found')
class Job(Resource):

    @api.doc('get_job', params={'aggregate_run_statuses': {'description': 'Aggregate job pipeline run statuses. Populates the pipeline_run_status_counts property. Value does not matter as long as it is set.', 'type': None}})
    @api.marshal_with(schema.job, code=200)
    def get(self, job_uuid):
        if False:
            print('Hello World!')
        'Fetches a job given its UUID.'
        job = models.Job.query.options(undefer(models.Job.env_variables)).filter_by(uuid=job_uuid).one_or_none()
        if job is None:
            abort(404, 'Job not found.')
        if 'aggregate_run_statuses' in request.args:
            status_agg = models.NonInteractivePipelineRun.query.filter_by(job_uuid=job_uuid).with_entities(models.NonInteractivePipelineRun.status, func.count(models.NonInteractivePipelineRun.status)).group_by(models.NonInteractivePipelineRun.status)
            status_agg = {k: v for (k, v) in status_agg}
            job = job.__dict__
            job['pipeline_run_status_counts'] = status_agg
        return job

    @api.expect(schema.job_parameters_update)
    @api.doc('update_job_parameters')
    def put(self, job_uuid):
        if False:
            print('Hello World!')
        'Updates a job parameters.\n\n        Update a job parameters. Updating the cron\n        schedule implies that the job will be rescheduled and will\n        follow the new given schedule. Updating the parameters of a job\n        implies that the next time the job will be run those parameters\n        will be used, thus affecting the number of pipeline runs that\n        are launched. Only recurring ongoing jobs can be updated.\n\n        '
        job_update = request.get_json()
        name = job_update.get('name')
        cron_schedule = job_update.get('cron_schedule')
        parameters = job_update.get('parameters')
        env_variables = job_update.get('env_variables')
        next_scheduled_time = job_update.get('next_scheduled_time')
        strategy_json = job_update.get('strategy_json')
        max_retained_pipeline_runs = job_update.get('max_retained_pipeline_runs')
        confirm_draft = 'confirm_draft' in job_update
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                UpdateJobParameters(tpe).transaction(job_uuid, name, cron_schedule, parameters, env_variables, next_scheduled_time, strategy_json, max_retained_pipeline_runs, confirm_draft)
        except Exception as e:
            current_app.logger.error(e)
            db.session.rollback()
            return ({'message': str(e)}, 500)
        return ({'message': 'Job was updated successfully'}, 200)

    @api.doc('delete_job')
    @api.response(200, 'Job terminated')
    def delete(self, job_uuid):
        if False:
            i = 10
            return i + 15
        'Stops a job given its UUID.\n\n        However, it will not delete any corresponding database entries,\n        it will update the status of corresponding objects to "ABORTED".\n        '
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                could_abort = AbortJob(tpe).transaction(job_uuid)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if could_abort:
            return ({'message': 'Job termination was successful.'}, 200)
        else:
            return ({'message': 'Job does not exist or is already completed.'}, 404)

@api.route('/<string:job_uuid>/pipeline')
@api.param('job_uuid', 'UUID of job')
@api.response(404, 'Job not found')
class DraftJobPipelineUpdate(Resource):

    @api.expect(schema.draft_job_pipeline_update)
    @api.doc('update_job')
    def put(self, job_uuid):
        if False:
            return 10
        'Changes a DRAFT job pipeline.\n\n        This can be used to allow users to change the pipeline of a\n        DRAFT job while "deciding" what pipeline to use. The reason\n        is that a job draft (and the project snapshot) is created\n        first, and the user will be allowed to change the pipeline\n        later.\n\n        '
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                UpdateDraftJobPipeline(tpe).transaction(job_uuid, request.get_json()['pipeline_uuid'])
        except ValueError as e:
            return ({'message': str(e), 'type': 'VALUE_ERROR'}, 409)
        except errors.ImageNotFound as e:
            return ({'message': str(e), 'type': 'IMAGE_NOT_FOUND'}, 409)
        except errors.PipelineDefinitionNotValid as e:
            return ({'message': str(e), 'type': 'PIPELINE_DEFINITION_NOT_VALID'}, 409)
        except Exception as e:
            current_app.logger.error(e)
            return ({'message': str(e), 'type': 'EXCEPTION'}, 500)
        return ({'message': 'Job pipeline was updated successfully'}, 200)

@api.route('/pipeline_runs', doc={'description': 'Retrieve list of job pipeline runs.'})
class PipelineRunsList(Resource):

    @api.doc('get_job_pipeline_runs', params={'page': {'description': 'Which page to query, 1 indexed. Must be specified if page_size is specified.', 'type': int}, 'page_size': {'description': 'Size of the page. Must be specified if page is specified.', 'type': int}, 'fuzzy_filter': {'description': 'Fuzzy filtering across pipeline run index, status and parameters.', 'type': str}, 'project_uuid__in': {'description': 'Comma separated uuids.', 'type': str}, 'project_pipeline_uuid__in': {'description': "Comma separated uuids, where the Nth uuid is a project uuid and the Nth+1 uuid is a pipeline uuid for all even Ns including 0, e.g. [proj_uuid, ppl_uuid, proj_uuid, ppl_uuid, ...]. This is necessary because the pipeline uuid is not unique making across projects. Note that, while all filters are AND'd, this particular filter makes an OR with the project_uuid__in one. Meaning that, for example, if you have a status__in, project_uuid__in and project_pipeline_uuid__in filter, you will get the records respecting a constraint like AND(status_in(), OR(project_in(), proj_ppl_in()). This allows for filtering runs that are part of a project OR of a particular pipeline.", 'type': str}, 'job_uuid__in': {'description': 'Comma separated uuids.', 'type': str}, 'status__in': {'description': 'Comma separated.', 'type': str}, 'created_time__gt': {'description': 'String representing a timestamp in ISOFORMAT, UTC, timezone not necessary.', 'type': str}, 'sort': {'description': "Either 'oldest' or or 'newest'. Default is 'newest'.", type: str}})
    @api.response(200, 'Success', schema.paginated_job_pipeline_runs)
    @api.response(200, 'Success', schema.job_pipeline_runs)
    def get(self):
        if False:
            while True:
                i = 10
        'Fetch pipeline runs of jobs, sorted newest first.\n\n        Runs are ordered by created_time DESC, job_run_index DESC,\n        job_run_pipeline_run_index DESC.\n\n        The endpoint has optional pagination. If pagination is used the\n        returned json also contains pagination data.\n        '
        parser = reqparse.RequestParser()
        parser.add_argument('page', type=int, location='args')
        parser.add_argument('page_size', type=int, location='args')
        parser.add_argument('fuzzy_filter', type=str, location='args')
        parser.add_argument('project_uuid__in', type=str, action='split')
        parser.add_argument('project_pipeline_uuid__in', type=str, action='split')
        parser.add_argument('job_uuid__in', type=str, action='split')
        parser.add_argument('status__in', type=str, action='split')
        parser.add_argument('created_time__gt', type=str)
        parser.add_argument('sort', type=str)
        args = parser.parse_args()
        page = args.page
        page_size = args.page_size
        project_uuids = args.project_uuid__in
        project_pipeline_uuids = args.project_pipeline_uuid__in
        job_uuids = args.job_uuid__in
        statuses = args.status__in
        created_time__gt = args.created_time__gt
        sort = args.sort
        if project_pipeline_uuids is not None and len(project_pipeline_uuids) % 2 != 0:
            return ({'message': f'Invalid project_pipeline_uuids {project_pipeline_uuids}.'}, 400)
        if project_pipeline_uuids is not None:
            project_pipeline_uuids = [(project_pipeline_uuids[i], project_pipeline_uuids[i + 1]) for i in range(0, len(project_pipeline_uuids), 2)]
        if created_time__gt is not None:
            try:
                created_time__gt = datetime.fromisoformat(created_time__gt)
            except ValueError:
                return ({'message': 'Invalid created_time__gt, must be iso format.'}, 400)
        if page is not None and page_size is None or (page is None and page_size is not None):
            return ({'message': 'Either both page and page_size are defined or none of them.'}, 400)
        if page is not None and page <= 0:
            return ({'message': 'page must be >= 1.'}, 400)
        if page_size is not None and page_size <= 0:
            return ({'message': 'page_size must be >= 1.'}, 400)
        job_runs_query = models.NonInteractivePipelineRun.query.options(noload(models.NonInteractivePipelineRun.pipeline_steps), undefer(models.NonInteractivePipelineRun.env_variables))
        if project_uuids is not None or project_pipeline_uuids is not None:
            exp = None
            if project_uuids is not None:
                exp = models.NonInteractivePipelineRun.project_uuid.in_(project_uuids)
            if project_pipeline_uuids is not None:
                exp2 = tuple_(models.NonInteractivePipelineRun.project_uuid, models.NonInteractivePipelineRun.pipeline_uuid).in_(project_pipeline_uuids)
                exp = exp2 if exp is None else or_(exp, exp2)
            job_runs_query = job_runs_query.filter(exp)
        if job_uuids is not None:
            job_runs_query = job_runs_query.filter(models.NonInteractivePipelineRun.job_uuid.in_(job_uuids))
        if statuses is not None:
            job_runs_query = job_runs_query.filter(models.NonInteractivePipelineRun.status.in_(statuses))
        if created_time__gt is not None:
            job_runs_query = job_runs_query.filter(models.NonInteractivePipelineRun.created_time > created_time__gt)
        if args.fuzzy_filter is not None:
            job_runs_query = fuzzy_filter_non_interactive_pipeline_runs(job_runs_query, args.fuzzy_filter)
        if sort == 'oldest':
            job_runs_query = job_runs_query.order_by(asc(models.NonInteractivePipelineRun.started_time), asc(models.NonInteractivePipelineRun.job_run_index), asc(models.NonInteractivePipelineRun.job_run_pipeline_run_index))
        else:
            job_runs_query = job_runs_query.order_by(desc(models.NonInteractivePipelineRun.started_time), desc(models.NonInteractivePipelineRun.job_run_index), desc(models.NonInteractivePipelineRun.job_run_pipeline_run_index))
        if args.page is not None and args.page_size is not None:
            job_runs_pagination = job_runs_query.paginate(args.page, args.page_size, False)
            job_runs = job_runs_pagination.items
            pagination_data = page_to_pagination_data(job_runs_pagination)
            return (marshal({'pipeline_runs': job_runs, 'pagination_data': pagination_data}, schema.paginated_job_pipeline_runs), 200)
        else:
            job_runs = job_runs_query.all()
            return (marshal({'pipeline_runs': job_runs}, schema.job_pipeline_runs), 200)

@api.route('/<string:job_uuid>/<string:run_uuid>', doc={'description': 'Set and get execution status of pipeline runs in a job. Also allows to abort a specific pipeline run.'})
@api.param('job_uuid', 'UUID of Job')
@api.param('run_uuid', 'UUID of Run')
@api.response(404, 'Pipeline run not found')
class PipelineRun(Resource):

    @api.doc('get_pipeline_run')
    @api.marshal_with(schema.non_interactive_run, code=200)
    def get(self, job_uuid, run_uuid):
        if False:
            while True:
                i = 10
        'Fetch a pipeline run of a job given their ids.'
        non_interactive_run = models.NonInteractivePipelineRun.query.options(undefer(models.NonInteractivePipelineRun.env_variables)).filter_by(uuid=run_uuid).one_or_none()
        if non_interactive_run is None:
            abort(404, 'Given job has no run with given run_uuid')
        return non_interactive_run.__dict__

    @api.doc('delete_run')
    @api.response(200, 'Run terminated')
    def delete(self, job_uuid, run_uuid):
        if False:
            print('Hello World!')
        'Stops a job pipeline run given its UUID.'
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                could_abort = AbortJobPipelineRun(tpe).transaction(job_uuid, run_uuid)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if could_abort:
            return ({'message': 'Run termination was successful.'}, 200)
        else:
            return ({'message': 'Run does not exist or is not running.'}, 404)

@api.route('/<string:job_uuid>/<string:run_uuid>/<string:step_uuid>', doc={'description': 'Set and get execution status of individual steps of pipeline runs in a job.'})
@api.param('job_uuid', 'UUID of Job')
@api.param('run_uuid', 'UUID of Run')
@api.param('step_uuid', 'UUID of Step')
@api.response(404, 'Pipeline step not found')
class PipelineStepStatus(Resource):

    @api.doc('get_pipeline_run_pipeline_step')
    @api.marshal_with(schema.non_interactive_run, code=200)
    def get(self, job_uuid, run_uuid, step_uuid):
        if False:
            for i in range(10):
                print('nop')
        'Fetch a pipeline step of a job run given uuids.'
        step = models.PipelineRunStep.query.get_or_404(ident=(run_uuid, step_uuid), description='Combination of given job, run and step not found')
        return step.__dict__

@api.route('/cleanup/<string:job_uuid>')
@api.param('job_uuid', 'UUID of job')
@api.response(404, 'Job not found')
class JobDeletion(Resource):

    @api.doc('delete_job')
    @api.response(200, 'Job deleted')
    def delete(self, job_uuid):
        if False:
            print('Hello World!')
        'Delete a job.\n\n        The job is stopped if its running, related entities\n        are then removed from the db.\n        '
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                could_delete = DeleteJob(tpe).transaction(job_uuid)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if could_delete:
            return ({'message': 'Job deletion was successful.'}, 200)
        else:
            return ({'message': 'Job does not exist.'}, 404)

@api.route('/cleanup/<string:job_uuid>/<string:run_uuid>')
@api.param('job_uuid', 'UUID of job')
@api.param('run_uuid', 'UUID of pipeline run')
@api.response(404, 'Job pipeline run not found')
class JobPipelineRunDeletion(Resource):

    @api.doc('delete_job_pipeline_run')
    @api.response(200, 'Job pipeline run deleted')
    def delete(self, job_uuid, run_uuid):
        if False:
            for i in range(10):
                print('nop')
        'Delete a job pipeline run.\n\n        The pipeline run is stopped if its running, related entities are\n        then removed from the db.\n        '
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                could_delete = DeleteJobPipelineRun(tpe).transaction(job_uuid, run_uuid)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if could_delete:
            return ({'message': 'Job pipelune run deletion was successful.'}, 200)
        else:
            return ({'message': 'Job pipeline run does not exist.'}, 404)

@api.route('/cronjobs/pause/<string:job_uuid>')
@api.param('job_uuid', 'UUID of job')
@api.response(404, 'Job not found')
class CronJobPause(Resource):

    @api.doc('pause_cronjob')
    @api.response(200, 'Cron job paused')
    def post(self, job_uuid):
        if False:
            while True:
                i = 10
        'Pauses a cron job.'
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                could_pause = PauseCronJob(tpe).transaction(job_uuid)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if could_pause:
            return ({'message': 'Cron job pausing was successful.'}, 200)
        else:
            return ({'message': 'Could not pause cron job.'}, 409)

@api.route('/cronjobs/resume/<string:job_uuid>')
@api.param('job_uuid', 'UUID of job')
@api.response(404, 'Job not found')
class CronJobResume(Resource):

    @api.doc('resume_cronjob')
    @api.response(200, 'Cron job resumed')
    def post(self, job_uuid):
        if False:
            for i in range(10):
                print('nop')
        'Resumes a cron job.'
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                next_scheduled_time = ResumeCronJob(tpe).transaction(job_uuid)
        except Exception as e:
            return ({'message': str(e)}, 500)
        if next_scheduled_time is not None:
            return ({'next_scheduled_time': next_scheduled_time}, 200)
        else:
            return ({'message': 'Could not resume cron job.'}, 409)

@api.route('/<string:job_uuid>/runs/trigger')
@api.param('job_uuid', 'UUID of job')
@api.response(404, 'Job not found')
@api.response(409, 'Job is not in a state which allows triggering a run.')
class JobRunTrigger(Resource):

    @api.doc('trigger_job_run')
    @api.response(200, 'Job run triggered')
    def post(self, job_uuid: str):
        if False:
            print('Hello World!')
        'Triggers a batch of runs for a non end state job.\n\n        The job should either be a PENDING|STARTED cronjob or a PENDING\n        one-off job scheduled in the future for a batch of runs to be\n        triggered. With batch of runs we mean a number of runs equal to\n        the "pipeline runs" that would be setup through the job\n        parameterization. For example, for a cronjob, this would be as\n        if the job run was performed because of its scheduled time.\n\n\n        '
        job = models.Job.query.get_or_404(ident=job_uuid, description='Job not found.')
        if not (job.schedule is not None and job.status in ['STARTED', 'PAUSED'] or (job.schedule is None and job.next_scheduled_time is not None and (job.status == 'PENDING'))):
            return ({'message': 'The job is not in a state which allows triggering a run.'}, 409)
        try:
            with TwoPhaseExecutor(db.session) as tpe:
                RunJob(tpe).transaction(job_uuid, triggered_by_user=True)
        except Exception as e:
            return ({'message': str(e)}, 500)
        return ({}, 200)

class DeleteNonRetainedJobPipelineRuns(TwoPhaseFunction):
    """See max_retained_pipeline_runs in models.py for docs."""

    def _transaction(self, job_uuid: str):
        if False:
            while True:
                i = 10
        job = db.session.query(models.Job.project_uuid, models.Job.pipeline_uuid, models.Job.max_retained_pipeline_runs, models.Job.total_scheduled_pipeline_runs).filter_by(uuid=job_uuid).one()
        self.collateral_kwargs['project_uuid'] = job.project_uuid
        self.collateral_kwargs['pipeline_uuid'] = job.pipeline_uuid
        self.collateral_kwargs['job_uuid'] = job_uuid
        self.collateral_kwargs['pipeline_run_uuids'] = []
        max_retained_pipeline_runs = job.max_retained_pipeline_runs
        current_app.logger.info(f'Deleting non retained runs for job {job_uuid}, max retained pipeline runs: {max_retained_pipeline_runs}.')
        if max_retained_pipeline_runs == -1:
            current_app.logger.info('Nothing to do.')
            return
        runs_to_be_deleted = db.session.query(models.NonInteractivePipelineRun.uuid).filter(models.NonInteractivePipelineRun.job_uuid == job_uuid, models.NonInteractivePipelineRun.status.in_(['SUCCESS', 'FAILURE', 'ABORTED']), models.NonInteractivePipelineRun.pipeline_run_index <= job.total_scheduled_pipeline_runs - 1 - max_retained_pipeline_runs).all()
        for run in runs_to_be_deleted:
            current_app.logger.info(f'Issuing deletion of run {run.uuid}.')
            self.collateral_kwargs['pipeline_run_uuids'].append(run.uuid)
        batch_size = 500
        for i in range(0, len(runs_to_be_deleted), batch_size):
            batch = runs_to_be_deleted[i:i + batch_size]
            batch_uuids = [run.uuid for run in batch]
            models.NonInteractivePipelineRun.query.filter(models.NonInteractivePipelineRun.uuid.in_(batch_uuids)).delete()

    def _collateral(self, project_uuid: str, pipeline_uuid: str, job_uuid: str, pipeline_run_uuids: List[str]):
        if False:
            return 10
        celery = current_app.config['CELERY']
        batch_size = 30
        for i in range(0, len(pipeline_run_uuids), batch_size):
            batch = pipeline_run_uuids[i:i + batch_size]
            celery_job_kwargs = {'project_uuid': project_uuid, 'pipeline_uuid': pipeline_uuid, 'job_uuid': job_uuid, 'pipeline_run_uuids': batch}
            task_args = {'name': 'app.core.tasks.delete_job_pipeline_run_directories', 'kwargs': celery_job_kwargs, 'task_id': str(uuid.uuid4())}
            res = celery.send_task(**task_args)
            res.forget()

def _delete_non_retained_pipeline_runs(job_uuid: str) -> None:
    if False:
        while True:
            i = 10
    job = db.session.query(models.Job.max_retained_pipeline_runs, models.Job.total_scheduled_pipeline_runs).filter_by(uuid=job_uuid).one()
    max_retained_pipeline_runs = job.max_retained_pipeline_runs
    current_app.logger.info(f'Deleting non retained runs for job {job_uuid}, max retained pipeline runs: {max_retained_pipeline_runs}.')
    if max_retained_pipeline_runs < 0:
        current_app.logger.info('Nothing to do.')
        return
    runs_to_be_deleted = db.session.query(models.NonInteractivePipelineRun.uuid).filter(models.NonInteractivePipelineRun.job_uuid == job_uuid, models.NonInteractivePipelineRun.status.in_(['SUCCESS', 'FAILURE', 'ABORTED']), models.NonInteractivePipelineRun.pipeline_run_index <= job.total_scheduled_pipeline_runs - 1 - max_retained_pipeline_runs).all()
    for run in runs_to_be_deleted:
        current_app.logger.info(f'Deleting run {run.uuid}.')
        path = f'/catch/api-proxy/api/jobs/cleanup/{job_uuid}/{run.uuid}'
        base_url = f"{current_app.config['ORCHEST_WEBSERVER_ADDRESS']}{path}"
        resp = requests.delete(base_url)
        if resp.status_code not in [200, 404]:
            current_app.logger.error(f'Unexpected status code ({resp.status_code}) while deleting run {run.uuid}.')
        else:
            current_app.logger.info(f'Successfully deleted run {run.uuid}.')

class RunJob(TwoPhaseFunction):
    """Start the pipeline runs related to a job"""

    def _transaction(self, job_uuid: str, triggered_by_user: bool=False):
        if False:
            for i in range(10):
                print('nop')
        job = models.Job.query.with_entities(models.Job).with_for_update().filter_by(uuid=job_uuid).one()
        if job.status == 'ABORTED':
            self.collateral_kwargs['job'] = dict()
            self.collateral_kwargs['tasks_to_launch'] = []
            self.collateral_kwargs['run_config'] = dict()
        if job.status == 'PENDING' and (not triggered_by_user):
            job.status = 'STARTED'
            events.register_job_started(job.project_uuid, job.uuid)
        if job.schedule is not None:
            events.register_cron_job_run_started(job.project_uuid, job.uuid, job.total_scheduled_executions)
        tasks_to_launch = []
        for (run_index, run_parameters) in enumerate(job.parameters):
            pipeline_def = copy.deepcopy(job.pipeline_definition)
            pipeline_def['parameters'] = run_parameters.get(_config.PIPELINE_PARAMETERS_RESERVED_KEY, {})
            for (step_uuid, step_parameters) in run_parameters.items():
                if step_uuid != _config.PIPELINE_PARAMETERS_RESERVED_KEY:
                    pipeline_def['steps'][step_uuid]['parameters'] = step_parameters
            pipeline_run_spec = copy.deepcopy(job.pipeline_run_spec)
            pipeline_run_spec['pipeline_definition'] = pipeline_def
            pipeline = construct_pipeline(**pipeline_run_spec)
            task_id = str(uuid.uuid4())
            tasks_to_launch.append((task_id, pipeline))
            non_interactive_run = {'job_uuid': job.uuid, 'uuid': task_id, 'pipeline_uuid': job.pipeline_uuid, 'project_uuid': job.project_uuid, 'status': 'PENDING', 'parameters': run_parameters, 'parameters_text_search_values': list(run_parameters.values()), 'job_run_index': job.total_scheduled_executions, 'job_run_pipeline_run_index': run_index, 'pipeline_run_index': job.total_scheduled_pipeline_runs, 'env_variables': job.env_variables}
            job.total_scheduled_pipeline_runs += 1
            db.session.add(models.NonInteractivePipelineRun(**non_interactive_run))
            db.session.flush()
            events.register_job_pipeline_run_created(job.project_uuid, job.uuid, task_id)
            step_uuids = [s.properties['uuid'] for s in pipeline.steps]
            pipeline_steps = []
            for step_uuid in step_uuids:
                pipeline_steps.append(models.PipelineRunStep(**{'run_uuid': task_id, 'step_uuid': step_uuid, 'status': 'PENDING'}))
            db.session.bulk_save_objects(pipeline_steps)
        job.total_scheduled_executions += 1
        DeleteNonRetainedJobPipelineRuns(self.tpe).transaction(job.uuid)
        self.collateral_kwargs['job'] = job.as_dict()
        env_uuid_to_image = {}
        for a in job.images_in_use:
            env_uuid_to_image[a.environment_uuid] = _config.ENVIRONMENT_IMAGE_NAME.format(project_uuid=a.project_uuid, environment_uuid=a.environment_uuid) + f':{a.environment_image_tag}'
        run_config = job.pipeline_run_spec['run_config']
        run_config['env_uuid_to_image'] = env_uuid_to_image
        run_config['user_env_variables'] = job.env_variables
        self.collateral_kwargs['run_config'] = run_config
        self.collateral_kwargs['tasks_to_launch'] = tasks_to_launch

    def _collateral(self, job: Dict[str, Any], run_config: Dict[str, Any], tasks_to_launch: Tuple[str, Pipeline]):
        if False:
            return 10
        if not tasks_to_launch:
            return
        celery = current_app.config['CELERY']
        for (task_id, pipeline) in tasks_to_launch:
            celery_job_kwargs = {'job_uuid': job['uuid'], 'project_uuid': job['project_uuid'], 'pipeline_definition': pipeline.to_dict(), 'run_config': run_config}
            task_args = {'name': 'app.core.tasks.start_non_interactive_pipeline_run', 'kwargs': celery_job_kwargs, 'task_id': task_id}
            res = celery.send_task(**task_args)
            res.forget()

    def _revert(self):
        if False:
            return 10
        job = self.collateral_kwargs['job']
        if job['schedule'] is None:
            models.Job.query.filter_by(uuid=job['uuid']).update({'status': 'FAILURE'})
            events.register_job_failed(job['project_uuid'], job['uuid'])
        tasks_ids = [task[0] for task in self.collateral_kwargs['tasks_to_launch']]
        models.PipelineRunStep.query.filter(models.PipelineRunStep.run_uuid.in_(tasks_ids)).update({'status': 'FAILURE'}, synchronize_session=False)
        for task_id in tasks_ids:
            events.register_job_pipeline_run_failed(job['project_uuid'], job['uuid'], task_id)
        if job.get('schedule') is not None:
            events.register_cron_job_run_failed(job['project_uuid'], job['uuid'], job['total_scheduled_executions'] - 1)
        models.NonInteractivePipelineRun.query.filter(models.PipelineRun.uuid.in_(tasks_ids)).update({'status': 'FAILURE'}, synchronize_session=False)
        db.session.commit()

class AbortJob(TwoPhaseFunction):
    """Abort a job."""

    def _transaction(self, job_uuid: str):
        if False:
            for i in range(10):
                print('nop')
        run_uuids = []
        self.collateral_kwargs['run_uuids'] = run_uuids
        self.collateral_kwargs['job_uuid'] = job_uuid
        self.collateral_kwargs['project_uuid'] = None
        models.Job.query.with_for_update().filter_by(uuid=job_uuid).one_or_none()
        job = models.Job.query.options(joinedload(models.Job.pipeline_runs)).filter_by(uuid=job_uuid).one_or_none()
        if job is None:
            return False
        self.collateral_kwargs['project_uuid'] = job.project_uuid
        if job.status in ['SUCCESS', 'FAILURE', 'ABORTED']:
            return
        job.status = 'ABORTED'
        job.next_scheduled_time = None
        for run in job.pipeline_runs:
            if run.status in ['PENDING', 'STARTED']:
                run_uuids.append(run.uuid)
        for run_uuid in run_uuids:
            filter_by = {'uuid': run_uuid}
            status_update = {'status': 'ABORTED'}
            if update_status_db(status_update, model=models.NonInteractivePipelineRun, filter_by=filter_by):
                events.register_job_pipeline_run_cancelled(job.project_uuid, job.uuid, run_uuid)
            filter_by = {'run_uuid': run_uuid}
            status_update = {'status': 'ABORTED'}
            update_status_db(status_update, model=models.PipelineRunStep, filter_by=filter_by)
        events.register_job_cancelled(job.project_uuid, job.uuid)
        return True

    def _collateral(self, project_uuid: str, run_uuids: List[str], **kwargs):
        if False:
            for i in range(10):
                print('nop')
        celery = current_app.config['CELERY']
        celery.control.revoke(run_uuids, timeout=1.0)
        for run_uuid in run_uuids:
            res = AbortableAsyncResult(run_uuid, app=celery)
            res.abort()

class CreateJob(TwoPhaseFunction):
    """Create a job."""

    def _verify_all_pipelines_have_valid_environments(self, snapshot_uuid: str) -> None:
        if False:
            while True:
                i = 10
        snapshot: models.Snapshot = models.Snapshot.query.get(snapshot_uuid)
        environments_pipelines_mapping: Dict[str, Set[str]] = {}
        for (_, data) in snapshot.pipelines.items():
            definition = data['definition']
            pipeline = construct_pipeline(uuids=[], run_type='full', pipeline_definition=definition)
            pipeline_uuid = pipeline.properties['uuid']
            for step in pipeline.steps:
                environment_uuid = step.properties['environment']
                if environment_uuid not in environments_pipelines_mapping:
                    environments_pipelines_mapping[environment_uuid] = {pipeline_uuid}
                else:
                    environments_pipelines_mapping[environment_uuid].add(pipeline_uuid)
        try:
            environments.get_env_uuids_to_image_mappings(snapshot.project_uuid, set(environments_pipelines_mapping.keys()), True)
        except errors.ImageNotFoundWithUUIDs as error:
            invalid_environments = error.uuids
            pipelines_with_invalid_environments: Set[str] = set()
            for environment_uuid in invalid_environments:
                pipelines_with_invalid_environments.update(environments_pipelines_mapping[environment_uuid])
            raise errors.PipelinesHaveInvalidEnvironments(list(pipelines_with_invalid_environments), 'Pipelines contains invalid environment UUIDs.')

    def _transaction(self, job_spec: Dict[str, Any]) -> models.Job:
        if False:
            i = 10
            return i + 15
        scheduled_start = job_spec.get('scheduled_start', None)
        cron_schedule = job_spec.get('cron_schedule', None)
        if cron_schedule is None and scheduled_start is None:
            next_scheduled_time = None
        elif cron_schedule is None:
            next_scheduled_time = datetime.fromisoformat(scheduled_start)
        elif cron_schedule is not None and scheduled_start is None:
            if not croniter.is_valid(cron_schedule):
                raise ValueError(f'Invalid cron schedule: {cron_schedule}')
            next_scheduled_time = croniter(cron_schedule, datetime.now(timezone.utc)).get_next(datetime)
        else:
            raise ValueError("Can't define both cron_schedule and scheduled_start.")
        if not job_spec.get('parameters', []):
            raise ValueError('Cannot use an empty list of parameters. That would result in the job having no runs.')
        self._verify_all_pipelines_have_valid_environments(job_spec['snapshot_uuid'])
        job = {'uuid': job_spec['uuid'], 'name': job_spec['name'], 'project_uuid': job_spec['project_uuid'], 'pipeline_uuid': job_spec['pipeline_uuid'], 'pipeline_name': job_spec['pipeline_name'], 'schedule': cron_schedule, 'parameters': job_spec['parameters'], 'env_variables': get_proj_pip_env_variables(job_spec['project_uuid'], job_spec['pipeline_uuid']) if 'env_variables' not in job_spec else job_spec['env_variables'], 'pipeline_definition': job_spec['pipeline_definition'], 'pipeline_run_spec': job_spec['pipeline_run_spec'], 'total_scheduled_executions': 0, 'next_scheduled_time': next_scheduled_time, 'status': 'DRAFT', 'strategy_json': job_spec.get('strategy_json', {}), 'created_time': datetime.now(timezone.utc), 'max_retained_pipeline_runs': job_spec.get('max_retained_pipeline_runs', -1), 'snapshot_uuid': job_spec['snapshot_uuid']}
        db.session.add(models.Job(**job))
        spec = copy.deepcopy(job_spec['pipeline_run_spec'])
        spec['pipeline_definition'] = job_spec['pipeline_definition']
        pipeline = construct_pipeline(**spec)
        environments.lock_environment_images_for_job(job_spec['uuid'], job_spec['project_uuid'], pipeline.get_environments())
        self.collateral_kwargs['job_uuid'] = job_spec['uuid']
        events.register_job_created(job['project_uuid'], job['uuid'])
        return job

    def _collateral(self, job_uuid: str):
        if False:
            while True:
                i = 10
        pass

    def _revert(self):
        if False:
            print('Hello World!')
        models.Job.query.filter_by(uuid=self.collateral_kwargs['job_uuid']).delete()
        db.session.commit()

class UpdateJobParameters(TwoPhaseFunction):
    """Update a job."""

    def _transaction(self, job_uuid: str, name: str, cron_schedule: str, parameters: Dict[str, Any], env_variables: Dict[str, str], next_scheduled_time: str, strategy_json: Dict[str, Any], max_retained_pipeline_runs: int, confirm_draft):
        if False:
            return 10
        job = models.Job.query.with_for_update().filter(models.Job.status.not_in(['SUCCESS', 'ABORTED', 'FAILURE'])).filter_by(uuid=job_uuid).one()
        old_job = job.as_dict()
        if name is not None:
            job.name = name
        if cron_schedule is not None:
            if job.schedule is None and job.status != 'DRAFT':
                raise ValueError('Failed update operation. Cannot set the schedule of a job which is not a cron job already.')
            if not croniter.is_valid(cron_schedule):
                raise ValueError(f'Failed update operation. Invalid cron schedule: {cron_schedule}')
            job.schedule = cron_schedule
            job.next_scheduled_time = croniter(cron_schedule, datetime.now(timezone.utc)).get_next(datetime)
        if parameters is not None:
            if job.schedule is None and job.status != 'DRAFT':
                raise ValueError('Failed update operation. Cannot update the parameters of a job which is not a cron job.')
            if not parameters:
                raise ValueError('Failed update operation. Cannot use an empty list of parameters. That would result in the job having no runs.')
            job.parameters = parameters
        if env_variables is not None:
            if job.schedule is None and job.status != 'DRAFT':
                raise ValueError('Failed update operation. Cannot update the env variables of a job which is not a cron job.')
            if not _utils.are_environment_variables_valid(env_variables):
                raise ValueError('Invalid environment variables definition.')
            job.env_variables = env_variables
        if next_scheduled_time is not None:
            if job.status != 'DRAFT':
                raise ValueError('Failed update operation. Cannot set the next scheduled time of a job which is not a draft.')
            if job.schedule is not None and cron_schedule is not None:
                raise ValueError('Failed update operation. Cannot set the next scheduled time of a cron job.')
            if cron_schedule is None:
                job.schedule = None
            job.next_scheduled_time = datetime.fromisoformat(next_scheduled_time)
        if job.status == 'DRAFT' and next_scheduled_time is None and (cron_schedule is None):
            job.schedule = None
            job.next_scheduled_time = None
        if strategy_json is not None:
            if job.schedule is None and job.status != 'DRAFT':
                raise ValueError('Failed update operation. Cannot set the strategy jsonof a job which is not a draft nor a cron job.')
            job.strategy_json = strategy_json
        if max_retained_pipeline_runs is not None:
            if job.schedule is None and job.status != 'DRAFT':
                raise ValueError('Failed update operation. Cannot update the max_retained_pipeline_runs of a job which is not a draft nor a cron job.')
            if max_retained_pipeline_runs < -1:
                raise ValueError(f'Failed update operation. Invalid max_retained_pipeline_runs: {max_retained_pipeline_runs}.')
            job.max_retained_pipeline_runs = max_retained_pipeline_runs
        update_already_registered = False
        if confirm_draft:
            if job.status != 'DRAFT':
                raise ValueError('Failed update operation. The job is not a draft.')
            pipeline_def = job.pipeline_definition
            pipeline_def_environment_uuids = [step['environment'] for step in pipeline_def['steps'].values()]
            environments.get_env_uuids_to_image_mappings(job.project_uuid, set(pipeline_def_environment_uuids))
            if job.schedule is None:
                job.status = 'PENDING'
                if job.next_scheduled_time is None or job.next_scheduled_time.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc):
                    job.next_scheduled_time = None
                    job.last_scheduled_time = datetime.now(timezone.utc)
                    UpdateJobParameters._register_job_updated_event(old_job, job.as_dict())
                    update_already_registered = True
                    RunJob(self.tpe).transaction(job.uuid)
                else:
                    job.last_scheduled_time = job.next_scheduled_time
            else:
                job.last_scheduled_time = job.next_scheduled_time
                job.status = 'STARTED'
                UpdateJobParameters._register_job_updated_event(old_job, job.as_dict())
                update_already_registered = True
                events.register_job_started(job.project_uuid, job.uuid)
        if not update_already_registered:
            UpdateJobParameters._register_job_updated_event(old_job, job.as_dict())

    def _collateral(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    @staticmethod
    def _register_job_updated_event(old_job: Dict[str, Any], new_job: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Register the job_updated_event along with the changes.\n\n        Note that we are banking on the fact that the logic before the\n        call to this function will catch invalid updates.\n        '
        changes = []
        changes.extend(get_env_vars_update(old_job['env_variables'], new_job['env_variables']))
        to_compare = [('name', False), ('schedule', True), ('parameters', False), ('strategy_json', False), ('max_retained_pipeline_runs', True), ('next_scheduled_time', True), ('status', True)]
        for (field, record_values) in to_compare:
            old_value = old_job[field]
            new_value = new_job[field]
            if old_value == new_value:
                continue
            change = app_types.Change(type=app_types.ChangeType.UPDATED, changed_object=field)
            if record_values:
                change['old_value'] = str(old_value)
                change['new_value'] = str(new_value)
            changes.append(change)
        if changes:
            events.register_job_updated(old_job['schedule'], new_job['project_uuid'], new_job['uuid'], update=app_types.EntityUpdate(changes=changes))

class UpdateDraftJobPipeline(TwoPhaseFunction):
    """Changes a DRAFT job pipeline."""

    @staticmethod
    def _resolve_environment_variables(project_env_vars: Dict[str, str], old_pipeline_env_vars: Dict[str, str], new_pipeline_env_vars: Dict[str, str], old_job_env_vars: Dict[str, str]) -> Dict[str, str]:
        if False:
            return 10
        "Resolves the environment variables to be used for the job.\n\n        When changing the pipeline for a draft job we'd like to carry\n        over all work that the user has done w.r.t. setting/changing the\n        environment variables of the job. Do to that, we have to\n        reconstruct the changes the user has made and resolve\n        ambiguities.\n\n        This logic identifies user changes as:\n        - removing env vars inherited by the project or pipeline\n        - changing env vars inherited by the project or pipeline\n        - adding new environment variables\n\n        Ambiguities:\n        - an env var inherited by the old pipeline which hasn't been\n            changed signals that the user wanted the default value of\n            the pipeline env var.  If the new pipeline has such env var,\n            use the default value coming from the new pipeline, if it\n            doesn't have the env var, ignore the variable, i.e. do not\n            include it in the resulting set.\n        - an env var inherited by the project wasn't changed, and the\n            old pipeline didn't overwrite the value of that variable. If\n            the new pipeline has that env var then it will overwrite the\n            value.\n\n        "
        old_proj_ppl_merge = {**project_env_vars, **old_pipeline_env_vars}
        removed_env_vars = set()
        changed_env_vars = dict()
        added_env_vars = dict()
        for env_var in old_proj_ppl_merge:
            if env_var not in old_job_env_vars:
                removed_env_vars.add(env_var)
            elif old_proj_ppl_merge[env_var] != old_job_env_vars[env_var]:
                changed_env_vars[env_var] = old_job_env_vars[env_var]
        for env_var in old_job_env_vars:
            if env_var not in old_proj_ppl_merge:
                added_env_vars[env_var] = old_job_env_vars[env_var]
        result = {**project_env_vars, **new_pipeline_env_vars, **changed_env_vars, **added_env_vars}
        for env_var in removed_env_vars:
            result.pop(env_var, None)
        return result

    def _transaction(self, job_uuid: str, pipeline_uuid: str) -> None:
        if False:
            return 10
        job: models.Job = models.Job.query.with_for_update().filter_by(uuid=job_uuid).one()
        if job.status != 'DRAFT':
            raise ValueError('Failed update operation. Only jobs in the DRAFT status can have their pipeline changed.')
        snapshot: models.Snapshot = models.Snapshot.query.get(job.snapshot_uuid)
        if pipeline_uuid not in snapshot.pipelines:
            raise ValueError(f"The job snapshot doesn't contain pipeline: {pipeline_uuid}.")
        old_pipeline_uuid = job.pipeline_uuid
        job.pipeline_uuid = pipeline_uuid
        job.pipeline_definition = snapshot.pipelines[pipeline_uuid]['definition']
        job.pipeline_name = job.pipeline_definition['name']
        job.pipeline_run_spec['run_config']['pipeline_path'] = snapshot.pipelines[pipeline_uuid]['path']
        attributes.flag_modified(job, 'pipeline_run_spec')
        job.env_variables = UpdateDraftJobPipeline._resolve_environment_variables(snapshot.project_env_variables, snapshot.pipelines_env_variables[old_pipeline_uuid], snapshot.pipelines_env_variables[pipeline_uuid], job.env_variables)
        job.parameters = [{}]
        job.strategy_json = {}
        environments.release_environment_images_for_job(job.uuid)
        pipeline = construct_pipeline(uuids=[], run_type='full', pipeline_definition=job.pipeline_definition)
        environments.lock_environment_images_for_job(job.uuid, job.project_uuid, pipeline.get_environments())

    def _collateral(self):
        if False:
            i = 10
            return i + 15
        pass

class DeleteJob(TwoPhaseFunction):
    """Delete a job."""

    def _transaction(self, job_uuid):
        if False:
            return 10
        self.collateral_kwargs['project_uuid'] = None
        job = models.Job.query.filter_by(uuid=job_uuid).one_or_none()
        if job is None:
            return False
        self.collateral_kwargs['project_uuid'] = job.project_uuid
        AbortJob(self.tpe).transaction(job_uuid)
        events.register_job_deleted(job.project_uuid, job.uuid)
        db.session.delete(job)
        return True

    def _collateral(self, project_uuid: str):
        if False:
            i = 10
            return i + 15
        pass

class DeleteJobPipelineRun(TwoPhaseFunction):
    """Delete a job pipeline run."""

    def _transaction(self, job_uuid, run_uuid):
        if False:
            for i in range(10):
                print('nop')
        self.collateral_kwargs['project_uuid'] = None
        self.collateral_kwargs['pipeline_uuid'] = None
        self.collateral_kwargs['job_uuid'] = job_uuid
        self.collateral_kwargs['run_uuid'] = run_uuid
        job = db.session.query(models.Job.project_uuid, models.Job.pipeline_uuid).filter_by(uuid=job_uuid).one_or_none()
        if job is None:
            return False
        self.collateral_kwargs['project_uuid'] = job.project_uuid
        self.collateral_kwargs['pipeline_uuid'] = job.pipeline_uuid
        run = models.NonInteractivePipelineRun.query.filter_by(uuid=run_uuid).one_or_none()
        if run is None:
            return False
        AbortJobPipelineRun(self.tpe).transaction(job_uuid, run_uuid)
        events.register_job_pipeline_run_deleted(job.project_uuid, job_uuid, run_uuid)
        db.session.delete(run)
        return True

    def _collateral(self, project_uuid: str, pipeline_uuid: str, job_uuid: str, run_uuid: str):
        if False:
            while True:
                i = 10
        if project_uuid is None or pipeline_uuid is None or job_uuid is None or (run_uuid is None):
            return
        celery_job_kwargs = {'project_uuid': project_uuid, 'pipeline_uuid': pipeline_uuid, 'job_uuid': job_uuid, 'pipeline_run_uuids': [run_uuid]}
        task_args = {'name': 'app.core.tasks.delete_job_pipeline_run_directories', 'kwargs': celery_job_kwargs, 'task_id': str(uuid.uuid4())}
        celery = current_app.config['CELERY']
        res = celery.send_task(**task_args)
        res.forget()

class UpdateJobPipelineRun(TwoPhaseFunction):
    """Update a pipeline run of a job."""

    def _transaction(self, job_uuid: str, pipeline_run_uuid: str, status: str):
        if False:
            return 10
        'Set the status of a pipeline run.'
        filter_by = {'job_uuid': job_uuid, 'uuid': pipeline_run_uuid}
        job = models.Job.query.with_for_update().filter_by(uuid=job_uuid).one()
        if update_status_db({'status': status}, model=models.NonInteractivePipelineRun, filter_by=filter_by):
            {'SUCCESS': events.register_job_pipeline_run_succeeded, 'FAILURE': events.register_job_pipeline_run_failed, 'ABORTED': events.register_job_pipeline_run_cancelled, 'STARTED': events.register_job_pipeline_run_started}[status](job.project_uuid, job_uuid, pipeline_run_uuid)
        if status in ['SUCCESS', 'FAILURE', 'ABORTED']:
            if job.schedule is None:
                self._update_one_off_job(job)
            else:
                self._update_cron_job_run(job, pipeline_run_uuid, status)
            DeleteNonRetainedJobPipelineRuns(self.tpe).transaction(job_uuid)
        return ({'message': 'Status was updated successfully'}, 200)

    def _update_cron_job_run(self, job: models.Job, pipeline_run_uuid: str, run_status: str) -> None:
        if False:
            i = 10
            return i + 15
        run_index = db.session.query(models.NonInteractivePipelineRun.job_run_index).filter(models.NonInteractivePipelineRun.job_uuid == job.uuid, models.NonInteractivePipelineRun.uuid == pipeline_run_uuid).one().job_run_index
        if run_status == 'FAILURE':
            event = models.CronJobRunEvent.query.filter(models.CronJobRunEvent.type == 'project:cron-job:run:failed', models.CronJobRunEvent.project_uuid == job.project_uuid, models.CronJobRunEvent.job_uuid == job.uuid, models.CronJobRunEvent.run_index == run_index).first()
            if event is None:
                events.register_cron_job_run_failed(job.project_uuid, job.uuid, run_index)
        elif run_status in ['SUCCESS', 'ABORTED']:
            runs_to_complete = models.NonInteractivePipelineRun.query.filter(models.NonInteractivePipelineRun.job_uuid == job.uuid, models.NonInteractivePipelineRun.job_run_index == run_index, models.NonInteractivePipelineRun.status.in_(['PENDING', 'STARTED'])).count()
            failed_runs = models.NonInteractivePipelineRun.query.filter(models.NonInteractivePipelineRun.job_uuid == job.uuid, models.NonInteractivePipelineRun.job_run_index == run_index, models.NonInteractivePipelineRun.status == 'FAILURE').count()
            if runs_to_complete == 0 and failed_runs == 0:
                events.register_cron_job_run_succeeded(job.project_uuid, job.uuid, run_index)

    def _update_one_off_job(self, job: models.Job) -> None:
        if False:
            return 10
        total_job_runs = db.session.query(func.jsonb_array_length(models.Job.parameters)).filter_by(uuid=job.uuid).one()
        runs_to_complete = models.NonInteractivePipelineRun.query.filter_by(job_uuid=job.uuid).filter(models.NonInteractivePipelineRun.status.in_(['PENDING', 'STARTED'])).count()
        current_app.logger.info(f'Non recurring job {job.uuid} has completed {total_job_runs[0] - runs_to_complete}/{total_job_runs[0]} runs.')
        if runs_to_complete == 0 and job.status == 'STARTED':
            status = 'SUCCESS'
            if models.NonInteractivePipelineRun.query.filter_by(job_uuid=job.uuid).filter(models.NonInteractivePipelineRun.status == 'FAILURE').count() > 0:
                status = 'FAILURE'
            if models.Job.query.filter_by(uuid=job.uuid).filter(models.Job.status.not_in(['SUCCESS', 'ABORTED', 'FAILURE'])).update({'status': status}) > 0:
                if status == 'SUCCESS':
                    events.register_job_succeeded(job.project_uuid, job.uuid)
                else:
                    events.register_job_failed(job.project_uuid, job.uuid)

    def _collateral(self):
        if False:
            print('Hello World!')
        pass

class AbortJobPipelineRun(TwoPhaseFunction):
    """Aborts a job pipeline run."""

    def _transaction(self, job_uuid, run_uuid):
        if False:
            while True:
                i = 10
        could_abort = AbortPipelineRun(self.tpe).transaction(run_uuid)
        if not could_abort:
            return False
        job = db.session.query(models.Job.project_uuid, models.Job.pipeline_uuid).filter_by(uuid=job_uuid).one()
        events.register_job_pipeline_run_cancelled(job.project_uuid, job_uuid, run_uuid)
        UpdateJobPipelineRun(self.tpe).transaction(job_uuid, run_uuid, 'ABORTED')
        return True

    def _collateral(self):
        if False:
            while True:
                i = 10
        pass

class PauseCronJob(TwoPhaseFunction):
    """Pauses a cron job."""

    def _transaction(self, job_uuid):
        if False:
            while True:
                i = 10
        job = models.Job.query.with_for_update().filter_by(uuid=job_uuid, status='STARTED').filter(models.Job.schedule.isnot(None)).one_or_none()
        if job is None:
            return False
        job.status = 'PAUSED'
        job.next_scheduled_time = None
        events.register_cron_job_paused(job.project_uuid, job.uuid)
        return True

    def _collateral(self):
        if False:
            return 10
        pass

class ResumeCronJob(TwoPhaseFunction):
    """Resumes a cron job."""

    def _transaction(self, job_uuid):
        if False:
            while True:
                i = 10
        job = models.Job.query.with_for_update().filter_by(uuid=job_uuid, status='PAUSED').filter(models.Job.schedule.isnot(None)).one_or_none()
        if job is None:
            return None
        job.status = 'STARTED'
        job.next_scheduled_time = croniter(job.schedule, datetime.now(timezone.utc)).get_next(datetime)
        events.register_cron_job_unpaused(job.project_uuid, job.uuid)
        return str(job.next_scheduled_time)

    def _collateral(self):
        if False:
            return 10
        pass