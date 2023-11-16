from flask import request
from flask_restx import Namespace, Resource
import app.models as models
from app import schema
api = Namespace('services', description='Get information about running services')
api = schema.register_schema(api)

@api.route('/')
class ServiceList(Resource):

    @api.doc('fetch_services')
    @api.marshal_with(schema.service_descriptions)
    def get(self):
        if False:
            for i in range(10):
                print('nop')
        'Fetches all services.\n\n        Services can be part of Job runs or InteractiveSessions'
        project_uuid_prefix_filter = request.args.get('project_uuid_prefix')
        session_uuid_prefix_filter = request.args.get('session_uuid_prefix')
        query = models.InteractiveSession.query
        if project_uuid_prefix_filter is not None:
            query = query.filter(models.InteractiveSession.project_uuid.startswith(project_uuid_prefix_filter))
        if session_uuid_prefix_filter is not None:
            query = query.filter(models.InteractiveSession.pipeline_uuid.startswith(session_uuid_prefix_filter))
        sessions = query.all()
        session_services = []
        for session in sessions:
            for service in session.user_services.values():
                session_services.append({'service': service, 'type': 'INTERACTIVE', 'project_uuid': session.project_uuid, 'pipeline_uuid': session.pipeline_uuid})
        query_runs = models.NonInteractivePipelineRun.query.with_entities(models.NonInteractivePipelineRun.job_uuid, models.NonInteractivePipelineRun.uuid)
        if project_uuid_prefix_filter is not None:
            query_runs = query_runs.join(models.Job).filter(models.Job.project_uuid.startswith(project_uuid_prefix_filter))
        if session_uuid_prefix_filter is not None:
            query_runs = query_runs.filter(models.NonInteractivePipelineRun.uuid.startswith(session_uuid_prefix_filter))
        active_pipeline_runs = query_runs.filter_by(status='STARTED').all()
        job_ids = {run.job_uuid for run in active_pipeline_runs}
        jobs_lookup = {}
        jobs_definitions = models.Job.query.with_entities(models.Job.pipeline_definition, models.Job.uuid, models.Job.project_uuid, models.Job.pipeline_uuid).filter(models.Job.uuid.in_(job_ids)).all()
        for job in jobs_definitions:
            jobs_lookup[job.uuid] = job
        run_services = []
        for run in active_pipeline_runs:
            pipeline_def = jobs_lookup[run.job_uuid].pipeline_definition
            if 'services' in pipeline_def:
                for service in pipeline_def['services'].values():
                    run_services.append({'service': service, 'type': 'NONINTERACTIVE', 'project_uuid': jobs_lookup[run.job_uuid].project_uuid, 'pipeline_uuid': jobs_lookup[run.job_uuid].pipeline_uuid, 'job_uuid': run.job_uuid, 'run_uuid': run.uuid})
        all_services = session_services + run_services
        return ({'services': all_services}, 200)