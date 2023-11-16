"""API endpoints for unspecified orchest-api level information."""
from datetime import datetime, timezone
from flask import current_app, request
from flask_restx import Namespace, Resource
from app import models, schema
from app.connections import db
from app.core import sessions
api = Namespace('info', description='Orchest-api information.')
api = schema.register_schema(api)

@api.route('/idle')
@api.param('skip_details', "If true then will return as soon as Orchest is found to not be idle. The 'details' entry of the returned json might thus be incomplete.")
class IdleCheck(Resource):

    @api.doc('orchest_api_idle')
    @api.marshal_with(schema.idleness_check_result, code=200, description='Orchest-api idleness check.')
    def get(self):
        if False:
            i = 10
            return i + 15
        'Checks if the Orchest-api is idle.\n\n        The Orchest-api is considered idle if:\n        - no environments are being built\n        - no jupyter images are being built\n        - there are no ongoing interactive-runs\n        - there are no ongoing job runs\n        - there are no busy kernels among running sessions, said busy\n            state is reported by JupyterLab, and reflects the fact that\n            a kernel is not actively doing some compute.\n        '
        idleness_data = is_orchest_idle(skip_details=request.args.get('skip_details', default=False, type=lambda v: v in ['True', 'true']))
        return (idleness_data, 200)

@api.route('/client-heartbeat')
class ClientHeartBeat(Resource):

    @api.doc('client_heartbeat')
    def get(self):
        if False:
            i = 10
            return i + 15
        'Allows to signal an heartbeat to the Orchest-api.\n\n        This allows the Orchest-api to know about the fact that some\n        clients are using Orchest.\n\n        '
        models.ClientHeartbeat.query.delete()
        db.session.add(models.ClientHeartbeat())
        db.session.commit()
        return ('', 200)

def is_orchest_idle(skip_details: bool=False) -> dict:
    if False:
        i = 10
        return i + 15
    'Checks if the orchest-api is idle.\n\n    Args:\n        skip_details: If True this function will return as soon as it\n        finds out that Orchest is not idle. The "details" entry of the\n        returned dictionary might thus be incomplete.\n\n    Returns:\n        See schema.idleness_check_result for details.\n    '
    data = {}
    result = {'details': data, 'idle': False}
    threshold = datetime.now(timezone.utc) - current_app.config['CLIENT_HEARTBEATS_IDLENESS_THRESHOLD']
    data['active_clients'] = db.session.query(db.session.query(models.ClientHeartbeat).filter(models.ClientHeartbeat.timestamp > threshold).exists()).scalar()
    if data['active_clients'] and skip_details:
        return result
    for (name, model) in [('ongoing_environment_image_builds', models.EnvironmentImageBuild), ('ongoing_jupyterlab_builds', models.JupyterImageBuild), ('ongoing_interactive_runs', models.InteractivePipelineRun), ('ongoing_job_runs', models.NonInteractivePipelineRun)]:
        data[name] = db.session.query(db.session.query(model).filter(model.status.in_(['PENDING', 'STARTED'])).exists()).scalar()
        if data[name] and skip_details:
            return result
    data['busy_kernels'] = False
    isessions = models.InteractiveSession.query.filter(models.InteractiveSession.status.in_(['RUNNING'])).all()
    for session in isessions:
        if sessions.has_busy_kernels(session.project_uuid[:18] + session.pipeline_uuid[:18]):
            data['busy_kernels'] = True
            if skip_details:
                return result
            break
    result['idle'] = not any(data.values())
    return result