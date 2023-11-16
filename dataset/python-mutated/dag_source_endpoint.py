from __future__ import annotations
from http import HTTPStatus
from flask import Response, current_app, request
from itsdangerous import BadSignature, URLSafeSerializer
from airflow.api_connexion import security
from airflow.api_connexion.exceptions import NotFound
from airflow.api_connexion.schemas.dag_source_schema import dag_source_schema
from airflow.auth.managers.models.resource_details import DagAccessEntity
from airflow.models.dagcode import DagCode

@security.requires_access_dag('GET', DagAccessEntity.CODE)
def get_dag_source(*, file_token: str) -> Response:
    if False:
        while True:
            i = 10
    'Get source code using file token.'
    secret_key = current_app.config['SECRET_KEY']
    auth_s = URLSafeSerializer(secret_key)
    try:
        path = auth_s.loads(file_token)
        dag_source = DagCode.code(path)
    except (BadSignature, FileNotFoundError):
        raise NotFound('Dag source not found')
    return_type = request.accept_mimetypes.best_match(['text/plain', 'application/json'])
    if return_type == 'text/plain':
        return Response(dag_source, headers={'Content-Type': return_type})
    if return_type == 'application/json':
        content = dag_source_schema.dumps({'content': dag_source})
        return Response(content, headers={'Content-Type': return_type})
    return Response('Not Allowed Accept Header', status=HTTPStatus.NOT_ACCEPTABLE)