from __future__ import annotations
import logging
import urllib.parse
import uuid
from typing import Any, Dict
from sentry import features, http, options
from sentry.datascrubbing import scrub_data
from sentry.event_manager import EventManager
from sentry.models.options.project_option import ProjectOption
from sentry.models.project import Project
from sentry.silo import SiloMode
from sentry.tasks.base import instrumented_task
from sentry.utils import json
from sentry.utils.safe import safe_execute
RECAP_SERVER_URL_OPTION = 'sentry:recap_server_url'
RECAP_SERVER_TOKEN_OPTION = 'sentry:recap_server_token'
RECAP_SERVER_LATEST_ID = 'sentry:recap_server_poll_id'
logger = logging.getLogger(__name__)

@instrumented_task(name='sentry.tasks.poll_recap_servers', queue='recap_servers', silo_mode=SiloMode.REGION)
def poll_recap_servers(**kwargs):
    if False:
        i = 10
        return i + 15
    project_ids = ProjectOption.objects.filter(key=RECAP_SERVER_URL_OPTION).exclude(value__isnull=True).values_list('project_id', flat=True)
    for project_id in project_ids:
        poll_project_recap_server.delay(project_id)

@instrumented_task(name='sentry.tasks.poll_project_recap_server', queue='recap_servers', silo_mode=SiloMode.REGION)
def poll_project_recap_server(project_id: int, **kwargs) -> None:
    if False:
        while True:
            i = 10
    try:
        project = Project.objects.get(id=project_id)
    except Project.DoesNotExist:
        logger.warning('Polled project do not exist', extra={'project_id': project_id})
        return
    if not features.has('organizations:recap-server', project.organization):
        logger.info('Recap server polling feature is not enabled for a given organization', extra={'organization': project.organization})
        return
    recap_server_url = project.get_option(RECAP_SERVER_URL_OPTION)
    if recap_server_url is None:
        logger.warning('Polled project has no recap server url configured', extra={'project': project})
        return
    latest_id = project.get_option(RECAP_SERVER_LATEST_ID, 0)
    url = recap_server_url.strip().rstrip('/') + '/rest/v1/crashes;sort=id:ascending'
    if latest_id == 0:
        url = f'{url};limit=1000'
    else:
        url = url + urllib.parse.quote(f';q=id:{{{latest_id} TO *}}', safe=';=:')
    headers = {'Accept': 'application/vnd.scea.recap.crashes+json; version=1'}
    access_token = project.get_option(RECAP_SERVER_TOKEN_OPTION, None)
    if access_token is not None:
        headers['Authorization'] = f'Bearer {access_token}'
    result = http.fetch_file(url, headers=headers)
    try:
        crashes = json.loads(result.body)
        if not isinstance(crashes, dict):
            logger.exception('Polled project endpoint did not responded with json object', extra={'project': project})
            return
    except json.JSONDecodeError as exc:
        logger.exception('Polled project endpoint did not responded with valid json', exc_info=exc, extra={'project': project, 'url': url})
        return
    if crashes.get('results') is None or crashes.get('results') == 0:
        return
    try:
        for crash in crashes['_embedded']['crash']:
            store_crash(crash, project, url)
            latest_id = max(latest_id, crash['id'])
    finally:
        project.update_option(RECAP_SERVER_LATEST_ID, latest_id)

def store_crash(crash, project: Project, url: str) -> None:
    if False:
        while True:
            i = 10
    try:
        event = translate_crash_to_event(crash, project, url)
    except KeyError as exc:
        logger.exception('Crash dump data has invalid payload', exc_info=exc, extra={'project': project, 'url': url})
        return
    if options.get('processing.can-use-scrubbers'):
        new_event = safe_execute(scrub_data, project=project, event=event, _with_transaction=False)
        if new_event is not None:
            event = new_event
    event_manager = EventManager(event, project=project)
    event_manager.save(project_id=project.id)

def translate_crash_to_event(crash, project: Project, url: str) -> Dict[str, Any]:
    if False:
        return 10
    event: dict[str, Any] = {'event_id': uuid.uuid4().hex, 'project': project.id, 'platform': 'c', 'exception': {'values': [{'type': crash['stopReason']}]}, 'tags': {'id': crash['id']}, 'contexts': {'request': {'url': crash['_links']['self']}}}
    if 'uploadDate' in crash:
        event['timestamp'] = crash['uploadDate']
    if 'stopLocation' in crash:
        event['exception']['values'][0]['value'] = crash['stopLocation']
    elif 'returnLocation' in crash:
        event['exception']['values'][0]['value'] = crash['returnLocation']
    if 'detailedStackTrace' in crash:
        frames = []
        for frame in crash['detailedStackTrace']:
            processed_frame = {'filename': frame['sourceFile'], 'lineno': frame['sourceLine'], 'instruction_addr': frame['absoluteAddress'], 'module': frame['moduleName'], 'function': frame['resolvedSymbol'], 'raw_function': frame['displayValue'], 'in_app': True}
            frames.append(processed_frame)
        event['exception']['values'][0]['stacktrace'] = {'frames': frames}
    elif 'stackTrace' in crash:
        frames = []
        for frame in crash['stackTrace']:
            processed_frame = {'function': frame, 'in_app': True}
            frames.append(processed_frame)
        event['exception']['values'][0]['stacktrace'] = {'frames': frames}
    if 'titleId' in crash:
        event['tags']['titleId'] = crash['titleId']
    if 'platform' in crash:
        if 'sysVersion' in crash:
            event['contexts']['runtime'] = {'name': crash['platform'], 'version': crash['sysVersion']}
        if 'hardwareId' in crash:
            event['contexts']['device'] = {'name': crash['platform'], 'model_id': crash['hardwareId']}
    if 'appVersion' in crash:
        event['contexts']['app'] = {'app_version': crash['appVersion']}
    if 'userData' in crash:
        event['contexts']['userData'] = crash['userData']
    return event