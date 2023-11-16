"""Tasks for managing Debug Information Files from Apple App Store Connect.

Users can instruct Sentry to download dSYM from App Store Connect and put them into Sentry's
debug files.  These tasks enable this functionality.
"""
import logging
import pathlib
import tempfile
from typing import List, Mapping, Tuple
import requests
import sentry_sdk
from django.utils import timezone
from sentry.lang.native import appconnect
from sentry.models.appconnectbuilds import AppConnectBuild
from sentry.models.debugfile import create_files_from_dif_zip
from sentry.models.latestappconnectbuildscheck import LatestAppConnectBuildsCheck
from sentry.models.options.project_option import ProjectOption
from sentry.models.project import Project
from sentry.tasks.base import instrumented_task
from sentry.utils import json, metrics, sdk
from sentry.utils.appleconnect import appstore_connect as appstoreconnect_api
logger = logging.getLogger(__name__)

@instrumented_task(name='sentry.tasks.app_store_connect.dsym_download', queue='appstoreconnect', ignore_result=True)
def dsym_download(project_id: int, config_id: str) -> None:
    if False:
        i = 10
        return i + 15
    inner_dsym_download(project_id=project_id, config_id=config_id)

def inner_dsym_download(project_id: int, config_id: str) -> None:
    if False:
        i = 10
        return i + 15
    "Downloads the dSYMs from App Store Connect and stores them in the Project's debug files."
    with sdk.configure_scope() as scope:
        scope.set_tag('project', project_id)
        scope.set_tag('config_id', config_id)
    project = Project.objects.get(pk=project_id)
    config = appconnect.AppStoreConnectConfig.from_project_config(project, config_id)
    client = appconnect.AppConnectClient.from_config(config)
    listed_builds = client.list_builds()
    builds = process_builds(project=project, config=config, to_process=listed_builds)
    if not builds:
        return
    for (i, (build, build_state)) in enumerate(builds):
        with sdk.configure_scope() as scope:
            scope.set_context('dsym_downloads', {'total': len(builds), 'completed': i})
        with tempfile.NamedTemporaryFile() as dsyms_zip:
            try:
                client.download_dsyms(build, pathlib.Path(dsyms_zip.name))
            except appconnect.NoDsymsError:
                logger.debug('No dSYMs for build %s', build)
            except appconnect.PendingDsymsError:
                logger.debug('dSYM url currently unavailable for build %s', build)
                continue
            except appstoreconnect_api.UnauthorizedError:
                sentry_sdk.capture_message('Not authorized to download dSYM using current App Store Connect credentials', level='info')
                return
            except appstoreconnect_api.ForbiddenError:
                sentry_sdk.capture_message('Forbidden from downloading dSYM using current App Store Connect credentials', level='info')
                return
            except ValueError as e:
                sdk.capture_exception(e)
                continue
            except appstoreconnect_api.RequestError as e:
                sdk.capture_exception(e)
                continue
            except requests.RequestException as e:
                sdk.capture_exception(e)
                continue
            else:
                create_difs_from_dsyms_zip(dsyms_zip.name, project)
                logger.debug('Uploaded dSYMs for build %s', build)
                metrics.incr('tasks.app_store_connect.builds_ingested', sample_rate=1)
        build_state.fetched = True
        build_state.save()

def create_difs_from_dsyms_zip(dsyms_zip: str, project: Project) -> None:
    if False:
        return 10
    with sentry_sdk.start_span(op='dsym-difs', description='Extract difs dSYM zip'):
        with open(dsyms_zip, 'rb') as fp:
            created = create_files_from_dif_zip(fp, project, accept_unknown=True)
            for proj_debug_file in created:
                logger.debug('Created %r for project %s', proj_debug_file, project.id)

def get_or_create_persisted_build(project: Project, config: appconnect.AppStoreConnectConfig, build: appconnect.BuildInfo) -> AppConnectBuild:
    if False:
        return 10
    'Fetches the sentry-internal :class:`AppConnectBuild`.\n\n    The build corresponds to the :class:`appconnect.BuildInfo` as returned by the\n    AppStore Connect API. If no build exists yet, a new "pending" build is created.\n    '
    try:
        build_state = AppConnectBuild.objects.get(project=project, app_id=build.app_id, platform=build.platform, bundle_short_version=build.version, bundle_version=build.build_number)
    except AppConnectBuild.DoesNotExist:
        build_state = AppConnectBuild(project=project, app_id=build.app_id, bundle_id=config.bundleId, platform=build.platform, bundle_short_version=build.version, bundle_version=build.build_number, uploaded_to_appstore=build.uploaded_date, first_seen=timezone.now(), fetched=False)
        build_state.save()
    return build_state

def process_builds(project: Project, config: appconnect.AppStoreConnectConfig, to_process: List[appconnect.BuildInfo]) -> List[Tuple[appconnect.BuildInfo, AppConnectBuild]]:
    if False:
        while True:
            i = 10
    'Returns a list of builds whose dSYMs need to be updated or fetched.\n\n    This will create a new "pending" :class:`AppConnectBuild` for any :class:`appconnect.BuildInfo`\n    that cannot be found in the DB. These pending :class:`AppConnectBuild`s are immediately saved\n    upon creation.\n    '
    pending_builds = []
    with sentry_sdk.start_span(op='appconnect-update-builds', description='Update AppStoreConnect builds in database'):
        for build in to_process:
            build_state = get_or_create_persisted_build(project, config, build)
            if not build_state.fetched:
                pending_builds.append((build, build_state))
    LatestAppConnectBuildsCheck.objects.create_or_update(project=project, source_id=config.id, values={'last_checked': timezone.now()})
    return pending_builds

@instrumented_task(name='sentry.tasks.app_store_connect.refresh_all_builds', queue='appstoreconnect', ignore_result=True)
def refresh_all_builds() -> None:
    if False:
        while True:
            i = 10
    inner_refresh_all_builds()

def inner_refresh_all_builds() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Refreshes all AppStoreConnect builds for all projects.\n\n    This iterates over all the projects configured in Sentry and for any which has an\n    AppStoreConnect symbol source configured will poll the AppStoreConnect API to check if\n    there are new builds.\n    '
    options = ProjectOption.objects.filter(key=appconnect.SYMBOL_SOURCES_PROP_NAME)
    count = 0
    for option in options:
        with sdk.push_scope() as scope:
            scope.set_tag('project', option.project_id)
            try:
                if not option.value:
                    continue
                all_sources: List[Mapping[str, str]] = json.loads(option.value)
                for source in all_sources:
                    try:
                        source_id = source['id']
                        source_type = source['type']
                    except KeyError:
                        logger.exception('Malformed symbol source')
                        continue
                    if source_type == appconnect.SYMBOL_SOURCE_TYPE_NAME:
                        dsym_download.apply_async(kwargs={'project_id': option.project_id, 'config_id': source_id})
                        count += 1
            except Exception:
                logger.exception('Failed to refresh AppStoreConnect builds')
    metrics.gauge('tasks.app_store_connect.refreshed', count, sample_rate=1)