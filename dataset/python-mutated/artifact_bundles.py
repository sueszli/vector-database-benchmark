from __future__ import annotations
import random
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple
import sentry_sdk
from django.conf import settings
from django.db import router
from django.db.models import Count
from django.utils import timezone
from sentry import options
from sentry.models.artifactbundle import INDEXING_THRESHOLD, ArtifactBundle, ArtifactBundleArchive, ArtifactBundleIndex, ArtifactBundleIndexingState, DebugIdArtifactBundle, FlatFileIndexState, ProjectArtifactBundle, ReleaseArtifactBundle
from sentry.models.project import Project
from sentry.utils import metrics, redis
from sentry.utils.db import atomic_transaction
MAX_BUNDLES_QUERY = 5
AVAILABLE_FOR_RENEWAL_DAYS = 30
INDEXING_CACHE_TIMEOUT = 600

def get_redis_cluster_for_artifact_bundles():
    if False:
        print('Hello World!')
    cluster_key = settings.SENTRY_ARTIFACT_BUNDLES_INDEXING_REDIS_CLUSTER
    return redis.redis_clusters.get(cluster_key)

def get_refresh_key() -> str:
    if False:
        print('Hello World!')
    return 'artifact_bundles_in_use'

def _generate_artifact_bundle_indexing_state_cache_key(organization_id: int, artifact_bundle_id: int) -> str:
    if False:
        return 10
    return f'ab::o:{organization_id}:b:{artifact_bundle_id}:bundle_indexing_state'

def set_artifact_bundle_being_indexed_if_null(organization_id: int, artifact_bundle_id: int) -> bool:
    if False:
        return 10
    redis_client = get_redis_cluster_for_artifact_bundles()
    cache_key = _generate_artifact_bundle_indexing_state_cache_key(organization_id, artifact_bundle_id)
    return redis_client.set(cache_key, 1, ex=INDEXING_CACHE_TIMEOUT, nx=True)

def remove_artifact_bundle_indexing_state(organization_id: int, artifact_bundle_id: int) -> None:
    if False:
        i = 10
        return i + 15
    redis_client = get_redis_cluster_for_artifact_bundles()
    cache_key = _generate_artifact_bundle_indexing_state_cache_key(organization_id, artifact_bundle_id)
    redis_client.delete(cache_key)

def index_artifact_bundles_for_release(organization_id: int, artifact_bundles: List[Tuple[ArtifactBundle, ArtifactBundleArchive | None]]) -> None:
    if False:
        i = 10
        return i + 15
    '\n    This indexes the contents of `artifact_bundles` into the database, using the given `release` and `dist` pair.\n\n    Synchronization is achieved using a mixture of redis cache with transient state and a binary state in the database.\n    '
    for (artifact_bundle, archive) in artifact_bundles:
        try:
            if not set_artifact_bundle_being_indexed_if_null(organization_id=organization_id, artifact_bundle_id=artifact_bundle.id):
                metrics.incr('artifact_bundle_indexing.bundle_already_being_indexed')
                continue
            _index_urls_in_bundle(organization_id, artifact_bundle, archive)
        except Exception as e:
            metrics.incr('artifact_bundle_indexing.index_single_artifact_bundle_error')
            sentry_sdk.capture_exception(e)

@sentry_sdk.tracing.trace
def _index_urls_in_bundle(organization_id: int, artifact_bundle: ArtifactBundle, existing_archive: ArtifactBundleArchive | None):
    if False:
        for i in range(10):
            print('nop')
    archive = existing_archive or ArtifactBundleArchive(artifact_bundle.file.getfile(), build_memory_map=False)
    urls_to_index = []
    try:
        for info in archive.get_files().values():
            if (url := info.get('url')):
                urls_to_index.append(ArtifactBundleIndex(artifact_bundle_id=artifact_bundle.id, url=url, organization_id=organization_id, date_added=artifact_bundle.date_added, release_name='', dist_name='', date_last_modified=artifact_bundle.date_last_modified or artifact_bundle.date_added))
    finally:
        if not existing_archive:
            archive.close()
    with atomic_transaction(using=(router.db_for_write(ArtifactBundle), router.db_for_write(ArtifactBundleIndex))):
        bundle_was_indexed = ArtifactBundle.objects.filter(id=artifact_bundle.id, indexing_state=ArtifactBundleIndexingState.WAS_INDEXED.value).exists()
        if bundle_was_indexed:
            metrics.incr('artifact_bundle_indexing.bundle_was_already_indexed')
            return
        ArtifactBundleIndex.objects.bulk_create(urls_to_index, batch_size=50)
        ArtifactBundle.objects.filter(id=artifact_bundle.id).update(indexing_state=ArtifactBundleIndexingState.WAS_INDEXED.value)
        metrics.incr('artifact_bundle_indexing.bundles_indexed')
        metrics.incr('artifact_bundle_indexing.urls_indexed', len(urls_to_index))

@sentry_sdk.tracing.trace
def maybe_renew_artifact_bundles_from_processing(project_id: int, used_download_ids: List[str]):
    if False:
        for i in range(10):
            print('nop')
    if random.random() >= options.get('symbolicator.sourcemaps-bundle-index-refresh-sample-rate'):
        return
    artifact_bundle_ids = []
    for download_id in used_download_ids:
        split = download_id.split('/')
        if len(split) < 2:
            continue
        (ty, ty_id, *_rest) = split
        if ty != 'artifact_bundle':
            continue
        artifact_bundle_ids.append(ty_id)
    redis_client = get_redis_cluster_for_artifact_bundles()
    redis_client.sadd(get_refresh_key(), *artifact_bundle_ids)

@sentry_sdk.tracing.trace
def refresh_artifact_bundles_in_use():
    if False:
        for i in range(10):
            print('nop')
    LOOP_TIMES = 100
    IDS_PER_LOOP = 50
    redis_client = get_redis_cluster_for_artifact_bundles()
    now = timezone.now()
    threshold_date = now - timedelta(days=AVAILABLE_FOR_RENEWAL_DAYS)
    for _ in range(LOOP_TIMES):
        artifact_bundle_ids = redis_client.spop(get_refresh_key(), IDS_PER_LOOP)
        used_artifact_bundles = {id: date_added for (id, date_added) in ArtifactBundle.objects.filter(id__in=artifact_bundle_ids, date_added__lte=threshold_date).values_list('id', 'date_added')}
        maybe_renew_artifact_bundles(used_artifact_bundles)
        if len(artifact_bundle_ids) < IDS_PER_LOOP:
            break

def maybe_renew_artifact_bundles(used_artifact_bundles: Dict[int, datetime]):
    if False:
        return 10
    now = timezone.now()
    threshold_date = now - timedelta(days=AVAILABLE_FOR_RENEWAL_DAYS)
    for (artifact_bundle_id, date_added) in used_artifact_bundles.items():
        if date_added > threshold_date:
            continue
        with metrics.timer('artifact_bundle_renewal'):
            renew_artifact_bundle(artifact_bundle_id, threshold_date, now)

@sentry_sdk.tracing.trace
def renew_artifact_bundle(artifact_bundle_id: int, threshold_date: datetime, now: datetime):
    if False:
        print('Hello World!')
    metrics.incr('artifact_bundle_renewal.need_renewal')
    with atomic_transaction(using=(router.db_for_write(ArtifactBundle), router.db_for_write(ProjectArtifactBundle), router.db_for_write(ReleaseArtifactBundle), router.db_for_write(DebugIdArtifactBundle), router.db_for_write(ArtifactBundleIndex), router.db_for_write(FlatFileIndexState))):
        updated_rows_count = ArtifactBundle.objects.filter(id=artifact_bundle_id, date_added__lte=threshold_date).update(date_added=now)
        if updated_rows_count > 0:
            ProjectArtifactBundle.objects.filter(artifact_bundle_id=artifact_bundle_id, date_added__lte=threshold_date).update(date_added=now)
            ReleaseArtifactBundle.objects.filter(artifact_bundle_id=artifact_bundle_id, date_added__lte=threshold_date).update(date_added=now)
            DebugIdArtifactBundle.objects.filter(artifact_bundle_id=artifact_bundle_id, date_added__lte=threshold_date).update(date_added=now)
            ArtifactBundleIndex.objects.filter(artifact_bundle_id=artifact_bundle_id, date_added__lte=threshold_date).update(date_added=now)
            FlatFileIndexState.objects.filter(artifact_bundle_id=artifact_bundle_id, date_added__lte=threshold_date).update(date_added=now)
    if updated_rows_count > 0:
        metrics.incr('artifact_bundle_renewal.were_renewed')

def _maybe_renew_and_return_bundles(bundles: Dict[int, Tuple[datetime, str]]) -> List[Tuple[int, str]]:
    if False:
        print('Hello World!')
    maybe_renew_artifact_bundles({id: date_added for (id, (date_added, _resolved)) in bundles.items()})
    return [(id, resolved) for (id, (_date_added, resolved)) in bundles.items()]

def query_artifact_bundles_containing_file(project: Project, release: str, dist: str, url: str, debug_id: str | None) -> List[Tuple[int, str]]:
    if False:
        i = 10
        return i + 15
    '\n    This looks up the artifact bundles that satisfy the query consisting of\n    `release`, `dist`, `url` and `debug_id`.\n\n    This function should ideally return a single bundle containing the file matching\n    the query. However it can also return more than a single bundle in case no\n    complete index is available, in which case the N most recent bundles will be\n    returned under the assumption that one of those may contain the file.\n\n    Along the bundles `id`, it also returns the most-precise method the bundles\n    was resolved with.\n    '
    if debug_id:
        bundles = get_artifact_bundles_containing_debug_id(project, debug_id)
        if bundles:
            return _maybe_renew_and_return_bundles({id: (date_added, 'debug-id') for (id, date_added) in bundles})
    (total_bundles, indexed_bundles) = get_bundles_indexing_state(project, release, dist)
    if not total_bundles:
        return []
    is_fully_indexed = total_bundles > INDEXING_THRESHOLD and indexed_bundles == total_bundles
    if total_bundles > INDEXING_THRESHOLD and indexed_bundles < total_bundles:
        metrics.incr('artifact_bundle_indexing.query_partial_index')
    artifact_bundles: Dict[int, Tuple[datetime, str]] = dict()

    def update_bundles(bundles: Set[Tuple[int, datetime]], resolved: str):
        if False:
            return 10
        for (bundle_id, date_added) in bundles:
            artifact_bundles[bundle_id] = (date_added, resolved)
    if not is_fully_indexed:
        bundles = get_artifact_bundles_by_release(project, release, dist)
        update_bundles(bundles, 'release')
    if url:
        bundles = get_artifact_bundles_containing_url(project, release, dist, url)
        update_bundles(bundles, 'index')
    return _maybe_renew_and_return_bundles(artifact_bundles)

def get_bundles_indexing_state(project: Project, release_name: str, dist_name: str):
    if False:
        while True:
            i = 10
    '\n    Returns the number of total bundles, and the number of fully indexed bundles\n    associated with the given `release` / `dist`.\n    '
    total_bundles = 0
    indexed_bundles = 0
    for (state, count) in ArtifactBundle.objects.filter(releaseartifactbundle__organization_id=project.organization.id, releaseartifactbundle__release_name=release_name, releaseartifactbundle__dist_name=dist_name, projectartifactbundle__project_id=project.id).values_list('indexing_state').annotate(count=Count('*')):
        if state == ArtifactBundleIndexingState.WAS_INDEXED.value:
            indexed_bundles = count
        total_bundles += count
    return (total_bundles, indexed_bundles)

def get_artifact_bundles_containing_debug_id(project: Project, debug_id: str) -> Set[Tuple[int, datetime]]:
    if False:
        i = 10
        return i + 15
    '\n    Returns the most recently uploaded artifact bundle containing the given `debug_id`.\n    '
    return set(ArtifactBundle.objects.filter(organization_id=project.organization.id, projectartifactbundle__project_id=project.id, debugidartifactbundle__debug_id=debug_id).values_list('id', 'date_added').order_by('-date_last_modified', '-id')[:1])

def get_artifact_bundles_containing_url(project: Project, release_name: str, dist_name: str, url: str) -> Set[Tuple[int, datetime]]:
    if False:
        print('Hello World!')
    '\n    Returns the most recently uploaded bundle containing a file matching the `release`, `dist` and `url`.\n    '
    return set(ArtifactBundle.objects.filter(releaseartifactbundle__organization_id=project.organization.id, releaseartifactbundle__release_name=release_name, releaseartifactbundle__dist_name=dist_name, projectartifactbundle__project_id=project.id, artifactbundleindex__organization_id=project.organization.id, artifactbundleindex__url__icontains=url).values_list('id', 'date_added').order_by('-date_last_modified', '-id').distinct('date_last_modified', 'id')[:MAX_BUNDLES_QUERY])

def get_artifact_bundles_by_release(project: Project, release_name: str, dist_name: str) -> Set[Tuple[int, datetime]]:
    if False:
        i = 10
        return i + 15
    '\n    Returns up to N most recently uploaded bundles for the given `release` and `dist`.\n    '
    return set(ArtifactBundle.objects.filter(releaseartifactbundle__organization_id=project.organization.id, releaseartifactbundle__release_name=release_name, releaseartifactbundle__dist_name=dist_name, projectartifactbundle__project_id=project.id).values_list('id', 'date_added').order_by('-date_last_modified', '-id')[:MAX_BUNDLES_QUERY])