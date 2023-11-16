from urllib.parse import urlparse
from django.utils.encoding import force_bytes, force_str
from packaging.version import Version
from rest_framework.exceptions import NotFound, ParseError
from sentry import eventstore
from sentry.api.endpoints.project_release_files import ArtifactSource
from sentry.models.distribution import Distribution
from sentry.models.release import Release
from sentry.models.releasefile import ReleaseFile, read_artifact_index
from sentry.models.sourcemapprocessingissue import SourceMapProcessingIssue
from sentry.utils.javascript import find_sourcemap
from sentry.utils.urls import non_standard_url_join
JS_VERSION_FOR_DEBUG_ID = '7.44.0'
NO_DEBUG_ID_FRAMEWORKS = {'sentry.javascript.react-native', 'sentry.javascript.remix', 'sentry.javascript.nextjs'}

def source_map_debug(project, event_id, exception_idx, frame_idx):
    if False:
        i = 10
        return i + 15
    event = eventstore.backend.get_event_by_id(project.id, event_id)
    if event is None:
        raise NotFound(detail='Event not found')
    try:
        if 'exception' not in event.interfaces:
            raise ParseError(detail='Event does not contain an exception')
        exception = event.interfaces['exception'].values[exception_idx]
    except IndexError:
        raise ParseError(detail="Query parameter 'exception_idx' is out of bounds")
    (frame, filename, abs_path) = _get_frame_filename_and_path(exception, frame_idx)
    if frame.data and 'sourcemap' in frame.data:
        return SourceMapDebug()
    if event.platform == 'node' and frame.context_line:
        return SourceMapDebug()
    if event.platform == 'node' and (not abs_path.startswith(('/', 'app:', 'webpack:'))):
        return SourceMapDebug()
    sdk_info = event.data.get('sdk')
    can_use_debug_id = sdk_info and Version(sdk_info['version']) >= Version(JS_VERSION_FOR_DEBUG_ID) and (sdk_info['name'] not in NO_DEBUG_ID_FRAMEWORKS)
    release = None
    if not can_use_debug_id:
        try:
            release = _extract_release(event, project)
        except (SourceMapException, Release.DoesNotExist):
            return SourceMapDebug(SourceMapProcessingIssue.MISSING_RELEASE)
        num_artifacts = release.count_artifacts()
        if num_artifacts == 0:
            return SourceMapDebug(issue=SourceMapProcessingIssue.MISSING_SOURCEMAPS)
    urlparts = urlparse(abs_path)
    if event.platform != 'node' and (not (urlparts.scheme and urlparts.path)):
        return SourceMapDebug(issue=SourceMapProcessingIssue.URL_NOT_VALID, data={'absPath': abs_path})
    if release:
        release_artifacts = _get_releasefiles(release, project.organization.id)
        try:
            artifact = _find_matching_artifact(release_artifacts, urlparts, abs_path, filename, release, event)
            sourcemap_url = _discover_sourcemap_url(artifact, filename)
        except SourceMapException as e:
            return SourceMapDebug(issue=e.issue, data=e.data)
        if not sourcemap_url:
            return SourceMapDebug(issue=SourceMapProcessingIssue.SOURCEMAP_NOT_FOUND, data={'filename': filename})
        sourcemap_url = non_standard_url_join(abs_path, sourcemap_url)
        try:
            sourcemap_artifact = _find_matching_artifact(release_artifacts, urlparse(sourcemap_url), sourcemap_url, filename, release, event)
        except SourceMapException as e:
            return SourceMapDebug(issue=e.issue, data=e.data)
        if not sourcemap_artifact.file.getfile().read():
            return SourceMapDebug(issue=SourceMapProcessingIssue.SOURCEMAP_NOT_FOUND, data={'filename': filename})
    if can_use_debug_id:
        return SourceMapDebug(issue=SourceMapProcessingIssue.DEBUG_ID_NO_SOURCEMAPS)
    return SourceMapDebug()

def _get_frame_filename_and_path(exception, frame_idx):
    if False:
        for i in range(10):
            print('nop')
    frame_list = exception.stacktrace.frames
    try:
        frame = frame_list[frame_idx]
    except IndexError:
        raise ParseError(detail="Query parameter 'frame_idx' is out of bounds")
    filename = frame.filename
    abs_path = frame.abs_path
    return (frame, filename, abs_path)

def _find_matches(release_artifacts, abs_path, unified_path, filename, release, event):
    if False:
        for i in range(10):
            print('nop')
    full_matches = [artifact for artifact in release_artifacts if (artifact.name == unified_path or artifact.name == abs_path) and _verify_dist_matches(release, event, artifact, filename)]
    partial_matches = _find_partial_matches(unified_path, release_artifacts)
    return (full_matches, partial_matches)

def _find_partial_matches(unified_path, artifacts):
    if False:
        i = 10
        return i + 15
    filename = unified_path.split('/')[-1]
    filename_matches = [artifact for artifact in artifacts if artifact.name.split('/')[-1] == filename]
    artifact_names = [artifact.name.split('/') for artifact in filename_matches]
    while any(artifact_names):
        for i in range(len(artifact_names)):
            if unified_path.endswith('/'.join(artifact_names[i])):
                return [filename_matches[i]]
            artifact_names[i] = artifact_names[i][1:]
    return []

def _extract_release(event, project):
    if False:
        while True:
            i = 10
    release_version = event.get_tag('sentry:release')
    if not release_version:
        raise SourceMapException(SourceMapProcessingIssue.MISSING_RELEASE)
    return Release.objects.get(organization=project.organization, version=release_version)

def _verify_dist_matches(release, event, artifact, filename):
    if False:
        for i in range(10):
            print('nop')
    try:
        if event.dist is None and artifact.dist_id is None:
            return True
        dist = Distribution.objects.get(release=release, name=event.dist)
    except Distribution.DoesNotExist:
        raise SourceMapException(SourceMapProcessingIssue.DIST_MISMATCH, {'eventDist': event.dist, 'filename': filename})
    if artifact.dist_id != dist.id:
        raise SourceMapException(SourceMapProcessingIssue.DIST_MISMATCH, {'eventDist': dist.id, 'artifactDist': artifact.dist_id, 'filename': filename})
    return True

def _find_matching_artifact(release_artifacts, urlparts, abs_path, filename, release, event):
    if False:
        for i in range(10):
            print('nop')
    unified_path = _unify_url(urlparts)
    (full_matches, partial_matches) = _find_matches(release_artifacts, abs_path, unified_path, filename, release, event)
    artifact_names = [artifact.name for artifact in release_artifacts]
    if len(full_matches) == 0:
        if len(partial_matches) > 0:
            partial_match = partial_matches[0]
            url_prefix = _find_url_prefix(filename, partial_match.name)
            raise SourceMapException(SourceMapProcessingIssue.PARTIAL_MATCH, {'absPath': abs_path, 'partialMatchPath': partial_match.name, 'filename': filename, 'unifiedPath': unified_path, 'urlPrefix': url_prefix, 'artifactNames': artifact_names})
        raise SourceMapException(SourceMapProcessingIssue.NO_URL_MATCH, {'absPath': abs_path, 'filename': filename, 'unifiedPath': unified_path, 'artifactNames': artifact_names})
    return full_matches[0]

def _discover_sourcemap_url(artifact, filename):
    if False:
        while True:
            i = 10
    file = artifact.file
    sourcemap_header = file.headers.get('Sourcemap', file.headers.get('X-SourceMap'))
    sourcemap_header = force_bytes(sourcemap_header) if sourcemap_header is not None else None
    try:
        sourcemap = find_sourcemap(sourcemap_header, file.getfile().read())
    except AssertionError:
        raise SourceMapException(SourceMapProcessingIssue.SOURCEMAP_NOT_FOUND, {'filename': filename})
    return force_str(sourcemap) if sourcemap is not None else None

def _unify_url(urlparts):
    if False:
        return 10
    return '~' + urlparts.path

def _get_releasefiles(release, organization_id):
    if False:
        print('Hello World!')
    data_sources = []
    file_list = ReleaseFile.public_objects.filter(release_id=release.id).exclude(artifact_count=0)
    file_list = file_list.select_related('file').order_by('name')
    data_sources.extend(list(file_list.order_by('name')))
    dists = Distribution.objects.filter(organization_id=organization_id, release=release)
    for dist in list(dists) + [None]:
        try:
            artifact_index = read_artifact_index(release, dist, artifact_count__gt=0)
        except Exception:
            artifact_index = None
        if artifact_index is not None:
            files = artifact_index.get('files', {})
            source = ArtifactSource(dist, files, [], [])
            data_sources.extend(source[:])
    return data_sources

def _find_url_prefix(filepath, artifact_name):
    if False:
        for i in range(10):
            print('nop')
    idx = artifact_name.find(filepath)
    if idx != -1:
        return artifact_name[:idx]
    filepath = filepath.split('/')
    artifact_name = artifact_name.split('/')
    if len(filepath) == len(artifact_name):
        matches = [filepath[i] != artifact_name[i] for i in range(len(filepath))]
        if sum(matches) == 1:
            idx = matches.index(True)
            return artifact_name[idx] + '/' if idx != -1 else None
    if len(filepath) + 1 == len(artifact_name):
        filepath = set(filepath)
        artifact_name = set(artifact_name)
        differences = list(filepath.symmetric_difference(artifact_name))
        if len(differences) == 1:
            return '/'.join(differences + [''])

class SourceMapException(Exception):

    def __init__(self, issue, data=None):
        if False:
            return 10
        super().__init__(issue, data)
        self.issue = issue
        self.data = data

class SourceMapDebug:

    def __init__(self, issue=None, data=None):
        if False:
            i = 10
            return i + 15
        self.issue = issue
        self.data = data