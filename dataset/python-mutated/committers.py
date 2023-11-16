from __future__ import annotations
import operator
from collections import defaultdict
from enum import Enum
from functools import reduce
from typing import Any, Iterator, List, Mapping, MutableMapping, Sequence, Set, Tuple, TypedDict, Union
from django.core.cache import cache
from django.db.models import Q
from sentry.api.serializers import serialize
from sentry.api.serializers.models.commit import CommitSerializer, get_users_for_commits
from sentry.api.serializers.models.release import Author
from sentry.eventstore.models import Event
from sentry.models.commit import Commit
from sentry.models.commitfilechange import CommitFileChange
from sentry.models.group import Group
from sentry.models.groupowner import GroupOwner, GroupOwnerType
from sentry.models.project import Project
from sentry.models.release import Release
from sentry.models.releasecommit import ReleaseCommit
from sentry.services.hybrid_cloud.user.service import user_service
from sentry.utils import metrics
from sentry.utils.event_frames import find_stack_frames, get_sdk_name, munged_filename_and_frames
from sentry.utils.hashlib import hash_values
PATH_SEPARATORS = frozenset(['/', '\\'])

def tokenize_path(path: str) -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    for sep in PATH_SEPARATORS:
        if sep in path:
            return reversed([x for x in path.split(sep) if x != ''])
    else:
        return iter([path])

def score_path_match_length(path_a: str, path_b: str) -> int:
    if False:
        return 10
    score = 0
    for (a, b) in zip(tokenize_path(path_a), tokenize_path(path_b)):
        if a.lower() != b.lower():
            break
        score += 1
    return score

def get_frame_paths(event: Event) -> Union[Any, Sequence[Any]]:
    if False:
        while True:
            i = 10
    return find_stack_frames(event.data)

def release_cache_key(release: Release) -> str:
    if False:
        print('Hello World!')
    return f'release_commits:{release.id}'

def _get_commits(releases: Sequence[Release]) -> Sequence[Commit]:
    if False:
        while True:
            i = 10
    commits = []
    fetched = cache.get_many([release_cache_key(release) for release in releases])
    if fetched:
        missed = []
        for release in releases:
            cached_commits = fetched.get(release_cache_key(release))
            if cached_commits is None:
                missed.append(release)
            else:
                commits += [c for c in cached_commits if c not in commits]
    else:
        missed = list(releases)
    if missed:
        release_commits = ReleaseCommit.objects.filter(release__in=missed).select_related('commit', 'release', 'commit__author')
        to_cache = defaultdict(list)
        for rc in release_commits:
            to_cache[release_cache_key(rc.release)].append(rc.commit)
            if rc.commit not in commits:
                commits.append(rc.commit)
        cache.set_many(to_cache)
    return commits

def _get_commit_file_changes(commits: Sequence[Commit], path_name_set: Set[str]) -> Sequence[CommitFileChange]:
    if False:
        return 10
    filenames = {next(tokenize_path(path), None) for path in path_name_set}
    filenames = {path for path in filenames if path is not None}
    if not len(filenames):
        return []
    path_query = reduce(operator.or_, (Q(filename__iendswith=path) for path in filenames))
    commit_file_change_matches = CommitFileChange.objects.filter(path_query, commit__in=commits)
    return list(commit_file_change_matches)

def _match_commits_path(commit_file_changes: Sequence[CommitFileChange], path: str) -> Sequence[Tuple[Commit, int]]:
    if False:
        for i in range(10):
            print('nop')
    matching_commits: MutableMapping[int, Tuple[Commit, int]] = {}
    best_score = 1
    for file_change in commit_file_changes:
        score = score_path_match_length(file_change.filename, path)
        if score > best_score:
            best_score = score
            matching_commits = {}
        if score == best_score:
            if score == 1 and len(list(tokenize_path(file_change.filename))) > 1:
                continue
            matching_commits[file_change.commit.id] = (file_change.commit, score)
    return list(matching_commits.values())

class AuthorCommits(TypedDict):
    author: Union[Author, None]
    commits: Sequence[Tuple[Commit, int]]

class AuthorCommitsSerialized(TypedDict):
    author: Union[Author, None]
    commits: Sequence[MutableMapping[str, Any]]

class AuthorCommitsWithReleaseSerialized(TypedDict):
    author: Author
    commits: Sequence[MutableMapping[str, Any]]
    release: Release

class AnnotatedFrame(TypedDict):
    frame: str
    commits: Sequence[Tuple[Commit, int]]

class SuspectCommitType(Enum):
    """Used to distinguish old suspect commits from the newer commits obtained via the commit_context."""
    RELEASE_COMMIT = 'via commit in release'
    INTEGRATION_COMMIT = 'via SCM integration'

def _get_committers(annotated_frames: Sequence[AnnotatedFrame], commits: Sequence[Tuple[Commit, int]]) -> Sequence[AuthorCommits]:
    if False:
        i = 10
        return i + 15
    committers: MutableMapping[int, int] = defaultdict(int)
    limit = 5
    for annotated_frame in annotated_frames:
        if limit == 0:
            break
        for (commit, score) in annotated_frame['commits']:
            if not commit.author_id:
                continue
            committers[commit.author_id] += limit
            limit -= 1
            if limit == 0:
                break
    author_users: Mapping[str, Author] = get_users_for_commits([c for (c, _) in commits])
    return [{'author': author_users.get(str(author_id)), 'commits': [(commit, score) for (commit, score) in commits if commit.author_id == author_id]} for (author_id, _) in sorted(committers.items(), key=operator.itemgetter(1))]

def get_previous_releases(project: Project, start_version: str, limit: int=5) -> Union[Any, Sequence[Release]]:
    if False:
        return 10
    key = 'get_previous_releases:1:%s' % hash_values([project.id, start_version, limit])
    rv = cache.get(key)
    if rv is None:
        try:
            first_release = Release.objects.filter(organization_id=project.organization_id, version=start_version, projects=project).get()
        except Release.DoesNotExist:
            rv = []
        else:
            start_date = first_release.date_released or first_release.date_added
            rv = list(Release.objects.raw('\n                        SELECT sr.*\n                        FROM sentry_release as sr\n                        INNER JOIN (\n                            SELECT release_id\n                            FROM sentry_release_project\n                            WHERE project_id = %s\n                            AND sentry_release_project.release_id <= %s\n                            ORDER BY release_id desc\n                            LIMIT 100\n                        ) AS srp ON (sr.id = srp.release_id)\n                        WHERE sr.organization_id = %s\n                        AND coalesce(sr.date_released, sr.date_added) <= %s\n                        ORDER BY coalesce(sr.date_released, sr.date_added) DESC\n                        LIMIT %s;\n                    ', [project.id, first_release.id, project.organization_id, start_date, limit]))
        cache.set(key, rv, 60)
    return rv

def get_event_file_committers(project: Project, group_id: int, event_frames: Sequence[Mapping[str, Any]], event_platform: str, frame_limit: int=25, sdk_name: str | None=None) -> Sequence[AuthorCommits]:
    if False:
        while True:
            i = 10
    group = Group.objects.get_from_cache(id=group_id)
    first_release_version = group.get_first_release()
    if not first_release_version:
        raise Release.DoesNotExist
    releases = get_previous_releases(project, first_release_version)
    if not releases:
        raise Release.DoesNotExist
    commits = _get_commits(releases)
    if not commits:
        raise Commit.DoesNotExist
    frames = event_frames or []
    munged = munged_filename_and_frames(event_platform, frames, 'munged_filename', sdk_name)
    if munged:
        frames = munged[1]
    app_frames = [frame for frame in frames if frame.get('in_app')][-frame_limit:]
    if not app_frames:
        app_frames = [frame for frame in frames][-frame_limit:]
    path_set = {str(f) for f in (get_stacktrace_path_from_event_frame(frame) for frame in app_frames) if f}
    file_changes: Sequence[CommitFileChange] = _get_commit_file_changes(commits, path_set) if path_set else []
    commit_path_matches: Mapping[str, Sequence[Tuple[Commit, int]]] = {path: _match_commits_path(file_changes, path) for path in path_set}
    annotated_frames: Sequence[AnnotatedFrame] = [{'frame': str(frame), 'commits': commit_path_matches.get(str(get_stacktrace_path_from_event_frame(frame)), [])} for frame in app_frames]
    relevant_commits: Sequence[Tuple[Commit, int]] = [match for matches in commit_path_matches.values() for match in matches]
    return _get_committers(annotated_frames, relevant_commits)

def get_serialized_event_file_committers(project: Project, event: Event, frame_limit: int=25) -> Sequence[AuthorCommitsSerialized]:
    if False:
        return 10
    group_owners = GroupOwner.objects.filter(group_id=event.group_id, project=project, organization_id=project.organization_id, type=GroupOwnerType.SUSPECT_COMMIT.value, context__isnull=False).order_by('-date_added')
    if len(group_owners) > 0:
        owner = next(filter(lambda go: go.context.get('commitId'), group_owners), None)
        if not owner:
            return []
        commit = Commit.objects.get(id=owner.context.get('commitId'))
        commit_author = commit.author
        if not commit_author:
            return []
        author = {'email': commit_author.email, 'name': commit_author.name}
        if owner.user_id is not None:
            serialized_owners = user_service.serialize_many(filter={'user_ids': [owner.user_id]})
            if serialized_owners:
                author = serialized_owners[0]
        return [{'author': author, 'commits': [serialize(commit, serializer=CommitSerializer(exclude=['author'], type=SuspectCommitType.INTEGRATION_COMMIT.value))]}]
    else:
        event_frames = get_frame_paths(event)
        sdk_name = get_sdk_name(event.data)
        committers = get_event_file_committers(project, event.group_id, event_frames, event.platform, frame_limit=frame_limit, sdk_name=sdk_name)
        commits = [commit for committer in committers for commit in committer['commits']]
        serialized_commits: Sequence[MutableMapping[str, Any]] = serialize([c for (c, score) in commits], serializer=CommitSerializer(exclude=['author'], type=SuspectCommitType.RELEASE_COMMIT.value))
        serialized_commits_by_id = {}
        for ((commit, score), serialized_commit) in zip(commits, serialized_commits):
            serialized_commit['score'] = score
            serialized_commits_by_id[commit.id] = serialized_commit
        serialized_committers: List[AuthorCommitsSerialized] = []
        for committer in committers:
            commit_ids = [commit.id for (commit, _) in committer['commits']]
            commits_result = [serialized_commits_by_id[commit_id] for commit_id in commit_ids]
            serialized_committers.append({'author': committer['author'], 'commits': dedupe_commits(commits_result)})
        metrics.incr('feature.owners.has-committers', instance='hit' if committers else 'miss', skip_internal=False)
        return serialized_committers

def dedupe_commits(commits: Sequence[MutableMapping[str, Any]]) -> Sequence[MutableMapping[str, Any]]:
    if False:
        print('Hello World!')
    return list({c['id']: c for c in commits}.values())

def get_stacktrace_path_from_event_frame(frame: Mapping[str, Any]) -> str | None:
    if False:
        i = 10
        return i + 15
    "\n    Returns the filepath from a stacktrace's frame.\n    frame: Event frame\n    "
    return frame.get('munged_filename') or frame.get('filename') or frame.get('abs_path')