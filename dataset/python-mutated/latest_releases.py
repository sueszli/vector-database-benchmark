import re
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple
from sentry.dynamic_sampling.rules.helpers.time_to_adoptions import Platform
from sentry.dynamic_sampling.rules.utils import BOOSTED_RELEASES_LIMIT, get_redis_client_for_ds
from sentry.models.project import Project
from sentry.models.release import Release
ENVIRONMENT_SEPARATOR = ':e:'
BOOSTED_RELEASE_CACHE_KEY_REGEX = re.compile('^ds::r:(?P<release_id>\\d+)(:e:(?P<environment>.+))?$')

def _get_environment_cache_key(environment: Optional[str]) -> str:
    if False:
        for i in range(10):
            print('nop')
    return f'{ENVIRONMENT_SEPARATOR}{environment}' if environment else ''

def _get_project_platform(project_id: int) -> Platform:
    if False:
        while True:
            i = 10
    try:
        return Platform(Project.objects.get(id=project_id).platform)
    except Project.DoesNotExist:
        return Platform()

@dataclass(frozen=True)
class BoostedRelease:
    """
    Class that represents a boosted release fetched from Redis.
    """
    id: int
    timestamp: float
    environment: Optional[str]
    cache_key: str

    def extend(self, release: Release, project_id: int) -> 'ExtendedBoostedRelease':
        if False:
            for i in range(10):
                print('nop')
        return ExtendedBoostedRelease(id=self.id, timestamp=self.timestamp, environment=self.environment, cache_key=self.cache_key, version=release.version, platform=_get_project_platform(project_id))

@dataclass(frozen=True)
class ExtendedBoostedRelease(BoostedRelease):
    """
    Class the represents a boosted release with added information that are injected after the base release is
    fetched from the cache.
    """
    version: str
    platform: Platform

    def is_active(self, current_timestamp: float) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return current_timestamp <= self.timestamp + self.platform.time_to_adoption

@dataclass
class BoostedReleases:
    """
    Class that hides the complexity of extending boosted releases.
    """
    boosted_releases: List[BoostedRelease] = field(default_factory=list)

    def add_release(self, cache_key: str, id: int, timestamp: float, environment: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        self.boosted_releases.append(BoostedRelease(cache_key=cache_key, id=id, timestamp=timestamp, environment=environment))

    def to_extended_boosted_releases(self, project_id: int) -> Tuple[List[ExtendedBoostedRelease], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        models = self._get_releases_models()
        current_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).timestamp()
        extended_boosted_releases = []
        expired_boosted_releases = []
        for boosted_release in self.boosted_releases:
            release_model = models.get(boosted_release.id, None)
            if release_model is None:
                expired_boosted_releases.append(boosted_release.cache_key)
                continue
            extended_boosted_release = boosted_release.extend(release=release_model, project_id=project_id)
            if extended_boosted_release.is_active(current_timestamp):
                extended_boosted_releases.append(extended_boosted_release)
            else:
                expired_boosted_releases.append(boosted_release.cache_key)
        return (extended_boosted_releases, expired_boosted_releases)

    def _get_last_release_ids(self) -> List[int]:
        if False:
            while True:
                i = 10
        return [boosted_release.id for boosted_release in self.boosted_releases]

    def _get_releases_models(self) -> Dict[int, Release]:
        if False:
            return 10
        return {release.id: release for release in Release.objects.filter(id__in=self._get_last_release_ids())}

class ProjectBoostedReleases:
    """
    Class responsible of hiding the complexity of handling boosted releases in the Redis hash. In addition, it provides
    all the logic to handle an upper bound in the number of boosted releases that can be simultaneously be added to
    a specific project.
    """
    BOOSTED_RELEASES_HASH_EXPIRATION = 60 * 60 * 1000

    def __init__(self, project_id: int):
        if False:
            return 10
        self.redis_client = get_redis_client_for_ds()
        self.project_id = project_id
        self.project_platform = _get_project_platform(self.project_id)

    @property
    def has_boosted_releases(self) -> bool:
        if False:
            return 10
        '\n        Checks whether a specific project has boosted releases.\n        '
        cache_key = self._generate_cache_key_for_boosted_releases_hash()
        return bool(self.redis_client.exists(cache_key) == 1)

    def add_boosted_release(self, release_id: int, environment: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Adds a release to the boosted releases hash with the boosting timestamp set to the current time, signaling that\n        the boosts starts now.\n        '
        self._remove_lrb_if_limit_is_reached()
        cache_key = self._generate_cache_key_for_boosted_releases_hash()
        self.redis_client.hset(cache_key, self._generate_cache_key_for_boosted_release(release_id, environment), datetime.utcnow().replace(tzinfo=timezone.utc).timestamp())
        self.redis_client.pexpire(cache_key, self.BOOSTED_RELEASES_HASH_EXPIRATION)

    def get_extended_boosted_releases(self) -> List[ExtendedBoostedRelease]:
        if False:
            print('Hello World!')
        '\n        Returns a list of boosted releases augmented with additional information such as release version and platform.\n        In addition, this function performs the cleanup of expired boosted releases.\n        '
        (active, expired) = self._get_boosted_releases().to_extended_boosted_releases(self.project_id)
        if expired:
            self.redis_client.hdel(self._generate_cache_key_for_boosted_releases_hash(), *expired)
        return active

    def _get_boosted_releases(self) -> BoostedReleases:
        if False:
            i = 10
            return i + 15
        '\n        Returns all the boosted releases and parses them based on key and value data.\n\n        This method should not be called directly as the boosted releases are not extended, thus they contain only a\n        subset of information.\n        '
        boosted_releases = BoostedReleases()
        for (boosted_release_cache_key, timestamp) in self.redis_client.hgetall(self._generate_cache_key_for_boosted_releases_hash()).items():
            extracted_data = self._extract_data_from_cache_key(boosted_release_cache_key)
            if extracted_data:
                (release_id, environment) = extracted_data
                boosted_releases.add_release(cache_key=boosted_release_cache_key, id=release_id, timestamp=float(timestamp), environment=environment)
        return boosted_releases

    def _remove_lrb_if_limit_is_reached(self) -> None:
        if False:
            print('Hello World!')
        '\n        Removes all the expired releases and also the least recently boosted release in case the limit of boosted\n        releases is reached.\n\n        For efficiency reasons, this function performs two things simultaneously:\n        1. It counts the number of active releases and keeps track of expired releases for deletion\n        2. It finds the least recently boosted active release to remove in case the limit of boosted release is reached\n        '
        cache_key = self._generate_cache_key_for_boosted_releases_hash()
        boosted_releases = self.redis_client.hgetall(cache_key)
        current_timestamp = datetime.utcnow().replace(tzinfo=timezone.utc).timestamp()
        LRBRelease = namedtuple('LRBRelease', ['key', 'timestamp'])
        lrb_release = None
        active_releases = 0
        keys_to_delete = []
        for (boosted_release_key, timestamp) in boosted_releases.items():
            timestamp = float(timestamp)
            if current_timestamp <= timestamp + self.project_platform.time_to_adoption:
                if lrb_release is None or timestamp < lrb_release.timestamp:
                    lrb_release = LRBRelease(key=boosted_release_key, timestamp=timestamp)
                active_releases += 1
            else:
                keys_to_delete.append(boosted_release_key)
        if active_releases >= BOOSTED_RELEASES_LIMIT and lrb_release:
            keys_to_delete.append(lrb_release.key)
        if keys_to_delete:
            self.redis_client.hdel(cache_key, *keys_to_delete)

    def _generate_cache_key_for_boosted_releases_hash(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'ds::p:{self.project_id}:boosted_releases'

    @staticmethod
    def _generate_cache_key_for_boosted_release(release_id: int, environment: Optional[str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'ds::r:{release_id}{_get_environment_cache_key(environment)}'

    @staticmethod
    def _extract_data_from_cache_key(cache_key: str) -> Optional[Tuple[int, Optional[str]]]:
        if False:
            while True:
                i = 10
        '\n        Extracts the release id and the environment from the cache key, in order to avoid storing the metadata also\n        in the value field.\n        '
        if (match := BOOSTED_RELEASE_CACHE_KEY_REGEX.match(cache_key)) is not None:
            release_id = match['release_id']
            environment = match['environment']
            return (int(release_id), environment)
        try:
            release_id = int(cache_key)
        except ValueError:
            return None
        else:
            return (release_id, None)

@dataclass(frozen=True)
class LatestReleaseParams:
    project: Project
    release: Release
    environment: Optional[str]

class LatestReleaseBias:
    """
    Class responsible of tracking all the (release, environment) pairs that have been observed in order to compute
    whether a certain release should be boosted.
    """
    OBSERVED_VALUE = '1'
    ONE_DAY_TIMEOUT_MS = 60 * 60 * 24 * 1000

    def __init__(self, latest_release_params: LatestReleaseParams):
        if False:
            while True:
                i = 10
        self.redis_client = get_redis_client_for_ds()
        self.latest_release_params = latest_release_params
        self.project_boosted_releases = ProjectBoostedReleases(self.latest_release_params.project.id)

    def observe_release(self, on_boosted_release_added: Callable[[], None]) -> None:
        if False:
            print('Hello World!')
        if not self._is_already_observed() and self._is_latest_release():
            self.project_boosted_releases.add_boosted_release(self.latest_release_params.release.id, self.latest_release_params.environment)
            on_boosted_release_added()

    def _is_already_observed(self) -> bool:
        if False:
            i = 10
            return i + 15
        cache_key = self._generate_cache_key_for_observed_release()
        release_observed = self.redis_client.getset(name=cache_key, value=self.OBSERVED_VALUE)
        self.redis_client.pexpire(cache_key, self.ONE_DAY_TIMEOUT_MS)
        return bool(release_observed == self.OBSERVED_VALUE)

    def _is_latest_release(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        incoming_release_date = self._get_release_date_from_incoming_release()
        latest_release_date = self._get_release_date_from_latest_release()
        if incoming_release_date is not None:
            if latest_release_date is None or incoming_release_date >= latest_release_date:
                self._update_latest_release_date(timestamp=incoming_release_date)
                return True
        return False

    def _update_latest_release_date(self, timestamp: float) -> None:
        if False:
            print('Hello World!')
        cache_key = self._generate_cache_key_for_project_latest_release()
        self.redis_client.set(cache_key, timestamp)

    def _get_release_date_from_incoming_release(self) -> Optional[float]:
        if False:
            print('Hello World!')
        release = self.latest_release_params.release
        if release.date_released:
            return float(release.date_released.timestamp())
        elif release.date_added:
            return float(release.date_added.timestamp())
        return None

    def _get_release_date_from_latest_release(self) -> Optional[float]:
        if False:
            while True:
                i = 10
        cache_key = self._generate_cache_key_for_project_latest_release()
        timestamp = self.redis_client.get(name=cache_key)
        return float(timestamp) if timestamp else None

    def _generate_cache_key_for_project_latest_release(self) -> str:
        if False:
            print('Hello World!')
        return f'ds::p:{self.latest_release_params.project.id}:latest_release'

    def _generate_cache_key_for_observed_release(self) -> str:
        if False:
            return 10
        return f'ds::p:{self.latest_release_params.project.id}:r:{self.latest_release_params.release.id}{_get_environment_cache_key(self.latest_release_params.environment)}'