from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type
import dagger
import semver
from base_images import consts, published_image
from base_images.bases import AirbyteConnectorBaseImage
from base_images.python.bases import AirbytePythonConnectorBaseImage
from base_images.utils import docker
from connector_ops.utils import ConnectorLanguage
MANAGED_BASE_IMAGES = [AirbytePythonConnectorBaseImage]

@dataclass
class ChangelogEntry:
    version: semver.VersionInfo
    changelog_entry: str
    dockerfile_example: str

    def to_serializable_dict(self):
        if False:
            return 10
        return {'version': str(self.version), 'changelog_entry': self.changelog_entry, 'dockerfile_example': self.dockerfile_example}

    @staticmethod
    def from_dict(entry_dict: Dict):
        if False:
            while True:
                i = 10
        return ChangelogEntry(version=semver.VersionInfo.parse(entry_dict['version']), changelog_entry=entry_dict['changelog_entry'], dockerfile_example=entry_dict['dockerfile_example'])

@dataclass
class VersionRegistryEntry:
    published_docker_image: Optional[published_image.PublishedImage]
    changelog_entry: Optional[ChangelogEntry]
    version: semver.VersionInfo

    @property
    def published(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self.published_docker_image is not None

class VersionRegistry:

    def __init__(self, ConnectorBaseImageClass: Type[AirbyteConnectorBaseImage], entries: List[VersionRegistryEntry]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.ConnectorBaseImageClass: Type[AirbyteConnectorBaseImage] = ConnectorBaseImageClass
        self._entries: List[VersionRegistryEntry] = entries

    @staticmethod
    def get_changelog_dump_path(ConnectorBaseImageClass: Type[AirbyteConnectorBaseImage]) -> Path:
        if False:
            return 10
        'Returns the path where the changelog is dumped to disk.\n\n        Args:\n            ConnectorBaseImageClass (Type[AirbyteConnectorBaseImage]): The base image version class bound to the registry.\n\n        Returns:\n            Path: The path where the changelog JSON is dumped to disk.\n        '
        registries_dir = Path('generated/changelogs')
        registries_dir.mkdir(exist_ok=True, parents=True)
        return registries_dir / f"{ConnectorBaseImageClass.repository.replace('-', '_').replace('/', '_')}.json"

    @property
    def changelog_dump_path(self) -> Path:
        if False:
            i = 10
            return i + 15
        'Returns the path where the changelog JSON is dumped to disk.\n\n        Returns:\n            Path: The path where the changelog JSON is dumped to disk.\n        '
        return self.get_changelog_dump_path(self.ConnectorBaseImageClass)

    @staticmethod
    def get_changelog_entries(ConnectorBaseImageClass: Type[AirbyteConnectorBaseImage]) -> List[ChangelogEntry]:
        if False:
            i = 10
            return i + 15
        'Returns the changelog entries for a given base image version class.\n        The changelog entries are loaded from the checked in changelog dump JSON file.\n\n        Args:\n            ConnectorBaseImageClass (Type[AirbyteConnectorBaseImage]): The base image version class bound to the registry.\n\n        Returns:\n            List[ChangelogEntry]: The changelog entries for a given base image version class.\n        '
        change_log_dump_path = VersionRegistry.get_changelog_dump_path(ConnectorBaseImageClass)
        if not change_log_dump_path.exists():
            changelog_entries = []
        else:
            changelog_entries = [ChangelogEntry.from_dict(raw_entry) for raw_entry in json.loads(change_log_dump_path.read_text())]
        return changelog_entries

    @staticmethod
    async def get_all_published_base_images(dagger_client: dagger.Client, docker_credentials: Tuple[str, str], ConnectorBaseImageClass: Type[AirbyteConnectorBaseImage]) -> List[published_image.PublishedImage]:
        """Returns all the published base images for a given base image version class.

        Args:
            dagger_client (dagger.Client): The dagger client used to build the registry.
            docker_credentials (Tuple[str, str]): The docker credentials used to fetch published images from DockerHub.
            ConnectorBaseImageClass (Type[AirbyteConnectorBaseImage]): The base image version class bound to the registry.

        Returns:
            List[published_image.PublishedImage]: The published base images for a given base image version class.
        """
        crane_client = docker.CraneClient(dagger_client, docker_credentials)
        remote_registry = docker.RemoteRepository(crane_client, consts.REMOTE_REGISTRY, ConnectorBaseImageClass.repository)
        return await remote_registry.get_all_images()

    @staticmethod
    async def load(ConnectorBaseImageClass: Type[AirbyteConnectorBaseImage], dagger_client: dagger.Client, docker_credentials: Tuple[str, str]) -> VersionRegistry:
        """Instantiates a registry by fetching available versions from the remote registry and loading the changelog from disk.

        Args:
            ConnectorBaseImageClass (Type[AirbyteConnectorBaseImage]): The base image version class bound to the registry.

        Returns:
            VersionRegistry: The registry.
        """
        changelog_entries = VersionRegistry.get_changelog_entries(ConnectorBaseImageClass)
        changelog_entries_by_version = {entry.version: entry for entry in changelog_entries}
        published_docker_images = await VersionRegistry.get_all_published_base_images(dagger_client, docker_credentials, ConnectorBaseImageClass)
        published_docker_images_by_version = {image.version: image for image in published_docker_images}
        all_versions = set(changelog_entries_by_version.keys()) | set(published_docker_images_by_version.keys())
        registry_entries = []
        for version in all_versions:
            published_docker_image = published_docker_images_by_version.get(version)
            changelog_entry = changelog_entries_by_version.get(version)
            registry_entries.append(VersionRegistryEntry(published_docker_image, changelog_entry, version))
        return VersionRegistry(ConnectorBaseImageClass, registry_entries)

    def save_changelog(self):
        if False:
            while True:
                i = 10
        'Writes the changelog to disk. The changelog is dumped as a json file with a list of ChangelogEntry objects.'
        as_json = json.dumps([entry.changelog_entry.to_serializable_dict() for entry in self.entries if entry.changelog_entry])
        self.changelog_dump_path.write_text(as_json)

    def add_entry(self, new_entry: VersionRegistryEntry) -> List[VersionRegistryEntry]:
        if False:
            i = 10
            return i + 15
        'Registers a new entry in the registry and saves the changelog locally.\n\n        Args:\n            new_entry (VersionRegistryEntry): The new entry to register.\n\n        Returns:\n            List[VersionRegistryEntry]: All the entries sorted by version number in descending order.\n        '
        self._entries.append(new_entry)
        self.save_changelog()
        return self.entries

    @property
    def entries(self) -> List[VersionRegistryEntry]:
        if False:
            i = 10
            return i + 15
        'Returns all the base image versions sorted by version number in descending order.\n\n        Returns:\n            List[Type[VersionRegistryEntry]]: All the published versions sorted by version number in descending order.\n        '
        return sorted(self._entries, key=lambda entry: entry.version, reverse=True)

    @property
    def latest_entry(self) -> Optional[VersionRegistryEntry]:
        if False:
            return 10
        'Returns the latest entry this registry.\n        The latest entry is the one with the highest version number.\n        If no entry is available, returns None.\n        Returns:\n            Optional[VersionRegistryEntry]: The latest registry entry, or None if no entry is available.\n        '
        try:
            return self.entries[0]
        except IndexError:
            return None

    @property
    def latest_published_entry(self) -> Optional[VersionRegistryEntry]:
        if False:
            print('Hello World!')
        'Returns the latest published entry this registry.\n        The latest published entry is the one with the highest version number among the published entries.\n        If no entry is available, returns None.\n        Returns:\n            Optional[VersionRegistryEntry]: The latest published registry entry, or None if no entry is available.\n        '
        try:
            return [entry for entry in self.entries if entry.published][0]
        except IndexError:
            return None

    def get_entry_for_version(self, version: semver.VersionInfo) -> Optional[VersionRegistryEntry]:
        if False:
            print('Hello World!')
        'Returns the entry for a given version.\n        If no entry is available, returns None.\n        Returns:\n            Optional[VersionRegistryEntry]: The registry entry for the given version, or None if no entry is available.\n        '
        for entry in self.entries:
            if entry.version == version:
                return entry
        return None

    @property
    def latest_not_pre_released_published_entry(self) -> Optional[VersionRegistryEntry]:
        if False:
            i = 10
            return i + 15
        'Returns the latest entry with a not pre-released version in this registry which is published.\n        If no entry is available, returns None.\n        It is meant to be used externally to get the latest published version.\n        Returns:\n            Optional[VersionRegistryEntry]: The latest registry entry with a not pre-released version, or None if no entry is available.\n        '
        try:
            not_pre_release_published_entries = [entry for entry in self.entries if not entry.version.prerelease and entry.published]
            return not_pre_release_published_entries[0]
        except IndexError:
            return None

async def get_python_registry(dagger_client: dagger.Client, docker_credentials: Tuple[str, str]) -> VersionRegistry:
    return await VersionRegistry.load(AirbytePythonConnectorBaseImage, dagger_client, docker_credentials)

async def get_registry_for_language(dagger_client: dagger.Client, language: ConnectorLanguage, docker_credentials: Tuple[str, str]) -> VersionRegistry:
    """Returns the registry for a given language.
    It is meant to be used externally to get the registry for a given connector language.

    Args:
        dagger_client (dagger.Client): The dagger client used to build the registry.
        language (ConnectorLanguage): The connector language.
        docker_credentials (Tuple[str, str]): The docker credentials used to fetch published images from DockerHub.

    Raises:
        NotImplementedError: Raised if the registry for the given language is not implemented yet.

    Returns:
        VersionRegistry: The registry for the given language.
    """
    if language in [ConnectorLanguage.PYTHON, ConnectorLanguage.LOW_CODE]:
        return await get_python_registry(dagger_client, docker_credentials)
    else:
        raise NotImplementedError(f'Registry for language {language} is not implemented yet.')

async def get_all_registries(dagger_client: dagger.Client, docker_credentials: Tuple[str, str]) -> List[VersionRegistry]:
    return [await get_python_registry(dagger_client, docker_credentials)]