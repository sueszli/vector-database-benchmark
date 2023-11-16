"""A facade for interfacing with multiple Galaxy instances."""
from __future__ import annotations
import typing as t
if t.TYPE_CHECKING:
    from ansible.galaxy.api import CollectionVersionMetadata
    from ansible.galaxy.collection.concrete_artifact_manager import ConcreteArtifactsManager
    from ansible.galaxy.dependency_resolution.dataclasses import Candidate, Requirement
from ansible.galaxy.api import GalaxyAPI, GalaxyError
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
display = Display()

class MultiGalaxyAPIProxy:
    """A proxy that abstracts talking to multiple Galaxy instances."""

    def __init__(self, apis, concrete_artifacts_manager, offline=False):
        if False:
            print('Hello World!')
        'Initialize the target APIs list.'
        self._apis = apis
        self._concrete_art_mgr = concrete_artifacts_manager
        self._offline = offline

    @property
    def is_offline_mode_requested(self):
        if False:
            i = 10
            return i + 15
        return self._offline

    def _assert_that_offline_mode_is_not_requested(self):
        if False:
            i = 10
            return i + 15
        if self.is_offline_mode_requested:
            raise NotImplementedError("The calling code is not supposed to be invoked in 'offline' mode.")

    def _get_collection_versions(self, requirement):
        if False:
            for i in range(10):
                print('nop')
        'Helper for get_collection_versions.\n\n        Yield api, version pairs for all APIs,\n        and reraise the last error if no valid API was found.\n        '
        if self._offline:
            return []
        found_api = False
        last_error = None
        api_lookup_order = (requirement.src,) if isinstance(requirement.src, GalaxyAPI) else self._apis
        for api in api_lookup_order:
            try:
                versions = api.get_collection_versions(requirement.namespace, requirement.name)
            except GalaxyError as api_err:
                last_error = api_err
            except Exception as unknown_err:
                display.warning('Skipping Galaxy server {server!s}. Got an unexpected error when getting available versions of collection {fqcn!s}: {err!s}'.format(server=api.api_server, fqcn=requirement.fqcn, err=to_text(unknown_err)))
                last_error = unknown_err
            else:
                found_api = True
                for version in versions:
                    yield (api, version)
        if not found_api and last_error is not None:
            raise last_error

    def get_collection_versions(self, requirement):
        if False:
            print('Hello World!')
        'Get a set of unique versions for FQCN on Galaxy servers.'
        if requirement.is_concrete_artifact:
            return {(self._concrete_art_mgr.get_direct_collection_version(requirement), requirement.src)}
        api_lookup_order = (requirement.src,) if isinstance(requirement.src, GalaxyAPI) else self._apis
        return set(((version, api) for (api, version) in self._get_collection_versions(requirement)))

    def get_collection_version_metadata(self, collection_candidate):
        if False:
            return 10
        'Retrieve collection metadata of a given candidate.'
        self._assert_that_offline_mode_is_not_requested()
        api_lookup_order = (collection_candidate.src,) if isinstance(collection_candidate.src, GalaxyAPI) else self._apis
        last_err: t.Optional[Exception]
        for api in api_lookup_order:
            try:
                version_metadata = api.get_collection_version_metadata(collection_candidate.namespace, collection_candidate.name, collection_candidate.ver)
            except GalaxyError as api_err:
                last_err = api_err
            except Exception as unknown_err:
                last_err = unknown_err
                display.warning('Skipping Galaxy server {server!s}. Got an unexpected error when getting available versions of collection {fqcn!s}: {err!s}'.format(server=api.api_server, fqcn=collection_candidate.fqcn, err=to_text(unknown_err)))
            else:
                self._concrete_art_mgr.save_collection_source(collection_candidate, version_metadata.download_url, version_metadata.artifact_sha256, api.token, version_metadata.signatures_url, version_metadata.signatures)
                return version_metadata
        raise last_err

    def get_collection_dependencies(self, collection_candidate):
        if False:
            while True:
                i = 10
        'Retrieve collection dependencies of a given candidate.'
        if collection_candidate.is_concrete_artifact:
            return self._concrete_art_mgr.get_direct_collection_dependencies(collection_candidate)
        return self.get_collection_version_metadata(collection_candidate).dependencies

    def get_signatures(self, collection_candidate):
        if False:
            i = 10
            return i + 15
        self._assert_that_offline_mode_is_not_requested()
        namespace = collection_candidate.namespace
        name = collection_candidate.name
        version = collection_candidate.ver
        last_err = None
        api_lookup_order = (collection_candidate.src,) if isinstance(collection_candidate.src, GalaxyAPI) else self._apis
        for api in api_lookup_order:
            try:
                return api.get_collection_signatures(namespace, name, version)
            except GalaxyError as api_err:
                last_err = api_err
            except Exception as unknown_err:
                last_err = unknown_err
                display.warning('Skipping Galaxy server {server!s}. Got an unexpected error when getting available versions of collection {fqcn!s}: {err!s}'.format(server=api.api_server, fqcn=collection_candidate.fqcn, err=to_text(unknown_err)))
        if last_err:
            raise last_err
        return []