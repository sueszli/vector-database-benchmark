"""Integration of native symbolication with Apple App Store Connect.

Sentry can download dSYMs directly from App Store Connect, this is the support code for
this.
"""
import dataclasses
import logging
import pathlib
from typing import Any, Dict, List
import jsonschema
import requests
import sentry_sdk
from django.db import router, transaction
from sentry.lang.native.sources import APP_STORE_CONNECT_SCHEMA, secret_fields
from sentry.models.project import Project
from sentry.utils import json
from sentry.utils.appleconnect import appstore_connect
logger = logging.getLogger(__name__)
BuildInfo = appstore_connect.BuildInfo
NoDsymUrl = appstore_connect.NoDsymUrl
SYMBOL_SOURCES_PROP_NAME = 'sentry:symbol_sources'
SYMBOL_SOURCE_TYPE_NAME = 'appStoreConnect'

class InvalidConfigError(Exception):
    """Invalid configuration for the appStoreConnect symbol source."""
    pass

class PendingDsymsError(Exception):
    """dSYM url is currently unavailable."""
    pass

class NoDsymsError(Exception):
    """No dSYMs were found."""
    pass

@dataclasses.dataclass(frozen=True)
class AppStoreConnectConfig:
    """The symbol source configuration for an App Store Connect source.

    This is stored as a symbol source inside symbolSources project option.
    """
    type: str
    id: str
    name: str
    appconnectIssuer: str
    appconnectKey: str
    appconnectPrivateKey: str
    appName: str
    appId: str
    bundleId: str

    def __post_init__(self) -> None:
        if False:
            return 10
        for field in dataclasses.fields(self):
            if not getattr(self, field.name, None):
                raise ValueError(f'Missing field: {field.name}')

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'AppStoreConnectConfig':
        if False:
            for i in range(10):
                print('nop')
        "Creates a new instance from **deserialised** JSON data.\n\n        This will include the JSON schema validation.  You can safely use this to create and\n        validate the config as deserialised by both plain JSON deserialiser or by Django Rest\n        Framework's deserialiser.\n\n        :raises InvalidConfigError: if the data does not contain a valid App Store Connect\n           symbol source configuration.\n        "
        try:
            jsonschema.validate(data, APP_STORE_CONNECT_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidConfigError from e
        return cls(**data)

    @classmethod
    def from_project_config(cls, project: Project, config_id: str) -> 'AppStoreConnectConfig':
        if False:
            while True:
                i = 10
        'Creates a new instance from the symbol source configured in the project.\n\n        :raises KeyError: if the config is not found.\n        :raises InvalidConfigError if the stored config is somehow invalid.\n        '
        raw = project.get_option(SYMBOL_SOURCES_PROP_NAME)
        if not raw:
            raw = '[]'
        all_sources = json.loads(raw)
        for source in all_sources:
            if source.get('type') == SYMBOL_SOURCE_TYPE_NAME and source.get('id') == config_id:
                return cls.from_json(source)
        else:
            raise KeyError(f'No {SYMBOL_SOURCE_TYPE_NAME} symbol source found with id {config_id}')

    @staticmethod
    def all_config_ids(project: Project) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return the config IDs of all appStoreConnect symbol sources configured in the project.'
        raw = project.get_option(SYMBOL_SOURCES_PROP_NAME)
        if not raw:
            raw = '[]'
        all_sources = json.loads(raw)
        return [s.get('id') for s in all_sources if s.get('type') == SYMBOL_SOURCE_TYPE_NAME and s.get('id')]

    def to_json(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Creates a dict which can be serialised to JSON. This dict should only be\n        used internally and should never be sent to external clients, as it contains\n        the raw content of all of the secrets contained in the config.\n\n        The generated dict will be validated according to the schema.\n\n        :raises InvalidConfigError: if somehow the data in the class is not valid, this\n           should only occur if the class was created in a weird way.\n        '
        data = dict()
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            data[field.name] = value
        try:
            jsonschema.validate(data, APP_STORE_CONNECT_SCHEMA)
        except jsonschema.exceptions.ValidationError as e:
            raise InvalidConfigError from e
        return data

    def to_redacted_json(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Creates a dict which can be serialised to JSON. This should be used when the\n        config is meant to be passed to some external consumer, like the front end client.\n        This dict will have its secrets redacted.\n\n        :raises InvalidConfigError: if somehow the data in the class is not valid, this\n           should only occur if the class was created in a weird way.\n        '
        data = self.to_json()
        for to_redact in secret_fields('appStoreConnect'):
            if to_redact in data:
                data[to_redact] = {'hidden-secret': True}
        return data

    def update_project_symbol_source(self, project: Project, allow_multiple: bool) -> json.JSONData:
        if False:
            while True:
                i = 10
        "Updates this configuration in the Project's symbol sources.\n\n        If a symbol source of type ``appStoreConnect`` already exists the ID must match and it\n        will be updated.  If no ``appStoreConnect`` source exists yet it is added.\n\n        :param allow_multiple: Whether multiple appStoreConnect sources are allowed for this\n           project.\n\n        :returns: The new value of the sources.  Use this in a call to\n           `ProjectEndpoint.create_audit_entry()` to create an audit log.\n\n        :raises ValueError: if an ``appStoreConnect`` source already exists but the ID does not\n           match\n        "
        with transaction.atomic(router.db_for_write(Project)):
            all_sources_raw = project.get_option(SYMBOL_SOURCES_PROP_NAME)
            all_sources = json.loads(all_sources_raw) if all_sources_raw else []
            for (i, source) in enumerate(all_sources):
                if source.get('type') == SYMBOL_SOURCE_TYPE_NAME:
                    if source.get('id') == self.id:
                        all_sources[i] = self.to_json()
                        break
                    elif not allow_multiple:
                        raise ValueError('Existing appStoreConnect symbolSource config does not match id')
            else:
                all_sources.append(self.to_json())
            project.update_option(SYMBOL_SOURCES_PROP_NAME, json.dumps(all_sources))
        return all_sources

class AppConnectClient:
    """Client to interact with a single app from App Store Connect."""

    def __init__(self, api_credentials: appstore_connect.AppConnectCredentials, app_id: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Internal init, use one of the classmethods instead.'
        self._api_credentials = api_credentials
        self._session = requests.Session()
        self._app_id = app_id

    @classmethod
    def from_project(cls, project: Project, config_id: str) -> 'AppConnectClient':
        if False:
            print('Hello World!')
        "Creates a new client for the project's appStoreConnect symbol source.\n\n        This will load the configuration from the symbol sources for the project if a symbol\n        source of the ``appStoreConnect`` type can be found which also has matching\n        ``credentials_id``.\n        "
        config = AppStoreConnectConfig.from_project_config(project, config_id)
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: AppStoreConnectConfig) -> 'AppConnectClient':
        if False:
            for i in range(10):
                print('nop')
        "Creates a new client from an appStoreConnect symbol source config.\n\n        This config is normally stored as a symbol source of type ``appStoreConnect`` in a\n        project's ``sentry:symbol_sources`` property.\n        "
        api_credentials = appstore_connect.AppConnectCredentials(key_id=config.appconnectKey, key=config.appconnectPrivateKey, issuer_id=config.appconnectIssuer)
        return cls(api_credentials=api_credentials, app_id=config.appId)

    def list_builds(self) -> List[BuildInfo]:
        if False:
            i = 10
            return i + 15
        'Returns the available AppStore builds.'
        return appstore_connect.get_build_info(self._session, self._api_credentials, self._app_id, include_expired=True)

    def download_dsyms(self, build: BuildInfo, path: pathlib.Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Downloads the dSYMs from the build into the filename given by `path`.\n\n        The dSYMs are downloaded as a zipfile so when this call succeeds the file at `path`\n        will contain a zipfile.\n        '
        with sentry_sdk.start_span(op='dsym', description='Download dSYMs'):
            if not isinstance(build.dsym_url, str):
                if build.dsym_url is NoDsymUrl.NOT_NEEDED:
                    raise NoDsymsError
                elif build.dsym_url is NoDsymUrl.PENDING:
                    raise PendingDsymsError
                else:
                    raise ValueError(f'dSYM URL missing: {build.dsym_url}')
            logger.debug('Fetching dSYMs from: %s', build.dsym_url)
            appstore_connect.download_dsyms(self._session, self._api_credentials, build.dsym_url, path)