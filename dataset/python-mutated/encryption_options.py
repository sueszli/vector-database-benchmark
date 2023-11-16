"""Support for automatic client-side field level encryption."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional
try:
    import pymongocrypt
    _HAVE_PYMONGOCRYPT = True
except ImportError:
    _HAVE_PYMONGOCRYPT = False
from bson import int64
from pymongo.common import validate_is_mapping
from pymongo.errors import ConfigurationError
from pymongo.uri_parser import _parse_kms_tls_options
if TYPE_CHECKING:
    from pymongo.mongo_client import MongoClient
    from pymongo.typings import _DocumentTypeArg

class AutoEncryptionOpts:
    """Options to configure automatic client-side field level encryption."""

    def __init__(self, kms_providers: Mapping[str, Any], key_vault_namespace: str, key_vault_client: Optional[MongoClient[_DocumentTypeArg]]=None, schema_map: Optional[Mapping[str, Any]]=None, bypass_auto_encryption: bool=False, mongocryptd_uri: str='mongodb://localhost:27020', mongocryptd_bypass_spawn: bool=False, mongocryptd_spawn_path: str='mongocryptd', mongocryptd_spawn_args: Optional[list[str]]=None, kms_tls_options: Optional[Mapping[str, Any]]=None, crypt_shared_lib_path: Optional[str]=None, crypt_shared_lib_required: bool=False, bypass_query_analysis: bool=False, encrypted_fields_map: Optional[Mapping[str, Any]]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Options to configure automatic client-side field level encryption.\n\n        Automatic client-side field level encryption requires MongoDB 4.2\n        enterprise or a MongoDB 4.2 Atlas cluster. Automatic encryption is not\n        supported for operations on a database or view and will result in\n        error.\n\n        Although automatic encryption requires MongoDB 4.2 enterprise or a\n        MongoDB 4.2 Atlas cluster, automatic *decryption* is supported for all\n        users. To configure automatic *decryption* without automatic\n        *encryption* set ``bypass_auto_encryption=True``. Explicit\n        encryption and explicit decryption is also supported for all users\n        with the :class:`~pymongo.encryption.ClientEncryption` class.\n\n        See :ref:`automatic-client-side-encryption` for an example.\n\n        :Parameters:\n          - `kms_providers`: Map of KMS provider options. The `kms_providers`\n            map values differ by provider:\n\n              - `aws`: Map with "accessKeyId" and "secretAccessKey" as strings.\n                These are the AWS access key ID and AWS secret access key used\n                to generate KMS messages. An optional "sessionToken" may be\n                included to support temporary AWS credentials.\n              - `azure`: Map with "tenantId", "clientId", and "clientSecret" as\n                strings. Additionally, "identityPlatformEndpoint" may also be\n                specified as a string (defaults to \'login.microsoftonline.com\').\n                These are the Azure Active Directory credentials used to\n                generate Azure Key Vault messages.\n              - `gcp`: Map with "email" as a string and "privateKey"\n                as `bytes` or a base64 encoded string.\n                Additionally, "endpoint" may also be specified as a string\n                (defaults to \'oauth2.googleapis.com\'). These are the\n                credentials used to generate Google Cloud KMS messages.\n              - `kmip`: Map with "endpoint" as a host with required port.\n                For example: ``{"endpoint": "example.com:443"}``.\n              - `local`: Map with "key" as `bytes` (96 bytes in length) or\n                a base64 encoded string which decodes\n                to 96 bytes. "key" is the master key used to encrypt/decrypt\n                data keys. This key should be generated and stored as securely\n                as possible.\n\n          - `key_vault_namespace`: The namespace for the key vault collection.\n            The key vault collection contains all data keys used for encryption\n            and decryption. Data keys are stored as documents in this MongoDB\n            collection. Data keys are protected with encryption by a KMS\n            provider.\n          - `key_vault_client` (optional): By default the key vault collection\n            is assumed to reside in the same MongoDB cluster as the encrypted\n            MongoClient. Use this option to route data key queries to a\n            separate MongoDB cluster.\n          - `schema_map` (optional): Map of collection namespace ("db.coll") to\n            JSON Schema.  By default, a collection\'s JSONSchema is periodically\n            polled with the listCollections command. But a JSONSchema may be\n            specified locally with the schemaMap option.\n\n            **Supplying a `schema_map` provides more security than relying on\n            JSON Schemas obtained from the server. It protects against a\n            malicious server advertising a false JSON Schema, which could trick\n            the client into sending unencrypted data that should be\n            encrypted.**\n\n            Schemas supplied in the schemaMap only apply to configuring\n            automatic encryption for client side encryption. Other validation\n            rules in the JSON schema will not be enforced by the driver and\n            will result in an error.\n          - `bypass_auto_encryption` (optional): If ``True``, automatic\n            encryption will be disabled but automatic decryption will still be\n            enabled. Defaults to ``False``.\n          - `mongocryptd_uri` (optional): The MongoDB URI used to connect\n            to the *local* mongocryptd process. Defaults to\n            ``\'mongodb://localhost:27020\'``.\n          - `mongocryptd_bypass_spawn` (optional): If ``True``, the encrypted\n            MongoClient will not attempt to spawn the mongocryptd process.\n            Defaults to ``False``.\n          - `mongocryptd_spawn_path` (optional): Used for spawning the\n            mongocryptd process. Defaults to ``\'mongocryptd\'`` and spawns\n            mongocryptd from the system path.\n          - `mongocryptd_spawn_args` (optional): A list of string arguments to\n            use when spawning the mongocryptd process. Defaults to\n            ``[\'--idleShutdownTimeoutSecs=60\']``. If the list does not include\n            the ``idleShutdownTimeoutSecs`` option then\n            ``\'--idleShutdownTimeoutSecs=60\'`` will be added.\n          - `kms_tls_options` (optional):  A map of KMS provider names to TLS\n            options to use when creating secure connections to KMS providers.\n            Accepts the same TLS options as\n            :class:`pymongo.mongo_client.MongoClient`. For example, to\n            override the system default CA file::\n\n              kms_tls_options={\'kmip\': {\'tlsCAFile\': certifi.where()}}\n\n            Or to supply a client certificate::\n\n              kms_tls_options={\'kmip\': {\'tlsCertificateKeyFile\': \'client.pem\'}}\n          - `crypt_shared_lib_path` (optional): Override the path to load the crypt_shared library.\n          - `crypt_shared_lib_required` (optional): If True, raise an error if libmongocrypt is\n            unable to load the crypt_shared library.\n          - `bypass_query_analysis` (optional): If ``True``, disable automatic analysis\n            of outgoing commands. Set `bypass_query_analysis` to use explicit\n            encryption on indexed fields without the MongoDB Enterprise Advanced\n            licensed crypt_shared library.\n          - `encrypted_fields_map`: Map of collection namespace ("db.coll") to documents\n            that described the encrypted fields for Queryable Encryption. For example::\n\n                {\n                  "db.encryptedCollection": {\n                      "escCollection": "enxcol_.encryptedCollection.esc",\n                      "ecocCollection": "enxcol_.encryptedCollection.ecoc",\n                      "fields": [\n                          {\n                              "path": "firstName",\n                              "keyId": Binary.from_uuid(UUID(\'00000000-0000-0000-0000-000000000000\')),\n                              "bsonType": "string",\n                              "queries": {"queryType": "equality"}\n                          },\n                          {\n                              "path": "ssn",\n                              "keyId": Binary.from_uuid(UUID(\'04104104-1041-0410-4104-104104104104\')),\n                              "bsonType": "string"\n                          }\n                      ]\n                  }\n                }\n\n        .. versionchanged:: 4.2\n           Added `encrypted_fields_map` `crypt_shared_lib_path`, `crypt_shared_lib_required`,\n           and `bypass_query_analysis` parameters.\n\n        .. versionchanged:: 4.0\n           Added the `kms_tls_options` parameter and the "kmip" KMS provider.\n\n        .. versionadded:: 3.9\n        '
        if not _HAVE_PYMONGOCRYPT:
            raise ConfigurationError("client side encryption requires the pymongocrypt library: install a compatible version with: python -m pip install 'pymongo[encryption]'")
        if encrypted_fields_map:
            validate_is_mapping('encrypted_fields_map', encrypted_fields_map)
        self._encrypted_fields_map = encrypted_fields_map
        self._bypass_query_analysis = bypass_query_analysis
        self._crypt_shared_lib_path = crypt_shared_lib_path
        self._crypt_shared_lib_required = crypt_shared_lib_required
        self._kms_providers = kms_providers
        self._key_vault_namespace = key_vault_namespace
        self._key_vault_client = key_vault_client
        self._schema_map = schema_map
        self._bypass_auto_encryption = bypass_auto_encryption
        self._mongocryptd_uri = mongocryptd_uri
        self._mongocryptd_bypass_spawn = mongocryptd_bypass_spawn
        self._mongocryptd_spawn_path = mongocryptd_spawn_path
        if mongocryptd_spawn_args is None:
            mongocryptd_spawn_args = ['--idleShutdownTimeoutSecs=60']
        self._mongocryptd_spawn_args = mongocryptd_spawn_args
        if not isinstance(self._mongocryptd_spawn_args, list):
            raise TypeError('mongocryptd_spawn_args must be a list')
        if not any(('idleShutdownTimeoutSecs' in s for s in self._mongocryptd_spawn_args)):
            self._mongocryptd_spawn_args.append('--idleShutdownTimeoutSecs=60')
        self._kms_ssl_contexts = _parse_kms_tls_options(kms_tls_options)
        self._bypass_query_analysis = bypass_query_analysis

class RangeOpts:
    """Options to configure encrypted queries using the rangePreview algorithm."""

    def __init__(self, sparsity: int, min: Optional[Any]=None, max: Optional[Any]=None, precision: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        'Options to configure encrypted queries using the rangePreview algorithm.\n\n        .. note:: This feature is experimental only, and not intended for public use.\n\n        :Parameters:\n          - `sparsity`: An integer.\n          - `min`: A BSON scalar value corresponding to the type being queried.\n          - `max`: A BSON scalar value corresponding to the type being queried.\n          - `precision`: An integer, may only be set for double or decimal128 types.\n\n        .. versionadded:: 4.4\n        '
        self.min = min
        self.max = max
        self.sparsity = sparsity
        self.precision = precision

    @property
    def document(self) -> dict[str, Any]:
        if False:
            return 10
        doc = {}
        for (k, v) in [('sparsity', int64.Int64(self.sparsity)), ('precision', self.precision), ('min', self.min), ('max', self.max)]:
            if v is not None:
                doc[k] = v
        return doc