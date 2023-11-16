"""Tools for connecting to MongoDB.

.. seealso:: :doc:`/examples/high_availability` for examples of connecting
   to replica sets or sets of mongos servers.

To get a :class:`~pymongo.database.Database` instance from a
:class:`MongoClient` use either dictionary-style or attribute-style
access:

.. doctest::

  >>> from pymongo import MongoClient
  >>> c = MongoClient()
  >>> c.test_database
  Database(MongoClient(host=['localhost:27017'], document_class=dict, tz_aware=False, connect=True), 'test_database')
  >>> c["test-database"]
  Database(MongoClient(host=['localhost:27017'], document_class=dict, tz_aware=False, connect=True), 'test-database')
"""
from __future__ import annotations
import contextlib
import os
import weakref
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, ContextManager, FrozenSet, Generic, Iterator, Mapping, MutableMapping, NoReturn, Optional, Sequence, Type, TypeVar, Union, cast
import bson
from bson.codec_options import DEFAULT_CODEC_OPTIONS, TypeRegistry
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot, client_session, common, database, helpers, message, periodic_executor, uri_parser
from pymongo.change_stream import ChangeStream, ClusterChangeStream
from pymongo.client_options import ClientOptions
from pymongo.client_session import _EmptyServerSession
from pymongo.command_cursor import CommandCursor
from pymongo.errors import AutoReconnect, BulkWriteError, ConfigurationError, ConnectionFailure, InvalidOperation, NotPrimaryError, OperationFailure, PyMongoError, ServerSelectionTimeoutError, WaitQueueTimeoutError
from pymongo.lock import _HAS_REGISTER_AT_FORK, _create_lock, _release_locks
from pymongo.monitoring import ConnectionClosedReason
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_selectors import writable_server_selector
from pymongo.server_type import SERVER_TYPE
from pymongo.settings import TopologySettings
from pymongo.topology import Topology, _ErrorContext
from pymongo.topology_description import TOPOLOGY_TYPE, TopologyDescription
from pymongo.typings import ClusterTime, _Address, _CollationIn, _DocumentType, _DocumentTypeArg, _Pipeline
from pymongo.uri_parser import _check_options, _handle_option_deprecations, _handle_security_options, _normalize_options
from pymongo.write_concern import DEFAULT_WRITE_CONCERN, WriteConcern
if TYPE_CHECKING:
    import sys
    from types import TracebackType
    from bson.objectid import ObjectId
    from pymongo.bulk import _Bulk
    from pymongo.client_session import ClientSession, _ServerSession
    from pymongo.cursor import _ConnectionManager
    from pymongo.database import Database
    from pymongo.message import _CursorAddress, _GetMore, _Query
    from pymongo.pool import Connection
    from pymongo.read_concern import ReadConcern
    from pymongo.response import Response
    from pymongo.server import Server
    from pymongo.server_selectors import Selection
    if sys.version_info[:2] >= (3, 9):
        from collections.abc import Generator
    else:
        from typing import Generator
T = TypeVar('T')
_WriteCall = Callable[[Optional['ClientSession'], 'Connection', bool], T]
_ReadCall = Callable[[Optional['ClientSession'], 'Server', 'Connection', _ServerMode], T]

class MongoClient(common.BaseObject, Generic[_DocumentType]):
    """
    A client-side representation of a MongoDB cluster.

    Instances can represent either a standalone MongoDB server, a replica
    set, or a sharded cluster. Instances of this class are responsible for
    maintaining up-to-date state of the cluster, and possibly cache
    resources related to this, including background threads for monitoring,
    and connection pools.
    """
    HOST = 'localhost'
    PORT = 27017
    _constructor_args = ('document_class', 'tz_aware', 'connect')
    _clients: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def __init__(self, host: Optional[Union[str, Sequence[str]]]=None, port: Optional[int]=None, document_class: Optional[Type[_DocumentType]]=None, tz_aware: Optional[bool]=None, connect: Optional[bool]=None, type_registry: Optional[TypeRegistry]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Client for a MongoDB instance, a replica set, or a set of mongoses.\n\n        .. warning:: Starting in PyMongo 4.0, ``directConnection`` now has a default value of\n          False instead of None.\n          For more details, see the relevant section of the PyMongo 4.x migration guide:\n          :ref:`pymongo4-migration-direct-connection`.\n\n        The client object is thread-safe and has connection-pooling built in.\n        If an operation fails because of a network error,\n        :class:`~pymongo.errors.ConnectionFailure` is raised and the client\n        reconnects in the background. Application code should handle this\n        exception (recognizing that the operation failed) and then continue to\n        execute.\n\n        The `host` parameter can be a full `mongodb URI\n        <http://dochub.mongodb.org/core/connections>`_, in addition to\n        a simple hostname. It can also be a list of hostnames but no more\n        than one URI. Any port specified in the host string(s) will override\n        the `port` parameter. For username and\n        passwords reserved characters like \':\', \'/\', \'+\' and \'@\' must be\n        percent encoded following RFC 2396::\n\n            from urllib.parse import quote_plus\n\n            uri = "mongodb://%s:%s@%s" % (\n                quote_plus(user), quote_plus(password), host)\n            client = MongoClient(uri)\n\n        Unix domain sockets are also supported. The socket path must be percent\n        encoded in the URI::\n\n            uri = "mongodb://%s:%s@%s" % (\n                quote_plus(user), quote_plus(password), quote_plus(socket_path))\n            client = MongoClient(uri)\n\n        But not when passed as a simple hostname::\n\n            client = MongoClient(\'/tmp/mongodb-27017.sock\')\n\n        Starting with version 3.6, PyMongo supports mongodb+srv:// URIs. The\n        URI must include one, and only one, hostname. The hostname will be\n        resolved to one or more DNS `SRV records\n        <https://en.wikipedia.org/wiki/SRV_record>`_ which will be used\n        as the seed list for connecting to the MongoDB deployment. When using\n        SRV URIs, the `authSource` and `replicaSet` configuration options can\n        be specified using `TXT records\n        <https://en.wikipedia.org/wiki/TXT_record>`_. See the\n        `Initial DNS Seedlist Discovery spec\n        <https://github.com/mongodb/specifications/blob/master/source/\n        initial-dns-seedlist-discovery/initial-dns-seedlist-discovery.rst>`_\n        for more details. Note that the use of SRV URIs implicitly enables\n        TLS support. Pass tls=false in the URI to override.\n\n        .. note:: MongoClient creation will block waiting for answers from\n          DNS when mongodb+srv:// URIs are used.\n\n        .. note:: Starting with version 3.0 the :class:`MongoClient`\n          constructor no longer blocks while connecting to the server or\n          servers, and it no longer raises\n          :class:`~pymongo.errors.ConnectionFailure` if they are\n          unavailable, nor :class:`~pymongo.errors.ConfigurationError`\n          if the user\'s credentials are wrong. Instead, the constructor\n          returns immediately and launches the connection process on\n          background threads. You can check if the server is available\n          like this::\n\n            from pymongo.errors import ConnectionFailure\n            client = MongoClient()\n            try:\n                # The ping command is cheap and does not require auth.\n                client.admin.command(\'ping\')\n            except ConnectionFailure:\n                print("Server not available")\n\n        .. warning:: When using PyMongo in a multiprocessing context, please\n          read :ref:`multiprocessing` first.\n\n        .. note:: Many of the following options can be passed using a MongoDB\n          URI or keyword parameters. If the same option is passed in a URI and\n          as a keyword parameter the keyword parameter takes precedence.\n\n        :Parameters:\n          - `host` (optional): hostname or IP address or Unix domain socket\n            path of a single mongod or mongos instance to connect to, or a\n            mongodb URI, or a list of hostnames (but no more than one mongodb\n            URI). If `host` is an IPv6 literal it must be enclosed in \'[\'\n            and \']\' characters\n            following the RFC2732 URL syntax (e.g. \'[::1]\' for localhost).\n            Multihomed and round robin DNS addresses are **not** supported.\n          - `port` (optional): port number on which to connect\n          - `document_class` (optional): default class to use for\n            documents returned from queries on this client\n          - `tz_aware` (optional): if ``True``,\n            :class:`~datetime.datetime` instances returned as values\n            in a document by this :class:`MongoClient` will be timezone\n            aware (otherwise they will be naive)\n          - `connect` (optional): if ``True`` (the default), immediately\n            begin connecting to MongoDB in the background. Otherwise connect\n            on the first operation.\n          - `type_registry` (optional): instance of\n            :class:`~bson.codec_options.TypeRegistry` to enable encoding\n            and decoding of custom types.\n          - `datetime_conversion`: Specifies how UTC datetimes should be decoded\n            within BSON. Valid options include \'datetime_ms\' to return as a\n            DatetimeMS, \'datetime\' to return as a datetime.datetime and\n            raising a ValueError for out-of-range values, \'datetime_auto\' to\n            return DatetimeMS objects when the underlying datetime is\n            out-of-range and \'datetime_clamp\' to clamp to the minimum and\n            maximum possible datetimes. Defaults to \'datetime\'. See\n            :ref:`handling-out-of-range-datetimes` for details.\n\n          | **Other optional parameters can be passed as keyword arguments:**\n\n          - `directConnection` (optional): if ``True``, forces this client to\n             connect directly to the specified MongoDB host as a standalone.\n             If ``false``, the client connects to the entire replica set of\n             which the given MongoDB host(s) is a part. If this is ``True``\n             and a mongodb+srv:// URI or a URI containing multiple seeds is\n             provided, an exception will be raised.\n          - `maxPoolSize` (optional): The maximum allowable number of\n            concurrent connections to each connected server. Requests to a\n            server will block if there are `maxPoolSize` outstanding\n            connections to the requested server. Defaults to 100. Can be\n            either 0 or None, in which case there is no limit on the number\n            of concurrent connections.\n          - `minPoolSize` (optional): The minimum required number of concurrent\n            connections that the pool will maintain to each connected server.\n            Default is 0.\n          - `maxIdleTimeMS` (optional): The maximum number of milliseconds that\n            a connection can remain idle in the pool before being removed and\n            replaced. Defaults to `None` (no limit).\n          - `maxConnecting` (optional): The maximum number of connections that\n            each pool can establish concurrently. Defaults to `2`.\n          - `timeoutMS`: (integer or None) Controls how long (in\n            milliseconds) the driver will wait when executing an operation\n            (including retry attempts) before raising a timeout error.\n            ``0`` or ``None`` means no timeout.\n          - `socketTimeoutMS`: (integer or None) Controls how long (in\n            milliseconds) the driver will wait for a response after sending an\n            ordinary (non-monitoring) database operation before concluding that\n            a network error has occurred. ``0`` or ``None`` means no timeout.\n            Defaults to ``None`` (no timeout).\n          - `connectTimeoutMS`: (integer or None) Controls how long (in\n            milliseconds) the driver will wait during server monitoring when\n            connecting a new socket to a server before concluding the server\n            is unavailable. ``0`` or ``None`` means no timeout.\n            Defaults to ``20000`` (20 seconds).\n          - `server_selector`: (callable or None) Optional, user-provided\n            function that augments server selection rules. The function should\n            accept as an argument a list of\n            :class:`~pymongo.server_description.ServerDescription` objects and\n            return a list of server descriptions that should be considered\n            suitable for the desired operation.\n          - `serverSelectionTimeoutMS`: (integer) Controls how long (in\n            milliseconds) the driver will wait to find an available,\n            appropriate server to carry out a database operation; while it is\n            waiting, multiple server monitoring operations may be carried out,\n            each controlled by `connectTimeoutMS`. Defaults to ``30000`` (30\n            seconds).\n          - `waitQueueTimeoutMS`: (integer or None) How long (in milliseconds)\n            a thread will wait for a socket from the pool if the pool has no\n            free sockets. Defaults to ``None`` (no timeout).\n          - `heartbeatFrequencyMS`: (optional) The number of milliseconds\n            between periodic server checks, or None to accept the default\n            frequency of 10 seconds.\n          - `serverMonitoringMode`: (optional) The server monitoring mode to use.\n            Valid values are the strings: "auto", "stream", "poll". Defaults to "auto".\n          - `appname`: (string or None) The name of the application that\n            created this MongoClient instance. The server will log this value\n            upon establishing each connection. It is also recorded in the slow\n            query log and profile collections.\n          - `driver`: (pair or None) A driver implemented on top of PyMongo can\n            pass a :class:`~pymongo.driver_info.DriverInfo` to add its name,\n            version, and platform to the message printed in the server log when\n            establishing a connection.\n          - `event_listeners`: a list or tuple of event listeners. See\n            :mod:`~pymongo.monitoring` for details.\n          - `retryWrites`: (boolean) Whether supported write operations\n            executed within this MongoClient will be retried once after a\n            network error. Defaults to ``True``.\n            The supported write operations are:\n\n              - :meth:`~pymongo.collection.Collection.bulk_write`, as long as\n                :class:`~pymongo.operations.UpdateMany` or\n                :class:`~pymongo.operations.DeleteMany` are not included.\n              - :meth:`~pymongo.collection.Collection.delete_one`\n              - :meth:`~pymongo.collection.Collection.insert_one`\n              - :meth:`~pymongo.collection.Collection.insert_many`\n              - :meth:`~pymongo.collection.Collection.replace_one`\n              - :meth:`~pymongo.collection.Collection.update_one`\n              - :meth:`~pymongo.collection.Collection.find_one_and_delete`\n              - :meth:`~pymongo.collection.Collection.find_one_and_replace`\n              - :meth:`~pymongo.collection.Collection.find_one_and_update`\n\n            Unsupported write operations include, but are not limited to,\n            :meth:`~pymongo.collection.Collection.aggregate` using the ``$out``\n            pipeline operator and any operation with an unacknowledged write\n            concern (e.g. {w: 0})). See\n            https://github.com/mongodb/specifications/blob/master/source/retryable-writes/retryable-writes.rst\n          - `retryReads`: (boolean) Whether supported read operations\n            executed within this MongoClient will be retried once after a\n            network error. Defaults to ``True``.\n            The supported read operations are:\n            :meth:`~pymongo.collection.Collection.find`,\n            :meth:`~pymongo.collection.Collection.find_one`,\n            :meth:`~pymongo.collection.Collection.aggregate` without ``$out``,\n            :meth:`~pymongo.collection.Collection.distinct`,\n            :meth:`~pymongo.collection.Collection.count`,\n            :meth:`~pymongo.collection.Collection.estimated_document_count`,\n            :meth:`~pymongo.collection.Collection.count_documents`,\n            :meth:`pymongo.collection.Collection.watch`,\n            :meth:`~pymongo.collection.Collection.list_indexes`,\n            :meth:`pymongo.database.Database.watch`,\n            :meth:`~pymongo.database.Database.list_collections`,\n            :meth:`pymongo.mongo_client.MongoClient.watch`,\n            and :meth:`~pymongo.mongo_client.MongoClient.list_databases`.\n\n            Unsupported read operations include, but are not limited to\n            :meth:`~pymongo.database.Database.command` and any getMore\n            operation on a cursor.\n\n            Enabling retryable reads makes applications more resilient to\n            transient errors such as network failures, database upgrades, and\n            replica set failovers. For an exact definition of which errors\n            trigger a retry, see the `retryable reads specification\n            <https://github.com/mongodb/specifications/blob/master/source/retryable-reads/retryable-reads.rst>`_.\n\n          - `compressors`: Comma separated list of compressors for wire\n            protocol compression. The list is used to negotiate a compressor\n            with the server. Currently supported options are "snappy", "zlib"\n            and "zstd". Support for snappy requires the\n            `python-snappy <https://pypi.org/project/python-snappy/>`_ package.\n            zlib support requires the Python standard library zlib module. zstd\n            requires the `zstandard <https://pypi.org/project/zstandard/>`_\n            package. By default no compression is used. Compression support\n            must also be enabled on the server. MongoDB 3.6+ supports snappy\n            and zlib compression. MongoDB 4.2+ adds support for zstd.\n            See :ref:`network-compression-example` for details.\n          - `zlibCompressionLevel`: (int) The zlib compression level to use\n            when zlib is used as the wire protocol compressor. Supported values\n            are -1 through 9. -1 tells the zlib library to use its default\n            compression level (usually 6). 0 means no compression. 1 is best\n            speed. 9 is best compression. Defaults to -1.\n          - `uuidRepresentation`: The BSON representation to use when encoding\n            from and decoding to instances of :class:`~uuid.UUID`. Valid\n            values are the strings: "standard", "pythonLegacy", "javaLegacy",\n            "csharpLegacy", and "unspecified" (the default). New applications\n            should consider setting this to "standard" for cross language\n            compatibility. See :ref:`handling-uuid-data-example` for details.\n          - `unicode_decode_error_handler`: The error handler to apply when\n            a Unicode-related error occurs during BSON decoding that would\n            otherwise raise :exc:`UnicodeDecodeError`. Valid options include\n            \'strict\', \'replace\', \'backslashreplace\', \'surrogateescape\', and\n            \'ignore\'. Defaults to \'strict\'.\n          - `srvServiceName`: (string) The SRV service name to use for\n            "mongodb+srv://" URIs. Defaults to "mongodb". Use it like so::\n\n                MongoClient("mongodb+srv://example.com/?srvServiceName=customname")\n          - `srvMaxHosts`: (int) limits the number of mongos-like hosts a client will\n            connect to. More specifically, when a "mongodb+srv://" connection string\n            resolves to more than srvMaxHosts number of hosts, the client will randomly\n            choose an srvMaxHosts sized subset of hosts.\n\n\n          | **Write Concern options:**\n          | (Only set if passed. No default values.)\n\n          - `w`: (integer or string) If this is a replica set, write operations\n            will block until they have been replicated to the specified number\n            or tagged set of servers. `w=<int>` always includes the replica set\n            primary (e.g. w=3 means write to the primary and wait until\n            replicated to **two** secondaries). Passing w=0 **disables write\n            acknowledgement** and all other write concern options.\n          - `wTimeoutMS`: (integer) Used in conjunction with `w`. Specify a value\n            in milliseconds to control how long to wait for write propagation\n            to complete. If replication does not complete in the given\n            timeframe, a timeout exception is raised. Passing wTimeoutMS=0\n            will cause **write operations to wait indefinitely**.\n          - `journal`: If ``True`` block until write operations have been\n            committed to the journal. Cannot be used in combination with\n            `fsync`. Write operations will fail with an exception if this\n            option is used when the server is running without journaling.\n          - `fsync`: If ``True`` and the server is running without journaling,\n            blocks until the server has synced all data files to disk. If the\n            server is running with journaling, this acts the same as the `j`\n            option, blocking until write operations have been committed to the\n            journal. Cannot be used in combination with `j`.\n\n          | **Replica set keyword arguments for connecting with a replica set\n            - either directly or via a mongos:**\n\n          - `replicaSet`: (string or None) The name of the replica set to\n            connect to. The driver will verify that all servers it connects to\n            match this name. Implies that the hosts specified are a seed list\n            and the driver should attempt to find all members of the set.\n            Defaults to ``None``.\n\n          | **Read Preference:**\n\n          - `readPreference`: The replica set read preference for this client.\n            One of ``primary``, ``primaryPreferred``, ``secondary``,\n            ``secondaryPreferred``, or ``nearest``. Defaults to ``primary``.\n          - `readPreferenceTags`: Specifies a tag set as a comma-separated list\n            of colon-separated key-value pairs. For example ``dc:ny,rack:1``.\n            Defaults to ``None``.\n          - `maxStalenessSeconds`: (integer) The maximum estimated\n            length of time a replica set secondary can fall behind the primary\n            in replication before it will no longer be selected for operations.\n            Defaults to ``-1``, meaning no maximum. If maxStalenessSeconds\n            is set, it must be a positive integer greater than or equal to\n            90 seconds.\n\n          .. seealso:: :doc:`/examples/server_selection`\n\n          | **Authentication:**\n\n          - `username`: A string.\n          - `password`: A string.\n\n            Although username and password must be percent-escaped in a MongoDB\n            URI, they must not be percent-escaped when passed as parameters. In\n            this example, both the space and slash special characters are passed\n            as-is::\n\n              MongoClient(username="user name", password="pass/word")\n\n          - `authSource`: The database to authenticate on. Defaults to the\n            database specified in the URI, if provided, or to "admin".\n          - `authMechanism`: See :data:`~pymongo.auth.MECHANISMS` for options.\n            If no mechanism is specified, PyMongo automatically SCRAM-SHA-1\n            when connected to MongoDB 3.6 and negotiates the mechanism to use\n            (SCRAM-SHA-1 or SCRAM-SHA-256) when connected to MongoDB 4.0+.\n          - `authMechanismProperties`: Used to specify authentication mechanism\n            specific options. To specify the service name for GSSAPI\n            authentication pass authMechanismProperties=\'SERVICE_NAME:<service\n            name>\'.\n            To specify the session token for MONGODB-AWS authentication pass\n            ``authMechanismProperties=\'AWS_SESSION_TOKEN:<session token>\'``.\n\n          .. seealso:: :doc:`/examples/authentication`\n\n          | **TLS/SSL configuration:**\n\n          - `tls`: (boolean) If ``True``, create the connection to the server\n            using transport layer security. Defaults to ``False``.\n          - `tlsInsecure`: (boolean) Specify whether TLS constraints should be\n            relaxed as much as possible. Setting ``tlsInsecure=True`` implies\n            ``tlsAllowInvalidCertificates=True`` and\n            ``tlsAllowInvalidHostnames=True``. Defaults to ``False``. Think\n            very carefully before setting this to ``True`` as it dramatically\n            reduces the security of TLS.\n          - `tlsAllowInvalidCertificates`: (boolean) If ``True``, continues\n            the TLS handshake regardless of the outcome of the certificate\n            verification process. If this is ``False``, and a value is not\n            provided for ``tlsCAFile``, PyMongo will attempt to load system\n            provided CA certificates. If the python version in use does not\n            support loading system CA certificates then the ``tlsCAFile``\n            parameter must point to a file of CA certificates.\n            ``tlsAllowInvalidCertificates=False`` implies ``tls=True``.\n            Defaults to ``False``. Think very carefully before setting this\n            to ``True`` as that could make your application vulnerable to\n            on-path attackers.\n          - `tlsAllowInvalidHostnames`: (boolean) If ``True``, disables TLS\n            hostname verification. ``tlsAllowInvalidHostnames=False`` implies\n            ``tls=True``. Defaults to ``False``. Think very carefully before\n            setting this to ``True`` as that could make your application\n            vulnerable to on-path attackers.\n          - `tlsCAFile`: A file containing a single or a bundle of\n            "certification authority" certificates, which are used to validate\n            certificates passed from the other end of the connection.\n            Implies ``tls=True``. Defaults to ``None``.\n          - `tlsCertificateKeyFile`: A file containing the client certificate\n            and private key. Implies ``tls=True``. Defaults to ``None``.\n          - `tlsCRLFile`: A file containing a PEM or DER formatted\n            certificate revocation list. Implies ``tls=True``. Defaults to\n            ``None``.\n          - `tlsCertificateKeyFilePassword`: The password or passphrase for\n            decrypting the private key in ``tlsCertificateKeyFile``. Only\n            necessary if the private key is encrypted. Defaults to ``None``.\n          - `tlsDisableOCSPEndpointCheck`: (boolean) If ``True``, disables\n            certificate revocation status checking via the OCSP responder\n            specified on the server certificate.\n            ``tlsDisableOCSPEndpointCheck=False`` implies ``tls=True``.\n            Defaults to ``False``.\n          - `ssl`: (boolean) Alias for ``tls``.\n\n          | **Read Concern options:**\n          | (If not set explicitly, this will use the server default)\n\n          - `readConcernLevel`: (string) The read concern level specifies the\n            level of isolation for read operations.  For example, a read\n            operation using a read concern level of ``majority`` will only\n            return data that has been written to a majority of nodes. If the\n            level is left unspecified, the server default will be used.\n\n          | **Client side encryption options:**\n          | (If not set explicitly, client side encryption will not be enabled.)\n\n          - `auto_encryption_opts`: A\n            :class:`~pymongo.encryption_options.AutoEncryptionOpts` which\n            configures this client to automatically encrypt collection commands\n            and automatically decrypt results. See\n            :ref:`automatic-client-side-encryption` for an example.\n            If a :class:`MongoClient` is configured with\n            ``auto_encryption_opts`` and a non-None ``maxPoolSize``, a\n            separate internal ``MongoClient`` is created if any of the\n            following are true:\n\n              - A ``key_vault_client`` is not passed to\n                :class:`~pymongo.encryption_options.AutoEncryptionOpts`\n              - ``bypass_auto_encrpytion=False`` is passed to\n                :class:`~pymongo.encryption_options.AutoEncryptionOpts`\n\n          | **Stable API options:**\n          | (If not set explicitly, Stable API will not be enabled.)\n\n          - `server_api`: A\n            :class:`~pymongo.server_api.ServerApi` which configures this\n            client to use Stable API. See :ref:`versioned-api-ref` for\n            details.\n\n        .. seealso:: The MongoDB documentation on `connections <https://dochub.mongodb.org/core/connections>`_.\n\n        .. versionchanged:: 4.5\n           Added the ``serverMonitoringMode`` keyword argument.\n\n        .. versionchanged:: 4.2\n           Added the ``timeoutMS`` keyword argument.\n\n        .. versionchanged:: 4.0\n\n             - Removed the fsync, unlock, is_locked, database_names, and\n               close_cursor methods.\n               See the :ref:`pymongo4-migration-guide`.\n             - Removed the ``waitQueueMultiple`` and ``socketKeepAlive``\n               keyword arguments.\n             - The default for `uuidRepresentation` was changed from\n               ``pythonLegacy`` to ``unspecified``.\n             - Added the ``srvServiceName``, ``maxConnecting``, and ``srvMaxHosts`` URI and\n               keyword arguments.\n\n        .. versionchanged:: 3.12\n           Added the ``server_api`` keyword argument.\n           The following keyword arguments were deprecated:\n\n             - ``ssl_certfile`` and ``ssl_keyfile`` were deprecated in favor\n               of ``tlsCertificateKeyFile``.\n\n        .. versionchanged:: 3.11\n           Added the following keyword arguments and URI options:\n\n             - ``tlsDisableOCSPEndpointCheck``\n             - ``directConnection``\n\n        .. versionchanged:: 3.9\n           Added the ``retryReads`` keyword argument and URI option.\n           Added the ``tlsInsecure`` keyword argument and URI option.\n           The following keyword arguments and URI options were deprecated:\n\n             - ``wTimeout`` was deprecated in favor of ``wTimeoutMS``.\n             - ``j`` was deprecated in favor of ``journal``.\n             - ``ssl_cert_reqs`` was deprecated in favor of\n               ``tlsAllowInvalidCertificates``.\n             - ``ssl_match_hostname`` was deprecated in favor of\n               ``tlsAllowInvalidHostnames``.\n             - ``ssl_ca_certs`` was deprecated in favor of ``tlsCAFile``.\n             - ``ssl_certfile`` was deprecated in favor of\n               ``tlsCertificateKeyFile``.\n             - ``ssl_crlfile`` was deprecated in favor of ``tlsCRLFile``.\n             - ``ssl_pem_passphrase`` was deprecated in favor of\n               ``tlsCertificateKeyFilePassword``.\n\n        .. versionchanged:: 3.9\n           ``retryWrites`` now defaults to ``True``.\n\n        .. versionchanged:: 3.8\n           Added the ``server_selector`` keyword argument.\n           Added the ``type_registry`` keyword argument.\n\n        .. versionchanged:: 3.7\n           Added the ``driver`` keyword argument.\n\n        .. versionchanged:: 3.6\n           Added support for mongodb+srv:// URIs.\n           Added the ``retryWrites`` keyword argument and URI option.\n\n        .. versionchanged:: 3.5\n           Add ``username`` and ``password`` options. Document the\n           ``authSource``, ``authMechanism``, and ``authMechanismProperties``\n           options.\n           Deprecated the ``socketKeepAlive`` keyword argument and URI option.\n           ``socketKeepAlive`` now defaults to ``True``.\n\n        .. versionchanged:: 3.0\n           :class:`~pymongo.mongo_client.MongoClient` is now the one and only\n           client class for a standalone server, mongos, or replica set.\n           It includes the functionality that had been split into\n           :class:`~pymongo.mongo_client.MongoReplicaSetClient`: it can connect\n           to a replica set, discover all its members, and monitor the set for\n           stepdowns, elections, and reconfigs.\n\n           The :class:`~pymongo.mongo_client.MongoClient` constructor no\n           longer blocks while connecting to the server or servers, and it no\n           longer raises :class:`~pymongo.errors.ConnectionFailure` if they\n           are unavailable, nor :class:`~pymongo.errors.ConfigurationError`\n           if the user\'s credentials are wrong. Instead, the constructor\n           returns immediately and launches the connection process on\n           background threads.\n\n           Therefore the ``alive`` method is removed since it no longer\n           provides meaningful information; even if the client is disconnected,\n           it may discover a server in time to fulfill the next operation.\n\n           In PyMongo 2.x, :class:`~pymongo.MongoClient` accepted a list of\n           standalone MongoDB servers and used the first it could connect to::\n\n               MongoClient([\'host1.com:27017\', \'host2.com:27017\'])\n\n           A list of multiple standalones is no longer supported; if multiple\n           servers are listed they must be members of the same replica set, or\n           mongoses in the same sharded cluster.\n\n           The behavior for a list of mongoses is changed from "high\n           availability" to "load balancing". Before, the client connected to\n           the lowest-latency mongos in the list, and used it until a network\n           error prompted it to re-evaluate all mongoses\' latencies and\n           reconnect to one of them. In PyMongo 3, the client monitors its\n           network latency to all the mongoses continuously, and distributes\n           operations evenly among those with the lowest latency. See\n           :ref:`mongos-load-balancing` for more information.\n\n           The ``connect`` option is added.\n\n           The ``start_request``, ``in_request``, and ``end_request`` methods\n           are removed, as well as the ``auto_start_request`` option.\n\n           The ``copy_database`` method is removed, see the\n           :doc:`copy_database examples </examples/copydb>` for alternatives.\n\n           The :meth:`MongoClient.disconnect` method is removed; it was a\n           synonym for :meth:`~pymongo.MongoClient.close`.\n\n           :class:`~pymongo.mongo_client.MongoClient` no longer returns an\n           instance of :class:`~pymongo.database.Database` for attribute names\n           with leading underscores. You must use dict-style lookups instead::\n\n               client[\'__my_database__\']\n\n           Not::\n\n               client.__my_database__\n        '
        doc_class = document_class or dict
        self.__init_kwargs: dict[str, Any] = {'host': host, 'port': port, 'document_class': doc_class, 'tz_aware': tz_aware, 'connect': connect, 'type_registry': type_registry, **kwargs}
        if host is None:
            host = self.HOST
        if isinstance(host, str):
            host = [host]
        if port is None:
            port = self.PORT
        if not isinstance(port, int):
            raise TypeError('port must be an instance of int')
        pool_class = kwargs.pop('_pool_class', None)
        monitor_class = kwargs.pop('_monitor_class', None)
        condition_class = kwargs.pop('_condition_class', None)
        keyword_opts = common._CaseInsensitiveDictionary(kwargs)
        keyword_opts['document_class'] = doc_class
        seeds = set()
        username = None
        password = None
        dbase = None
        opts = common._CaseInsensitiveDictionary()
        fqdn = None
        srv_service_name = keyword_opts.get('srvservicename')
        srv_max_hosts = keyword_opts.get('srvmaxhosts')
        if len([h for h in host if '/' in h]) > 1:
            raise ConfigurationError('host must not contain multiple MongoDB URIs')
        for entity in host:
            if '/' in entity:
                timeout = keyword_opts.get('connecttimeoutms')
                if timeout is not None:
                    timeout = common.validate_timeout_or_none_or_zero(keyword_opts.cased_key('connecttimeoutms'), timeout)
                res = uri_parser.parse_uri(entity, port, validate=True, warn=True, normalize=False, connect_timeout=timeout, srv_service_name=srv_service_name, srv_max_hosts=srv_max_hosts)
                seeds.update(res['nodelist'])
                username = res['username'] or username
                password = res['password'] or password
                dbase = res['database'] or dbase
                opts = res['options']
                fqdn = res['fqdn']
            else:
                seeds.update(uri_parser.split_hosts(entity, port))
        if not seeds:
            raise ConfigurationError('need to specify at least one host')
        if type_registry is not None:
            keyword_opts['type_registry'] = type_registry
        if tz_aware is None:
            tz_aware = opts.get('tz_aware', False)
        if connect is None:
            connect = opts.get('connect', True)
        keyword_opts['tz_aware'] = tz_aware
        keyword_opts['connect'] = connect
        keyword_opts = _handle_option_deprecations(keyword_opts)
        keyword_opts = common._CaseInsensitiveDictionary(dict((common.validate(keyword_opts.cased_key(k), v) for (k, v) in keyword_opts.items())))
        opts.update(keyword_opts)
        if srv_service_name is None:
            srv_service_name = opts.get('srvServiceName', common.SRV_SERVICE_NAME)
        srv_max_hosts = srv_max_hosts or opts.get('srvmaxhosts')
        opts = _handle_security_options(opts)
        opts = _normalize_options(opts)
        _check_options(seeds, opts)
        username = opts.get('username', username)
        password = opts.get('password', password)
        self.__options = options = ClientOptions(username, password, dbase, opts)
        self.__default_database_name = dbase
        self.__lock = _create_lock()
        self.__kill_cursors_queue: list = []
        self._event_listeners = options.pool_options._event_listeners
        super().__init__(options.codec_options, options.read_preference, options.write_concern, options.read_concern)
        self._topology_settings = TopologySettings(seeds=seeds, replica_set_name=options.replica_set_name, pool_class=pool_class, pool_options=options.pool_options, monitor_class=monitor_class, condition_class=condition_class, local_threshold_ms=options.local_threshold_ms, server_selection_timeout=options.server_selection_timeout, server_selector=options.server_selector, heartbeat_frequency=options.heartbeat_frequency, fqdn=fqdn, direct_connection=options.direct_connection, load_balanced=options.load_balanced, srv_service_name=srv_service_name, srv_max_hosts=srv_max_hosts, server_monitoring_mode=options.server_monitoring_mode)
        self._init_background()
        if connect:
            self._get_topology()
        self._encrypter = None
        if self.__options.auto_encryption_opts:
            from pymongo.encryption import _Encrypter
            self._encrypter = _Encrypter(self, self.__options.auto_encryption_opts)
        self._timeout = self.__options.timeout
        if _HAS_REGISTER_AT_FORK:
            MongoClient._clients[self._topology._topology_id] = self

    def _init_background(self) -> None:
        if False:
            return 10
        self._topology = Topology(self._topology_settings)

        def target() -> bool:
            if False:
                for i in range(10):
                    print('nop')
            client = self_ref()
            if client is None:
                return False
            MongoClient._process_periodic_tasks(client)
            return True
        executor = periodic_executor.PeriodicExecutor(interval=common.KILL_CURSOR_FREQUENCY, min_interval=common.MIN_HEARTBEAT_INTERVAL, target=target, name='pymongo_kill_cursors_thread')
        self_ref: Any = weakref.ref(self, executor.close)
        self._kill_cursors_executor = executor

    def _after_fork(self) -> None:
        if False:
            i = 10
            return i + 15
        'Resets topology in a child after successfully forking.'
        self._init_background()

    def _duplicate(self, **kwargs: Any) -> MongoClient:
        if False:
            i = 10
            return i + 15
        args = self.__init_kwargs.copy()
        args.update(kwargs)
        return MongoClient(**args)

    def _server_property(self, attr_name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        "An attribute of the current server's description.\n\n        If the client is not connected, this will block until a connection is\n        established or raise ServerSelectionTimeoutError if no server is\n        available.\n\n        Not threadsafe if used multiple times in a single method, since\n        the server may change. In such cases, store a local reference to a\n        ServerDescription first, then use its properties.\n        "
        server = self._topology.select_server(writable_server_selector)
        return getattr(server.description, attr_name)

    def watch(self, pipeline: Optional[_Pipeline]=None, full_document: Optional[str]=None, resume_after: Optional[Mapping[str, Any]]=None, max_await_time_ms: Optional[int]=None, batch_size: Optional[int]=None, collation: Optional[_CollationIn]=None, start_at_operation_time: Optional[Timestamp]=None, session: Optional[client_session.ClientSession]=None, start_after: Optional[Mapping[str, Any]]=None, comment: Optional[Any]=None, full_document_before_change: Optional[str]=None, show_expanded_events: Optional[bool]=None) -> ChangeStream[_DocumentType]:
        if False:
            for i in range(10):
                print('nop')
        'Watch changes on this cluster.\n\n        Performs an aggregation with an implicit initial ``$changeStream``\n        stage and returns a\n        :class:`~pymongo.change_stream.ClusterChangeStream` cursor which\n        iterates over changes on all databases on this cluster.\n\n        Introduced in MongoDB 4.0.\n\n        .. code-block:: python\n\n           with client.watch() as stream:\n               for change in stream:\n                   print(change)\n\n        The :class:`~pymongo.change_stream.ClusterChangeStream` iterable\n        blocks until the next change document is returned or an error is\n        raised. If the\n        :meth:`~pymongo.change_stream.ClusterChangeStream.next` method\n        encounters a network error when retrieving a batch from the server,\n        it will automatically attempt to recreate the cursor such that no\n        change events are missed. Any error encountered during the resume\n        attempt indicates there may be an outage and will be raised.\n\n        .. code-block:: python\n\n            try:\n                with client.watch([{"$match": {"operationType": "insert"}}]) as stream:\n                    for insert_change in stream:\n                        print(insert_change)\n            except pymongo.errors.PyMongoError:\n                # The ChangeStream encountered an unrecoverable error or the\n                # resume attempt failed to recreate the cursor.\n                logging.error("...")\n\n        For a precise description of the resume process see the\n        `change streams specification`_.\n\n        :Parameters:\n          - `pipeline` (optional): A list of aggregation pipeline stages to\n            append to an initial ``$changeStream`` stage. Not all\n            pipeline stages are valid after a ``$changeStream`` stage, see the\n            MongoDB documentation on change streams for the supported stages.\n          - `full_document` (optional): The fullDocument to pass as an option\n            to the ``$changeStream`` stage. Allowed values: \'updateLookup\',\n            \'whenAvailable\', \'required\'. When set to \'updateLookup\', the\n            change notification for partial updates will include both a delta\n            describing the changes to the document, as well as a copy of the\n            entire document that was changed from some time after the change\n            occurred.\n          - `full_document_before_change`: Allowed values: \'whenAvailable\'\n            and \'required\'. Change events may now result in a\n            \'fullDocumentBeforeChange\' response field.\n          - `resume_after` (optional): A resume token. If provided, the\n            change stream will start returning changes that occur directly\n            after the operation specified in the resume token. A resume token\n            is the _id value of a change document.\n          - `max_await_time_ms` (optional): The maximum time in milliseconds\n            for the server to wait for changes before responding to a getMore\n            operation.\n          - `batch_size` (optional): The maximum number of documents to return\n            per batch.\n          - `collation` (optional): The :class:`~pymongo.collation.Collation`\n            to use for the aggregation.\n          - `start_at_operation_time` (optional): If provided, the resulting\n            change stream will only return changes that occurred at or after\n            the specified :class:`~bson.timestamp.Timestamp`. Requires\n            MongoDB >= 4.0.\n          - `session` (optional): a\n            :class:`~pymongo.client_session.ClientSession`.\n          - `start_after` (optional): The same as `resume_after` except that\n            `start_after` can resume notifications after an invalidate event.\n            This option and `resume_after` are mutually exclusive.\n          - `comment` (optional): A user-provided comment to attach to this\n            command.\n          - `show_expanded_events` (optional): Include expanded events such as DDL events like `dropIndexes`.\n\n        :Returns:\n          A :class:`~pymongo.change_stream.ClusterChangeStream` cursor.\n\n        .. versionchanged:: 4.3\n           Added `show_expanded_events` parameter.\n\n        .. versionchanged:: 4.2\n            Added ``full_document_before_change`` parameter.\n\n        .. versionchanged:: 4.1\n           Added ``comment`` parameter.\n\n        .. versionchanged:: 3.9\n           Added the ``start_after`` parameter.\n\n        .. versionadded:: 3.7\n\n        .. seealso:: The MongoDB documentation on `changeStreams <https://mongodb.com/docs/manual/changeStreams/>`_.\n\n        .. _change streams specification:\n            https://github.com/mongodb/specifications/blob/master/source/change-streams/change-streams.rst\n        '
        return ClusterChangeStream(self.admin, pipeline, full_document, resume_after, max_await_time_ms, batch_size, collation, start_at_operation_time, session, start_after, comment, full_document_before_change, show_expanded_events=show_expanded_events)

    @property
    def topology_description(self) -> TopologyDescription:
        if False:
            while True:
                i = 10
        "The description of the connected MongoDB deployment.\n\n        >>> client.topology_description\n        <TopologyDescription id: 605a7b04e76489833a7c6113, topology_type: ReplicaSetWithPrimary, servers: [<ServerDescription ('localhost', 27017) server_type: RSPrimary, rtt: 0.0007973677999995488>, <ServerDescription ('localhost', 27018) server_type: RSSecondary, rtt: 0.0005540556000003249>, <ServerDescription ('localhost', 27019) server_type: RSSecondary, rtt: 0.0010367483999999649>]>\n        >>> client.topology_description.topology_type_name\n        'ReplicaSetWithPrimary'\n\n        Note that the description is periodically updated in the background\n        but the returned object itself is immutable. Access this property again\n        to get a more recent\n        :class:`~pymongo.topology_description.TopologyDescription`.\n\n        :Returns:\n          An instance of\n          :class:`~pymongo.topology_description.TopologyDescription`.\n\n        .. versionadded:: 4.0\n        "
        return self._topology.description

    @property
    def address(self) -> Optional[tuple[str, int]]:
        if False:
            print('Hello World!')
        '(host, port) of the current standalone, primary, or mongos, or None.\n\n        Accessing :attr:`address` raises :exc:`~.errors.InvalidOperation` if\n        the client is load-balancing among mongoses, since there is no single\n        address. Use :attr:`nodes` instead.\n\n        If the client is not connected, this will block until a connection is\n        established or raise ServerSelectionTimeoutError if no server is\n        available.\n\n        .. versionadded:: 3.0\n        '
        topology_type = self._topology._description.topology_type
        if topology_type == TOPOLOGY_TYPE.Sharded and len(self.topology_description.server_descriptions()) > 1:
            raise InvalidOperation('Cannot use "address" property when load balancing among mongoses, use "nodes" instead.')
        if topology_type not in (TOPOLOGY_TYPE.ReplicaSetWithPrimary, TOPOLOGY_TYPE.Single, TOPOLOGY_TYPE.LoadBalanced, TOPOLOGY_TYPE.Sharded):
            return None
        return self._server_property('address')

    @property
    def primary(self) -> Optional[tuple[str, int]]:
        if False:
            while True:
                i = 10
        'The (host, port) of the current primary of the replica set.\n\n        Returns ``None`` if this client is not connected to a replica set,\n        there is no primary, or this client was created without the\n        `replicaSet` option.\n\n        .. versionadded:: 3.0\n           MongoClient gained this property in version 3.0.\n        '
        return self._topology.get_primary()

    @property
    def secondaries(self) -> set[_Address]:
        if False:
            return 10
        'The secondary members known to this client.\n\n        A sequence of (host, port) pairs. Empty if this client is not\n        connected to a replica set, there are no visible secondaries, or this\n        client was created without the `replicaSet` option.\n\n        .. versionadded:: 3.0\n           MongoClient gained this property in version 3.0.\n        '
        return self._topology.get_secondaries()

    @property
    def arbiters(self) -> set[_Address]:
        if False:
            while True:
                i = 10
        'Arbiters in the replica set.\n\n        A sequence of (host, port) pairs. Empty if this client is not\n        connected to a replica set, there are no arbiters, or this client was\n        created without the `replicaSet` option.\n        '
        return self._topology.get_arbiters()

    @property
    def is_primary(self) -> bool:
        if False:
            while True:
                i = 10
        'If this client is connected to a server that can accept writes.\n\n        True if the current server is a standalone, mongos, or the primary of\n        a replica set. If the client is not connected, this will block until a\n        connection is established or raise ServerSelectionTimeoutError if no\n        server is available.\n        '
        return self._server_property('is_writable')

    @property
    def is_mongos(self) -> bool:
        if False:
            i = 10
            return i + 15
        'If this client is connected to mongos. If the client is not\n        connected, this will block until a connection is established or raise\n        ServerSelectionTimeoutError if no server is available.\n        '
        return self._server_property('server_type') == SERVER_TYPE.Mongos

    @property
    def nodes(self) -> FrozenSet[_Address]:
        if False:
            for i in range(10):
                print('nop')
        "Set of all currently connected servers.\n\n        .. warning:: When connected to a replica set the value of :attr:`nodes`\n          can change over time as :class:`MongoClient`'s view of the replica\n          set changes. :attr:`nodes` can also be an empty set when\n          :class:`MongoClient` is first instantiated and hasn't yet connected\n          to any servers, or a network partition causes it to lose connection\n          to all servers.\n        "
        description = self._topology.description
        return frozenset((s.address for s in description.known_servers))

    @property
    def options(self) -> ClientOptions:
        if False:
            i = 10
            return i + 15
        'The configuration options for this client.\n\n        :Returns:\n          An instance of :class:`~pymongo.client_options.ClientOptions`.\n\n        .. versionadded:: 4.0\n        '
        return self.__options

    def _end_sessions(self, session_ids: list[_ServerSession]) -> None:
        if False:
            i = 10
            return i + 15
        'Send endSessions command(s) with the given session ids.'
        try:
            with self._conn_for_reads(ReadPreference.PRIMARY_PREFERRED, None) as (conn, read_pref):
                if not conn.supports_sessions:
                    return
                for i in range(0, len(session_ids), common._MAX_END_SESSIONS):
                    spec = SON([('endSessions', session_ids[i:i + common._MAX_END_SESSIONS])])
                    conn.command('admin', spec, read_preference=read_pref, client=self)
        except PyMongoError:
            pass

    def close(self) -> None:
        if False:
            return 10
        'Cleanup client resources and disconnect from MongoDB.\n\n        End all server sessions created by this client by sending one or more\n        endSessions commands.\n\n        Close all sockets in the connection pools and stop the monitor threads.\n\n        .. versionchanged:: 4.0\n           Once closed, the client cannot be used again and any attempt will\n           raise :exc:`~pymongo.errors.InvalidOperation`.\n\n        .. versionchanged:: 3.6\n           End all server sessions created by this client.\n        '
        session_ids = self._topology.pop_all_sessions()
        if session_ids:
            self._end_sessions(session_ids)
        self._kill_cursors_executor.close()
        self._process_kill_cursors()
        self._topology.close()
        if self._encrypter:
            self._encrypter.close()

    def _get_topology(self) -> Topology:
        if False:
            while True:
                i = 10
        'Get the internal :class:`~pymongo.topology.Topology` object.\n\n        If this client was created with "connect=False", calling _get_topology\n        launches the connection process in the background.\n        '
        self._topology.open()
        with self.__lock:
            self._kill_cursors_executor.open()
        return self._topology

    @contextlib.contextmanager
    def _checkout(self, server: Server, session: Optional[ClientSession]) -> Iterator[Connection]:
        if False:
            while True:
                i = 10
        in_txn = session and session.in_transaction
        with _MongoClientErrorHandler(self, server, session) as err_handler:
            if in_txn and session and session._pinned_connection:
                err_handler.contribute_socket(session._pinned_connection)
                yield session._pinned_connection
                return
            with server.checkout(handler=err_handler) as conn:
                if in_txn and session and (server.description.server_type in (SERVER_TYPE.Mongos, SERVER_TYPE.LoadBalancer)):
                    session._pin(server, conn)
                err_handler.contribute_socket(conn)
                if self._encrypter and (not self._encrypter._bypass_auto_encryption) and (conn.max_wire_version < 8):
                    raise ConfigurationError('Auto-encryption requires a minimum MongoDB version of 4.2')
                yield conn

    def _select_server(self, server_selector: Callable[[Selection], Selection], session: Optional[ClientSession], address: Optional[_Address]=None, deprioritized_servers: Optional[list[Server]]=None) -> Server:
        if False:
            for i in range(10):
                print('nop')
        'Select a server to run an operation on this client.\n\n        :Parameters:\n          - `server_selector`: The server selector to use if the session is\n            not pinned and no address is given.\n          - `session`: The ClientSession for the next operation, or None. May\n            be pinned to a mongos server address.\n          - `address` (optional): Address when sending a message\n            to a specific server, used for getMore.\n        '
        try:
            topology = self._get_topology()
            if session and (not session.in_transaction):
                session._transaction.reset()
            if not address and session:
                address = session._pinned_address
            if address:
                server = topology.select_server_by_address(address)
                if not server:
                    raise AutoReconnect('server %s:%s no longer available' % address)
            else:
                server = topology.select_server(server_selector, deprioritized_servers=deprioritized_servers)
            return server
        except PyMongoError as exc:
            if session and session.in_transaction:
                exc._add_error_label('TransientTransactionError')
                session._unpin()
            raise

    def _conn_for_writes(self, session: Optional[ClientSession]) -> ContextManager[Connection]:
        if False:
            print('Hello World!')
        server = self._select_server(writable_server_selector, session)
        return self._checkout(server, session)

    @contextlib.contextmanager
    def _conn_from_server(self, read_preference: _ServerMode, server: Server, session: Optional[ClientSession]) -> Iterator[tuple[Connection, _ServerMode]]:
        if False:
            for i in range(10):
                print('nop')
        assert read_preference is not None, 'read_preference must not be None'
        topology = self._get_topology()
        single = topology.description.topology_type == TOPOLOGY_TYPE.Single
        with self._checkout(server, session) as conn:
            if single:
                if conn.is_repl and (not (session and session.in_transaction)):
                    read_preference = ReadPreference.PRIMARY_PREFERRED
                elif conn.is_standalone:
                    read_preference = ReadPreference.PRIMARY
            yield (conn, read_preference)

    def _conn_for_reads(self, read_preference: _ServerMode, session: Optional[ClientSession]) -> ContextManager[tuple[Connection, _ServerMode]]:
        if False:
            while True:
                i = 10
        assert read_preference is not None, 'read_preference must not be None'
        _ = self._get_topology()
        server = self._select_server(read_preference, session)
        return self._conn_from_server(read_preference, server, session)

    def _should_pin_cursor(self, session: Optional[ClientSession]) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self.__options.load_balanced and (not (session and session.in_transaction))

    @_csot.apply
    def _run_operation(self, operation: Union[_Query, _GetMore], unpack_res: Callable, address: Optional[_Address]=None) -> Response:
        if False:
            for i in range(10):
                print('nop')
        'Run a _Query/_GetMore operation and return a Response.\n\n        :Parameters:\n          - `operation`: a _Query or _GetMore object.\n          - `unpack_res`: A callable that decodes the wire protocol response.\n          - `address` (optional): Optional address when sending a message\n            to a specific server, used for getMore.\n        '
        if operation.conn_mgr:
            server = self._select_server(operation.read_preference, operation.session, address=address)
            with operation.conn_mgr.lock:
                with _MongoClientErrorHandler(self, server, operation.session) as err_handler:
                    err_handler.contribute_socket(operation.conn_mgr.conn)
                    return server.run_operation(operation.conn_mgr.conn, operation, operation.read_preference, self._event_listeners, unpack_res)

        def _cmd(_session: Optional[ClientSession], server: Server, conn: Connection, read_preference: _ServerMode) -> Response:
            if False:
                while True:
                    i = 10
            operation.reset()
            return server.run_operation(conn, operation, read_preference, self._event_listeners, unpack_res)
        return self._retryable_read(_cmd, operation.read_preference, operation.session, address=address, retryable=isinstance(operation, message._Query))

    def _retry_with_session(self, retryable: bool, func: _WriteCall[T], session: Optional[ClientSession], bulk: Optional[_Bulk]) -> T:
        if False:
            while True:
                i = 10
        "Execute an operation with at most one consecutive retries\n\n        Returns func()'s return value on success. On error retries the same\n        command.\n\n        Re-raises any exception thrown by func().\n        "
        retryable = bool(retryable and self.options.retry_writes and session and (not session.in_transaction))
        return self._retry_internal(func=func, session=session, bulk=bulk, retryable=retryable)

    @_csot.apply
    def _retry_internal(self, func: _WriteCall[T] | _ReadCall[T], session: Optional[ClientSession], bulk: Optional[_Bulk], is_read: bool=False, address: Optional[_Address]=None, read_pref: Optional[_ServerMode]=None, retryable: bool=False) -> T:
        if False:
            i = 10
            return i + 15
        'Internal retryable helper for all client transactions.\n\n        :Parameters:\n          - `func`: Callback function we want to retry\n          - `session`: Client Session on which the transaction should occur\n          - `bulk`: Abstraction to handle bulk write operations\n          - `is_read`: If this is an exclusive read transaction, defaults to False\n          - `address`: Server Address, defaults to None\n          - `read_pref`: Topology of read operation, defaults to None\n          - `retryable`: If the operation should be retried once, defaults to None\n\n        :Returns:\n          Output of the calling func()\n        '
        return _ClientConnectionRetryable(mongo_client=self, func=func, bulk=bulk, is_read=is_read, session=session, read_pref=read_pref, address=address, retryable=retryable).run()

    def _retryable_read(self, func: _ReadCall[T], read_pref: _ServerMode, session: Optional[ClientSession], address: Optional[_Address]=None, retryable: bool=True) -> T:
        if False:
            print('Hello World!')
        "Execute an operation with consecutive retries if possible\n\n        Returns func()'s return value on success. On error retries the same\n        command.\n\n        Re-raises any exception thrown by func().\n\n          - `func`: Read call we want to execute\n          - `read_pref`: Desired topology of read operation\n          - `session`: Client session we should use to execute operation\n          - `address`: Optional address when sending a message, defaults to None\n          - `retryable`: if we should attempt retries\n            (may not always be supported even if supplied), defaults to False\n        "
        retryable = bool(retryable and self.options.retry_reads and (not (session and session.in_transaction)))
        return self._retry_internal(func, session, None, is_read=True, address=address, read_pref=read_pref, retryable=retryable)

    def _retryable_write(self, retryable: bool, func: _WriteCall[T], session: Optional[ClientSession], bulk: Optional[_Bulk]=None) -> T:
        if False:
            i = 10
            return i + 15
        "Execute an operation with consecutive retries if possible\n\n        Returns func()'s return value on success. On error retries the same\n        command.\n\n        Re-raises any exception thrown by func().\n\n        :Parameters:\n          - `retryable`: if we should attempt retries (may not always be supported)\n          - `func`: write call we want to execute during a session\n          - `session`: Client session we will use to execute write operation\n          - `bulk`: bulk abstraction to execute operations in bulk, defaults to None\n        "
        with self._tmp_session(session) as s:
            return self._retry_with_session(retryable, func, s, bulk)

    def __eq__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if isinstance(other, self.__class__):
            return self._topology == other._topology
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        if False:
            return 10
        return not self == other

    def __hash__(self) -> int:
        if False:
            print('Hello World!')
        return hash(self._topology)

    def _repr_helper(self) -> str:
        if False:
            while True:
                i = 10

        def option_repr(option: str, value: Any) -> str:
            if False:
                i = 10
                return i + 15
            "Fix options whose __repr__ isn't usable in a constructor."
            if option == 'document_class':
                if value is dict:
                    return 'document_class=dict'
                else:
                    return f'document_class={value.__module__}.{value.__name__}'
            if option in common.TIMEOUT_OPTIONS and value is not None:
                return f'{option}={int(value * 1000)}'
            return f'{option}={value!r}'
        options = ['host=%r' % ['%s:%d' % (host, port) if port is not None else host for (host, port) in self._topology_settings.seeds]]
        options.extend((option_repr(key, self.__options._options[key]) for key in self._constructor_args))
        options.extend((option_repr(key, self.__options._options[key]) for key in self.__options._options if key not in set(self._constructor_args) and key != 'username' and (key != 'password')))
        return ', '.join(options)

    def __repr__(self) -> str:
        if False:
            return 10
        return f'MongoClient({self._repr_helper()})'

    def __getattr__(self, name: str) -> database.Database[_DocumentType]:
        if False:
            i = 10
            return i + 15
        'Get a database by name.\n\n        Raises :class:`~pymongo.errors.InvalidName` if an invalid\n        database name is used.\n\n        :Parameters:\n          - `name`: the name of the database to get\n        '
        if name.startswith('_'):
            raise AttributeError(f'MongoClient has no attribute {name!r}. To access the {name} database, use client[{name!r}].')
        return self.__getitem__(name)

    def __getitem__(self, name: str) -> database.Database[_DocumentType]:
        if False:
            print('Hello World!')
        'Get a database by name.\n\n        Raises :class:`~pymongo.errors.InvalidName` if an invalid\n        database name is used.\n\n        :Parameters:\n          - `name`: the name of the database to get\n        '
        return database.Database(self, name)

    def _cleanup_cursor(self, locks_allowed: bool, cursor_id: int, address: Optional[_CursorAddress], conn_mgr: _ConnectionManager, session: Optional[ClientSession], explicit_session: bool) -> None:
        if False:
            return 10
        "Cleanup a cursor from cursor.close() or __del__.\n\n        This method handles cleanup for Cursors/CommandCursors including any\n        pinned connection or implicit session attached at the time the cursor\n        was closed or garbage collected.\n\n        :Parameters:\n          - `locks_allowed`: True if we are allowed to acquire locks.\n          - `cursor_id`: The cursor id which may be 0.\n          - `address`: The _CursorAddress.\n          - `conn_mgr`: The _ConnectionManager for the pinned connection or None.\n          - `session`: The cursor's session.\n          - `explicit_session`: True if the session was passed explicitly.\n        "
        if locks_allowed:
            if cursor_id:
                if conn_mgr and conn_mgr.more_to_come:
                    assert conn_mgr.conn is not None
                    conn_mgr.conn.close_conn(ConnectionClosedReason.ERROR)
                else:
                    self._close_cursor_now(cursor_id, address, session=session, conn_mgr=conn_mgr)
            if conn_mgr:
                conn_mgr.close()
        elif cursor_id or conn_mgr:
            self._close_cursor_soon(cursor_id, address, conn_mgr)
        if session and (not explicit_session):
            session._end_session(lock=locks_allowed)

    def _close_cursor_soon(self, cursor_id: int, address: Optional[_CursorAddress], conn_mgr: Optional[_ConnectionManager]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Request that a cursor and/or connection be cleaned up soon.'
        self.__kill_cursors_queue.append((address, cursor_id, conn_mgr))

    def _close_cursor_now(self, cursor_id: int, address: Optional[_CursorAddress], session: Optional[ClientSession]=None, conn_mgr: Optional[_ConnectionManager]=None) -> None:
        if False:
            print('Hello World!')
        'Send a kill cursors message with the given id.\n\n        The cursor is closed synchronously on the current thread.\n        '
        if not isinstance(cursor_id, int):
            raise TypeError('cursor_id must be an instance of int')
        try:
            if conn_mgr:
                with conn_mgr.lock:
                    assert address is not None
                    assert conn_mgr.conn is not None
                    self._kill_cursor_impl([cursor_id], address, session, conn_mgr.conn)
            else:
                self._kill_cursors([cursor_id], address, self._get_topology(), session)
        except PyMongoError:
            self._close_cursor_soon(cursor_id, address)

    def _kill_cursors(self, cursor_ids: Sequence[int], address: Optional[_CursorAddress], topology: Topology, session: Optional[ClientSession]) -> None:
        if False:
            i = 10
            return i + 15
        'Send a kill cursors message with the given ids.'
        if address:
            server = topology.select_server_by_address(tuple(address))
        else:
            server = topology.select_server(writable_server_selector)
        with self._checkout(server, session) as conn:
            assert address is not None
            self._kill_cursor_impl(cursor_ids, address, session, conn)

    def _kill_cursor_impl(self, cursor_ids: Sequence[int], address: _CursorAddress, session: Optional[ClientSession], conn: Connection) -> None:
        if False:
            for i in range(10):
                print('nop')
        namespace = address.namespace
        (db, coll) = namespace.split('.', 1)
        spec = SON([('killCursors', coll), ('cursors', cursor_ids)])
        conn.command(db, spec, session=session, client=self)

    def _process_kill_cursors(self) -> None:
        if False:
            print('Hello World!')
        'Process any pending kill cursors requests.'
        address_to_cursor_ids = defaultdict(list)
        pinned_cursors = []
        while True:
            try:
                (address, cursor_id, conn_mgr) = self.__kill_cursors_queue.pop()
            except IndexError:
                break
            if conn_mgr:
                pinned_cursors.append((address, cursor_id, conn_mgr))
            else:
                address_to_cursor_ids[address].append(cursor_id)
        for (address, cursor_id, conn_mgr) in pinned_cursors:
            try:
                self._cleanup_cursor(True, cursor_id, address, conn_mgr, None, False)
            except Exception as exc:
                if isinstance(exc, InvalidOperation) and self._topology._closed:
                    raise
                else:
                    helpers._handle_exception()
        if address_to_cursor_ids:
            topology = self._get_topology()
            for (address, cursor_ids) in address_to_cursor_ids.items():
                try:
                    self._kill_cursors(cursor_ids, address, topology, session=None)
                except Exception as exc:
                    if isinstance(exc, InvalidOperation) and self._topology._closed:
                        raise
                    else:
                        helpers._handle_exception()

    def _process_periodic_tasks(self) -> None:
        if False:
            return 10
        'Process any pending kill cursors requests and\n        maintain connection pool parameters.\n        '
        try:
            self._process_kill_cursors()
            self._topology.update_pool()
        except Exception as exc:
            if isinstance(exc, InvalidOperation) and self._topology._closed:
                return
            else:
                helpers._handle_exception()

    def __start_session(self, implicit: bool, **kwargs: Any) -> ClientSession:
        if False:
            i = 10
            return i + 15
        if implicit:
            self._topology._check_implicit_session_support()
            server_session: Union[_EmptyServerSession, _ServerSession] = _EmptyServerSession()
        else:
            server_session = self._get_server_session()
        opts = client_session.SessionOptions(**kwargs)
        return client_session.ClientSession(self, server_session, opts, implicit)

    def start_session(self, causal_consistency: Optional[bool]=None, default_transaction_options: Optional[client_session.TransactionOptions]=None, snapshot: Optional[bool]=False) -> client_session.ClientSession:
        if False:
            while True:
                i = 10
        'Start a logical session.\n\n        This method takes the same parameters as\n        :class:`~pymongo.client_session.SessionOptions`. See the\n        :mod:`~pymongo.client_session` module for details and examples.\n\n        A :class:`~pymongo.client_session.ClientSession` may only be used with\n        the MongoClient that started it. :class:`ClientSession` instances are\n        **not thread-safe or fork-safe**. They can only be used by one thread\n        or process at a time. A single :class:`ClientSession` cannot be used\n        to run multiple operations concurrently.\n\n        :Returns:\n          An instance of :class:`~pymongo.client_session.ClientSession`.\n\n        .. versionadded:: 3.6\n        '
        return self.__start_session(False, causal_consistency=causal_consistency, default_transaction_options=default_transaction_options, snapshot=snapshot)

    def _get_server_session(self) -> _ServerSession:
        if False:
            print('Hello World!')
        'Internal: start or resume a _ServerSession.'
        return self._topology.get_server_session()

    def _return_server_session(self, server_session: Union[_ServerSession, _EmptyServerSession], lock: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Internal: return a _ServerSession to the pool.'
        if isinstance(server_session, _EmptyServerSession):
            return None
        return self._topology.return_server_session(server_session, lock)

    def _ensure_session(self, session: Optional[ClientSession]=None) -> Optional[ClientSession]:
        if False:
            print('Hello World!')
        'If provided session is None, lend a temporary session.'
        if session:
            return session
        try:
            return self.__start_session(True, causal_consistency=False)
        except (ConfigurationError, InvalidOperation):
            return None

    @contextlib.contextmanager
    def _tmp_session(self, session: Optional[client_session.ClientSession], close: bool=True) -> Generator[Optional[client_session.ClientSession], None, None]:
        if False:
            return 10
        'If provided session is None, lend a temporary session.'
        if session is not None:
            if not isinstance(session, client_session.ClientSession):
                raise ValueError("'session' argument must be a ClientSession or None.")
            yield session
            return
        s = self._ensure_session(session)
        if s:
            try:
                yield s
            except Exception as exc:
                if isinstance(exc, ConnectionFailure):
                    s._server_session.mark_dirty()
                s.end_session()
                raise
            finally:
                if close:
                    s.end_session()
        else:
            yield None

    def _send_cluster_time(self, command: MutableMapping[str, Any], session: Optional[ClientSession]) -> None:
        if False:
            print('Hello World!')
        topology_time = self._topology.max_cluster_time()
        session_time = session.cluster_time if session else None
        if topology_time and session_time:
            if topology_time['clusterTime'] > session_time['clusterTime']:
                cluster_time: Optional[ClusterTime] = topology_time
            else:
                cluster_time = session_time
        else:
            cluster_time = topology_time or session_time
        if cluster_time:
            command['$clusterTime'] = cluster_time

    def _process_response(self, reply: Mapping[str, Any], session: Optional[ClientSession]) -> None:
        if False:
            while True:
                i = 10
        self._topology.receive_cluster_time(reply.get('$clusterTime'))
        if session is not None:
            session._process_response(reply)

    def server_info(self, session: Optional[client_session.ClientSession]=None) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "Get information about the MongoDB server we're connected to.\n\n        :Parameters:\n          - `session` (optional): a\n            :class:`~pymongo.client_session.ClientSession`.\n\n        .. versionchanged:: 3.6\n           Added ``session`` parameter.\n        "
        return cast(dict, self.admin.command('buildinfo', read_preference=ReadPreference.PRIMARY, session=session))

    def list_databases(self, session: Optional[client_session.ClientSession]=None, comment: Optional[Any]=None, **kwargs: Any) -> CommandCursor[dict[str, Any]]:
        if False:
            return 10
        'Get a cursor over the databases of the connected server.\n\n        :Parameters:\n          - `session` (optional): a\n            :class:`~pymongo.client_session.ClientSession`.\n          - `comment` (optional): A user-provided comment to attach to this\n            command.\n          - `**kwargs` (optional): Optional parameters of the\n            `listDatabases command\n            <https://mongodb.com/docs/manual/reference/command/listDatabases/>`_\n            can be passed as keyword arguments to this method. The supported\n            options differ by server version.\n\n\n        :Returns:\n          An instance of :class:`~pymongo.command_cursor.CommandCursor`.\n\n        .. versionadded:: 3.6\n        '
        cmd = SON([('listDatabases', 1)])
        cmd.update(kwargs)
        if comment is not None:
            cmd['comment'] = comment
        admin = self._database_default_options('admin')
        res = admin._retryable_read_command(cmd, session=session)
        cursor = {'id': 0, 'firstBatch': res['databases'], 'ns': 'admin.$cmd'}
        return CommandCursor(admin['$cmd'], cursor, None, comment=comment)

    def list_database_names(self, session: Optional[client_session.ClientSession]=None, comment: Optional[Any]=None) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        'Get a list of the names of all databases on the connected server.\n\n        :Parameters:\n          - `session` (optional): a\n            :class:`~pymongo.client_session.ClientSession`.\n          - `comment` (optional): A user-provided comment to attach to this\n            command.\n\n        .. versionchanged:: 4.1\n           Added ``comment`` parameter.\n\n        .. versionadded:: 3.6\n        '
        return [doc['name'] for doc in self.list_databases(session, nameOnly=True, comment=comment)]

    @_csot.apply
    def drop_database(self, name_or_database: Union[str, database.Database[_DocumentTypeArg]], session: Optional[client_session.ClientSession]=None, comment: Optional[Any]=None) -> None:
        if False:
            return 10
        "Drop a database.\n\n        Raises :class:`TypeError` if `name_or_database` is not an instance of\n        :class:`str` or :class:`~pymongo.database.Database`.\n\n        :Parameters:\n          - `name_or_database`: the name of a database to drop, or a\n            :class:`~pymongo.database.Database` instance representing the\n            database to drop\n          - `session` (optional): a\n            :class:`~pymongo.client_session.ClientSession`.\n          - `comment` (optional): A user-provided comment to attach to this\n            command.\n\n        .. versionchanged:: 4.1\n           Added ``comment`` parameter.\n\n        .. versionchanged:: 3.6\n           Added ``session`` parameter.\n\n        .. note:: The :attr:`~pymongo.mongo_client.MongoClient.write_concern` of\n           this client is automatically applied to this operation.\n\n        .. versionchanged:: 3.4\n           Apply this client's write concern automatically to this operation\n           when connected to MongoDB >= 3.4.\n\n        "
        name = name_or_database
        if isinstance(name, database.Database):
            name = name.name
        if not isinstance(name, str):
            raise TypeError('name_or_database must be an instance of str or a Database')
        with self._conn_for_writes(session) as conn:
            self[name]._command(conn, {'dropDatabase': 1, 'comment': comment}, read_preference=ReadPreference.PRIMARY, write_concern=self._write_concern_for(session), parse_write_concern_error=True, session=session)

    def get_default_database(self, default: Optional[str]=None, codec_options: Optional[bson.CodecOptions[_DocumentTypeArg]]=None, read_preference: Optional[_ServerMode]=None, write_concern: Optional[WriteConcern]=None, read_concern: Optional[ReadConcern]=None) -> database.Database[_DocumentType]:
        if False:
            while True:
                i = 10
        "Get the database named in the MongoDB connection URI.\n\n        >>> uri = 'mongodb://host/my_database'\n        >>> client = MongoClient(uri)\n        >>> db = client.get_default_database()\n        >>> assert db.name == 'my_database'\n        >>> db = client.get_database()\n        >>> assert db.name == 'my_database'\n\n        Useful in scripts where you want to choose which database to use\n        based only on the URI in a configuration file.\n\n        :Parameters:\n          - `default` (optional): the database name to use if no database name\n            was provided in the URI.\n          - `codec_options` (optional): An instance of\n            :class:`~bson.codec_options.CodecOptions`. If ``None`` (the\n            default) the :attr:`codec_options` of this :class:`MongoClient` is\n            used.\n          - `read_preference` (optional): The read preference to use. If\n            ``None`` (the default) the :attr:`read_preference` of this\n            :class:`MongoClient` is used. See :mod:`~pymongo.read_preferences`\n            for options.\n          - `write_concern` (optional): An instance of\n            :class:`~pymongo.write_concern.WriteConcern`. If ``None`` (the\n            default) the :attr:`write_concern` of this :class:`MongoClient` is\n            used.\n          - `read_concern` (optional): An instance of\n            :class:`~pymongo.read_concern.ReadConcern`. If ``None`` (the\n            default) the :attr:`read_concern` of this :class:`MongoClient` is\n            used.\n          - `comment` (optional): A user-provided comment to attach to this\n            command.\n\n        .. versionchanged:: 4.1\n           Added ``comment`` parameter.\n\n        .. versionchanged:: 3.8\n           Undeprecated. Added the ``default``, ``codec_options``,\n           ``read_preference``, ``write_concern`` and ``read_concern``\n           parameters.\n\n        .. versionchanged:: 3.5\n           Deprecated, use :meth:`get_database` instead.\n        "
        if self.__default_database_name is None and default is None:
            raise ConfigurationError('No default database name defined or provided.')
        name = cast(str, self.__default_database_name or default)
        return database.Database(self, name, codec_options, read_preference, write_concern, read_concern)

    def get_database(self, name: Optional[str]=None, codec_options: Optional[bson.CodecOptions[_DocumentTypeArg]]=None, read_preference: Optional[_ServerMode]=None, write_concern: Optional[WriteConcern]=None, read_concern: Optional[ReadConcern]=None) -> database.Database[_DocumentType]:
        if False:
            return 10
        "Get a :class:`~pymongo.database.Database` with the given name and\n        options.\n\n        Useful for creating a :class:`~pymongo.database.Database` with\n        different codec options, read preference, and/or write concern from\n        this :class:`MongoClient`.\n\n          >>> client.read_preference\n          Primary()\n          >>> db1 = client.test\n          >>> db1.read_preference\n          Primary()\n          >>> from pymongo import ReadPreference\n          >>> db2 = client.get_database(\n          ...     'test', read_preference=ReadPreference.SECONDARY)\n          >>> db2.read_preference\n          Secondary(tag_sets=None)\n\n        :Parameters:\n          - `name` (optional): The name of the database - a string. If ``None``\n            (the default) the database named in the MongoDB connection URI is\n            returned.\n          - `codec_options` (optional): An instance of\n            :class:`~bson.codec_options.CodecOptions`. If ``None`` (the\n            default) the :attr:`codec_options` of this :class:`MongoClient` is\n            used.\n          - `read_preference` (optional): The read preference to use. If\n            ``None`` (the default) the :attr:`read_preference` of this\n            :class:`MongoClient` is used. See :mod:`~pymongo.read_preferences`\n            for options.\n          - `write_concern` (optional): An instance of\n            :class:`~pymongo.write_concern.WriteConcern`. If ``None`` (the\n            default) the :attr:`write_concern` of this :class:`MongoClient` is\n            used.\n          - `read_concern` (optional): An instance of\n            :class:`~pymongo.read_concern.ReadConcern`. If ``None`` (the\n            default) the :attr:`read_concern` of this :class:`MongoClient` is\n            used.\n\n        .. versionchanged:: 3.5\n           The `name` parameter is now optional, defaulting to the database\n           named in the MongoDB connection URI.\n        "
        if name is None:
            if self.__default_database_name is None:
                raise ConfigurationError('No default database defined')
            name = self.__default_database_name
        return database.Database(self, name, codec_options, read_preference, write_concern, read_concern)

    def _database_default_options(self, name: str) -> Database:
        if False:
            return 10
        'Get a Database instance with the default settings.'
        return self.get_database(name, codec_options=DEFAULT_CODEC_OPTIONS, read_preference=ReadPreference.PRIMARY, write_concern=DEFAULT_WRITE_CONCERN)

    def __enter__(self) -> MongoClient[_DocumentType]:
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.close()
    __iter__ = None

    def __next__(self) -> NoReturn:
        if False:
            print('Hello World!')
        raise TypeError("'MongoClient' object is not iterable")
    next = __next__

def _retryable_error_doc(exc: PyMongoError) -> Optional[Mapping[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    'Return the server response from PyMongo exception or None.'
    if isinstance(exc, BulkWriteError):
        wces = exc.details['writeConcernErrors']
        return wces[-1] if wces else None
    if isinstance(exc, (NotPrimaryError, OperationFailure)):
        return cast(Mapping[str, Any], exc.details)
    return None

def _add_retryable_write_error(exc: PyMongoError, max_wire_version: int) -> None:
    if False:
        print('Hello World!')
    doc = _retryable_error_doc(exc)
    if doc:
        code = doc.get('code', 0)
        if code == 20 and str(exc).startswith('Transaction numbers'):
            errmsg = 'This MongoDB deployment does not support retryable writes. Please add retryWrites=false to your connection string.'
            raise OperationFailure(errmsg, code, exc.details)
        if max_wire_version >= 9:
            for label in doc.get('errorLabels', []):
                exc._add_error_label(label)
        elif code in helpers._RETRYABLE_ERROR_CODES:
            exc._add_error_label('RetryableWriteError')
    if isinstance(exc, ConnectionFailure) and (not isinstance(exc, (NotPrimaryError, WaitQueueTimeoutError))):
        exc._add_error_label('RetryableWriteError')

class _MongoClientErrorHandler:
    """Handle errors raised when executing an operation."""
    __slots__ = ('client', 'server_address', 'session', 'max_wire_version', 'sock_generation', 'completed_handshake', 'service_id', 'handled')

    def __init__(self, client: MongoClient, server: Server, session: Optional[ClientSession]):
        if False:
            return 10
        self.client = client
        self.server_address = server.description.address
        self.session = session
        self.max_wire_version = common.MIN_WIRE_VERSION
        self.sock_generation = server.pool.gen.get_overall()
        self.completed_handshake = False
        self.service_id: Optional[ObjectId] = None
        self.handled = False

    def contribute_socket(self, conn: Connection, completed_handshake: bool=True) -> None:
        if False:
            return 10
        'Provide socket information to the error handler.'
        self.max_wire_version = conn.max_wire_version
        self.sock_generation = conn.generation
        self.service_id = conn.service_id
        self.completed_handshake = completed_handshake

    def handle(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException]) -> None:
        if False:
            print('Hello World!')
        if self.handled or exc_val is None:
            return
        self.handled = True
        if self.session:
            if isinstance(exc_val, ConnectionFailure):
                if self.session.in_transaction:
                    exc_val._add_error_label('TransientTransactionError')
                self.session._server_session.mark_dirty()
            if isinstance(exc_val, PyMongoError):
                if exc_val.has_error_label('TransientTransactionError') or exc_val.has_error_label('RetryableWriteError'):
                    self.session._unpin()
        err_ctx = _ErrorContext(exc_val, self.max_wire_version, self.sock_generation, self.completed_handshake, self.service_id)
        self.client._topology.handle_error(self.server_address, err_ctx)

    def __enter__(self) -> _MongoClientErrorHandler:
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type: Optional[Type[Exception]], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> None:
        if False:
            return 10
        return self.handle(exc_type, exc_val)

class _ClientConnectionRetryable(Generic[T]):
    """Responsible for executing retryable connections on read or write operations"""

    def __init__(self, mongo_client: MongoClient, func: _WriteCall[T] | _ReadCall[T], bulk: Optional[_Bulk], is_read: bool=False, session: Optional[ClientSession]=None, read_pref: Optional[_ServerMode]=None, address: Optional[_Address]=None, retryable: bool=False):
        if False:
            print('Hello World!')
        self._last_error: Optional[Exception] = None
        self._retrying = False
        self._multiple_retries = _csot.get_timeout() is not None
        self._client = mongo_client
        self._func = func
        self._bulk = bulk
        self._session = session
        self._is_read = is_read
        self._retryable = retryable
        self._read_pref = read_pref
        self._server_selector: Callable[[Selection], Selection] = read_pref if is_read else writable_server_selector
        self._address = address
        self._server: Server = None
        self._deprioritized_servers: list[Server] = []

    def run(self) -> T:
        if False:
            for i in range(10):
                print('nop')
        'Runs the supplied func() and attempts a retry\n\n        :Raises:\n            self._last_error: Last exception raised\n\n        :Returns:\n            Result of the func() call\n        '
        if self._is_session_state_retryable() and self._retryable and (not self._is_read):
            self._session._start_retryable_write()
            if self._bulk:
                self._bulk.started_retryable_write = True
        while True:
            self._check_last_error(check_csot=True)
            try:
                return self._read() if self._is_read else self._write()
            except ServerSelectionTimeoutError:
                self._check_last_error()
                raise
            except PyMongoError as exc:
                if self._is_read:
                    if isinstance(exc, (ConnectionFailure, OperationFailure)):
                        exc_code = getattr(exc, 'code', None)
                        if self._is_not_eligible_for_retry() or (isinstance(exc, OperationFailure) and exc_code not in helpers._RETRYABLE_ERROR_CODES):
                            raise
                        self._retrying = True
                        self._last_error = exc
                    else:
                        raise
                if not self._is_read:
                    if not self._retryable:
                        raise
                    retryable_write_error_exc = exc.has_error_label('RetryableWriteError')
                    if retryable_write_error_exc:
                        assert self._session
                        self._session._unpin()
                    if not retryable_write_error_exc or self._is_not_eligible_for_retry():
                        if exc.has_error_label('NoWritesPerformed') and self._last_error:
                            raise self._last_error from exc
                        else:
                            raise
                    if self._bulk:
                        self._bulk.retrying = True
                    else:
                        self._retrying = True
                    if not exc.has_error_label('NoWritesPerformed'):
                        self._last_error = exc
                    if self._last_error is None:
                        self._last_error = exc
                if self._client.topology_description.topology_type == TOPOLOGY_TYPE.Sharded:
                    self._deprioritized_servers.append(self._server)

    def _is_not_eligible_for_retry(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Checks if the exchange is not eligible for retry'
        return not self._retryable or (self._is_retrying() and (not self._multiple_retries))

    def _is_retrying(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Checks if the exchange is currently undergoing a retry'
        return self._bulk.retrying if self._bulk else self._retrying

    def _is_session_state_retryable(self) -> bool:
        if False:
            return 10
        'Checks if provided session is eligible for retry\n\n        reads: Make sure there is no ongoing transaction (if provided a session)\n        writes: Make sure there is a session without an active transaction\n        '
        if self._is_read:
            return not (self._session and self._session.in_transaction)
        return bool(self._session and (not self._session.in_transaction))

    def _check_last_error(self, check_csot: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks if the ongoing client exchange experienced a exception previously.\n        If so, raise last error\n\n        :Parameters:\n          - `check_csot`: Checks CSOT to ensure we are retrying with time remaining defaults to False\n        '
        if self._is_retrying():
            remaining = _csot.remaining()
            if not check_csot or (remaining is not None and remaining <= 0):
                assert self._last_error is not None
                raise self._last_error

    def _get_server(self) -> Server:
        if False:
            print('Hello World!')
        'Retrieves a server object based on provided object context\n\n        :Returns:\n            Abstraction to connect to server\n        '
        return self._client._select_server(self._server_selector, self._session, address=self._address, deprioritized_servers=self._deprioritized_servers)

    def _write(self) -> T:
        if False:
            return 10
        "Wrapper method for write-type retryable client executions\n\n        :Returns:\n            Output for func()'s call\n        "
        try:
            max_wire_version = 0
            self._server = self._get_server()
            supports_session = self._session is not None and self._server.description.retryable_writes_supported
            with self._client._checkout(self._server, self._session) as conn:
                max_wire_version = conn.max_wire_version
                if self._retryable and (not supports_session):
                    self._check_last_error()
                    self._retryable = False
                return self._func(self._session, conn, self._retryable)
        except PyMongoError as exc:
            if not self._retryable:
                raise
            _add_retryable_write_error(exc, max_wire_version)
            raise

    def _read(self) -> T:
        if False:
            for i in range(10):
                print('nop')
        "Wrapper method for read-type retryable client executions\n\n        :Returns:\n            Output for func()'s call\n        "
        self._server = self._get_server()
        assert self._read_pref is not None, 'Read Preference required on read calls'
        with self._client._conn_from_server(self._read_pref, self._server, self._session) as (conn, read_pref):
            if self._retrying and (not self._retryable):
                self._check_last_error()
            return self._func(self._session, self._server, conn, read_pref)

def _after_fork_child() -> None:
    if False:
        return 10
    'Releases the locks in child process and resets the\n    topologies in all MongoClients.\n    '
    _release_locks()
    for (_, client) in MongoClient._clients.items():
        client._after_fork()
if _HAS_REGISTER_AT_FORK:
    os.register_at_fork(after_in_child=_after_fork_child)