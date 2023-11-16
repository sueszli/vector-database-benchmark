"""
Support for Elasticsearch (1.0.0 or newer).

Provides an :class:`ElasticsearchTarget` and a :class:`CopyToIndex` template task.

Modeled after :class:`luigi.contrib.rdbms.CopyToTable`.

A minimal example (assuming elasticsearch is running on localhost:9200):

.. code-block:: python

    class ExampleIndex(CopyToIndex):
        index = 'example'

        def docs(self):
            return [{'_id': 1, 'title': 'An example document.'}]

    if __name__ == '__main__':
        task = ExampleIndex()
        luigi.build([task], local_scheduler=True)

All options:

.. code-block:: python

    class ExampleIndex(CopyToIndex):
        host = 'localhost'
        port = 9200
        index = 'example'
        doc_type = 'default'
        purge_existing_index = True
        marker_index_hist_size = 1

        def docs(self):
            return [{'_id': 1, 'title': 'An example document.'}]

    if __name__ == '__main__':
        task = ExampleIndex()
        luigi.build([task], local_scheduler=True)

`Host`, `port`, `index`, `doc_type` parameters are standard elasticsearch.

`purge_existing_index` will delete the index, whenever an update is required.
This is useful, when one deals with "dumps" that represent the whole data, not just updates.

`marker_index_hist_size` sets the maximum number of entries in the 'marker'
index:

* 0 (default) keeps all updates,
* 1 to only remember the most recent update to the index.

This can be useful, if an index needs to recreated, even though
the corresponding indexing task has been run sometime in the past - but
a later indexing task might have altered the index in the meantime.

There are a two luigi `luigi.cfg` configuration options:

.. code-block:: ini

    [elasticsearch]

    marker-index = update_log
    marker-doc-type = entry

"""
import abc
import datetime
import hashlib
import json
import logging
import itertools
import luigi
logger = logging.getLogger('luigi-interface')
try:
    import elasticsearch
    if elasticsearch.__version__ < (1, 0, 0):
        logger.warning('This module works with elasticsearch 1.0.0 or newer only.')
    from elasticsearch.helpers import bulk
    from elasticsearch.connection import Urllib3HttpConnection
except ImportError:
    logger.warning('Loading esindex module without elasticsearch installed. Will crash at runtime if esindex functionality is used.')

class ElasticsearchTarget(luigi.Target):
    """ Target for a resource in Elasticsearch."""
    marker_index = luigi.configuration.get_config().get('elasticsearch', 'marker-index', 'update_log')
    marker_doc_type = luigi.configuration.get_config().get('elasticsearch', 'marker-doc-type', 'entry')

    def __init__(self, host, port, index, doc_type, update_id, marker_index_hist_size=0, http_auth=None, timeout=10, extra_elasticsearch_args=None):
        if False:
            while True:
                i = 10
        '\n        :param host: Elasticsearch server host\n        :type host: str\n        :param port: Elasticsearch server port\n        :type port: int\n        :param index: index name\n        :type index: str\n        :param doc_type: doctype name\n        :type doc_type: str\n        :param update_id: an identifier for this data set\n        :type update_id: str\n        :param marker_index_hist_size: list of changes to the index to remember\n        :type marker_index_hist_size: int\n        :param timeout: Elasticsearch connection timeout\n        :type timeout: int\n        :param extra_elasticsearch_args: extra args for Elasticsearch\n        :type Extra: dict\n        '
        if extra_elasticsearch_args is None:
            extra_elasticsearch_args = {}
        self.host = host
        self.port = port
        self.http_auth = http_auth
        self.index = index
        self.doc_type = doc_type
        self.update_id = update_id
        self.marker_index_hist_size = marker_index_hist_size
        self.timeout = timeout
        self.extra_elasticsearch_args = extra_elasticsearch_args
        self.es = elasticsearch.Elasticsearch(connection_class=Urllib3HttpConnection, host=self.host, port=self.port, http_auth=self.http_auth, timeout=self.timeout, **self.extra_elasticsearch_args)

    def marker_index_document_id(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate an id for the indicator document.\n        '
        params = '%s:%s:%s' % (self.index, self.doc_type, self.update_id)
        return hashlib.sha1(params.encode('utf-8')).hexdigest()

    def touch(self):
        if False:
            print('Hello World!')
        '\n        Mark this update as complete.\n\n        The document id would be sufficient but,\n        for documentation,\n        we index the parameters `update_id`, `target_index`, `target_doc_type` and `date` as well.\n        '
        self.create_marker_index()
        self.es.index(index=self.marker_index, doc_type=self.marker_doc_type, id=self.marker_index_document_id(), body={'update_id': self.update_id, 'target_index': self.index, 'target_doc_type': self.doc_type, 'date': datetime.datetime.now()})
        self.es.indices.flush(index=self.marker_index)
        self.ensure_hist_size()

    def exists(self):
        if False:
            print('Hello World!')
        '\n        Test, if this task has been run.\n        '
        try:
            self.es.get(index=self.marker_index, doc_type=self.marker_doc_type, id=self.marker_index_document_id())
            return True
        except elasticsearch.NotFoundError:
            logger.debug('Marker document not found.')
        except elasticsearch.ElasticsearchException as err:
            logger.warn(err)
        return False

    def create_marker_index(self):
        if False:
            print('Hello World!')
        '\n        Create the index that will keep track of the tasks if necessary.\n        '
        if not self.es.indices.exists(index=self.marker_index):
            self.es.indices.create(index=self.marker_index)

    def ensure_hist_size(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Shrink the history of updates for\n        a `index/doc_type` combination down to `self.marker_index_hist_size`.\n        '
        if self.marker_index_hist_size == 0:
            return
        result = self.es.search(index=self.marker_index, doc_type=self.marker_doc_type, body={'query': {'term': {'target_index': self.index}}}, sort=('date:desc',))
        for (i, hit) in enumerate(result.get('hits').get('hits'), start=1):
            if i > self.marker_index_hist_size:
                marker_document_id = hit.get('_id')
                self.es.delete(id=marker_document_id, index=self.marker_index, doc_type=self.marker_doc_type)
        self.es.indices.flush(index=self.marker_index)

class CopyToIndex(luigi.Task):
    """
    Template task for inserting a data set into Elasticsearch.

    Usage:

    1. Subclass and override the required `index` attribute.

    2. Implement a custom `docs` method, that returns an iterable over the documents.
       A document can be a JSON string,
       e.g. from a newline-delimited JSON (ldj) file (default implementation)
       or some dictionary.

    Optional attributes:

    * doc_type (default),
    * host (localhost),
    * port (9200),
    * settings ({'settings': {}})
    * mapping (None),
    * chunk_size (2000),
    * raise_on_error (True),
    * purge_existing_index (False),
    * marker_index_hist_size (0)

    If settings are defined, they are only applied at index creation time.
    """

    @property
    def host(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        ES hostname.\n        '
        return 'localhost'

    @property
    def port(self):
        if False:
            return 10
        '\n        ES port.\n        '
        return 9200

    @property
    def http_auth(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        ES optional http auth information as either ‘:’ separated string or a tuple,\n        e.g. `(\'user\', \'pass\')` or `"user:pass"`.\n        '
        return None

    @property
    @abc.abstractmethod
    def index(self):
        if False:
            while True:
                i = 10
        '\n        The target index.\n\n        May exist or not.\n        '
        return None

    @property
    def doc_type(self):
        if False:
            while True:
                i = 10
        '\n        The target doc_type.\n        '
        return 'default'

    @property
    def mapping(self):
        if False:
            while True:
                i = 10
        '\n        Dictionary with custom mapping or `None`.\n        '
        return None

    @property
    def settings(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Settings to be used at index creation time.\n        '
        return {'settings': {}}

    @property
    def chunk_size(self):
        if False:
            return 10
        '\n        Single API call for this number of docs.\n        '
        return 2000

    @property
    def raise_on_error(self):
        if False:
            print('Hello World!')
        '\n        Whether to fail fast.\n        '
        return True

    @property
    def purge_existing_index(self):
        if False:
            return 10
        '\n        Whether to delete the `index` completely before any indexing.\n        '
        return False

    @property
    def marker_index_hist_size(self):
        if False:
            i = 10
            return i + 15
        '\n        Number of event log entries in the marker index. 0: unlimited.\n        '
        return 0

    @property
    def timeout(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Timeout.\n        '
        return 10

    @property
    def extra_elasticsearch_args(self):
        if False:
            i = 10
            return i + 15
        '\n        Extra arguments to pass to the Elasticsearch constructor\n        '
        return {}

    def docs(self):
        if False:
            print('Hello World!')
        '\n        Return the documents to be indexed.\n\n        Beside the user defined fields, the document may contain an `_index`, `_type` and `_id`.\n        '
        with self.input().open('r') as fobj:
            for line in fobj:
                yield line

    def _docs(self):
        if False:
            while True:
                i = 10
        '\n        Since `self.docs` may yield documents that do not explicitly contain `_index` or `_type`,\n        add those attributes here, if necessary.\n        '
        iterdocs = iter(self.docs())
        first = next(iterdocs)
        needs_parsing = False
        if isinstance(first, str):
            needs_parsing = True
        elif isinstance(first, dict):
            pass
        else:
            raise RuntimeError('Document must be either JSON strings or dict.')
        for doc in itertools.chain([first], iterdocs):
            if needs_parsing:
                doc = json.loads(doc)
            if '_index' not in doc:
                doc['_index'] = self.index
            if '_type' not in doc:
                doc['_type'] = self.doc_type
            yield doc

    def _init_connection(self):
        if False:
            while True:
                i = 10
        return elasticsearch.Elasticsearch(connection_class=Urllib3HttpConnection, host=self.host, port=self.port, http_auth=self.http_auth, timeout=self.timeout, **self.extra_elasticsearch_args)

    def create_index(self):
        if False:
            i = 10
            return i + 15
        '\n        Override to provide code for creating the target index.\n\n        By default it will be created without any special settings or mappings.\n        '
        es = self._init_connection()
        if not es.indices.exists(index=self.index):
            es.indices.create(index=self.index, body=self.settings)

    def delete_index(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete the index, if it exists.\n        '
        es = self._init_connection()
        if es.indices.exists(index=self.index):
            es.indices.delete(index=self.index)

    def update_id(self):
        if False:
            print('Hello World!')
        '\n        This id will be a unique identifier for this indexing task.\n        '
        return self.task_id

    def output(self):
        if False:
            while True:
                i = 10
        "\n        Returns a ElasticsearchTarget representing the inserted dataset.\n\n        Normally you don't override this.\n        "
        return ElasticsearchTarget(host=self.host, port=self.port, http_auth=self.http_auth, index=self.index, doc_type=self.doc_type, update_id=self.update_id(), marker_index_hist_size=self.marker_index_hist_size, timeout=self.timeout, extra_elasticsearch_args=self.extra_elasticsearch_args)

    def run(self):
        if False:
            return 10
        '\n        Run task, namely:\n\n        * purge existing index, if requested (`purge_existing_index`),\n        * create the index, if missing,\n        * apply mappings, if given,\n        * set refresh interval to -1 (disable) for performance reasons,\n        * bulk index in batches of size `chunk_size` (2000),\n        * set refresh interval to 1s,\n        * refresh Elasticsearch,\n        * create entry in marker index.\n        '
        if self.purge_existing_index:
            self.delete_index()
        self.create_index()
        es = self._init_connection()
        if self.mapping:
            es.indices.put_mapping(index=self.index, doc_type=self.doc_type, body=self.mapping)
        es.indices.put_settings({'index': {'refresh_interval': '-1'}}, index=self.index)
        bulk(es, self._docs(), chunk_size=self.chunk_size, raise_on_error=self.raise_on_error)
        es.indices.put_settings({'index': {'refresh_interval': '1s'}}, index=self.index)
        es.indices.refresh()
        self.output().touch()