import datetime
import json
import luigi
from luigi.contrib.esindex import CopyToIndex

class FakeDocuments(luigi.Task):
    """
    Generates a local file containing 5 elements of data in JSON format.
    """
    date = luigi.DateParameter(default=datetime.date.today())

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Writes data in JSON format into the task's output target.\n\n        The data objects have the following attributes:\n\n        * `_id` is the default Elasticsearch id field,\n        * `text`: the text,\n        * `date`: the day when the data was created.\n\n        "
        today = datetime.date.today()
        with self.output().open('w') as output:
            for i in range(5):
                output.write(json.dumps({'_id': i, 'text': 'Hi %s' % i, 'date': str(today)}))
                output.write('\n')

    def output(self):
        if False:
            while True:
                i = 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget(path='/tmp/_docs-%s.ldj' % self.date)

class IndexDocuments(CopyToIndex):
    """
    This task loads JSON data contained in a :py:class:`luigi.target.Target` into an ElasticSearch index.

    This task's input will the target returned by :py:meth:`~.FakeDocuments.output`.

    This class uses :py:meth:`luigi.contrib.esindex.CopyToIndex.run`.

    After running this task you can run:

    .. code-block:: console

        $ curl "localhost:9200/example_index/_search?pretty"

    to see the indexed documents.

    To see the update log, run

    .. code-block:: console

        $ curl "localhost:9200/update_log/_search?q=target_index:example_index&pretty"

    To cleanup both indexes run:

    .. code-block:: console

        $ curl -XDELETE "localhost:9200/example_index"
        $ curl -XDELETE "localhost:9200/update_log/_query?q=target_index:example_index"

    """
    date = luigi.DateParameter(default=datetime.date.today())
    index = 'example_index'
    doc_type = 'greetings'
    host = 'localhost'
    port = 9200

    def requires(self):
        if False:
            print('Hello World!')
        "\n        This task's dependencies:\n\n        * :py:class:`~.FakeDocuments`\n\n        :return: object (:py:class:`luigi.task.Task`)\n        "
        return FakeDocuments()
if __name__ == '__main__':
    luigi.run(['IndexDocuments', '--local-scheduler'])