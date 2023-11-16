import random
from collections import defaultdict
from heapq import nlargest
import luigi
import luigi.contrib.hdfs
import luigi.contrib.postgres
import luigi.contrib.spark

class ExternalStreams(luigi.ExternalTask):
    """
    Example of a possible external data dump

    To depend on external targets (typically at the top of your dependency graph), you can define
    an ExternalTask like this.
    """
    date = luigi.DateParameter()

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, it expects a file to be present in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget(self.date.strftime('data/streams_%Y-%m-%d.tsv'))

class Streams(luigi.Task):
    """
    Faked version right now, just generates bogus data.
    """
    date = luigi.DateParameter()

    def run(self):
        if False:
            while True:
                i = 10
        '\n        Generates bogus data and writes it into the :py:meth:`~.Streams.output` target.\n        '
        with self.output().open('w') as output:
            for _ in range(1000):
                output.write('{} {} {}\n'.format(random.randint(0, 999), random.randint(0, 999), random.randint(0, 999)))

    def output(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in the local file system.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget(self.date.strftime('data/streams_%Y_%m_%d_faked.tsv'))

class StreamsHdfs(Streams):
    """
    This task performs the same work as :py:class:`~.Streams` but its output is written to HDFS.

    This class uses :py:meth:`~.Streams.run` and
    overrides :py:meth:`~.Streams.output` so redefine HDFS as its target.
    """

    def output(self):
        if False:
            return 10
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget(self.date.strftime('data/streams_%Y_%m_%d_faked.tsv'))

class AggregateArtists(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.Streams.output` and
    writes the result into its :py:meth:`~.AggregateArtists.output` target (local file).
    """
    date_interval = luigi.DateIntervalParameter()

    def output(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget('data/artist_streams_{}.tsv'.format(self.date_interval))

    def requires(self):
        if False:
            i = 10
            return i + 15
        "\n        This task's dependencies:\n\n        * :py:class:`~.Streams`\n\n        :return: list of object (:py:class:`luigi.task.Task`)\n        "
        return [Streams(date) for date in self.date_interval]

    def run(self):
        if False:
            i = 10
            return i + 15
        artist_count = defaultdict(int)
        for t in self.input():
            with t.open('r') as in_file:
                for line in in_file:
                    (_, artist, track) = line.strip().split()
                    artist_count[artist] += 1
        with self.output().open('w') as out_file:
            for (artist, count) in artist_count.items():
                out_file.write('{}\t{}\n'.format(artist, count))

class AggregateArtistsSpark(luigi.contrib.spark.SparkSubmitTask):
    """
    This task runs a :py:class:`luigi.contrib.spark.SparkSubmitTask` task
    over each target data returned by :py:meth:`~/.StreamsHdfs.output` and
    writes the result into its :py:meth:`~.AggregateArtistsSpark.output` target (a file in HDFS).
    """
    date_interval = luigi.DateIntervalParameter()
    '\n    The Pyspark script to run.\n\n    For Spark applications written in Java or Scala, the name of a jar file should be supplied instead.\n    '
    app = 'top_artists_spark.py'
    '\n    Address of the Spark cluster master. In this case, we are not using a cluster, but running\n    Spark in local mode.\n    '
    master = 'local[*]'

    def output(self):
        if False:
            print('Hello World!')
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file in HDFS.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.contrib.hdfs.HdfsTarget('data/artist_streams_%s.tsv' % self.date_interval)

    def requires(self):
        if False:
            return 10
        "\n        This task's dependencies:\n\n        * :py:class:`~.StreamsHdfs`\n\n        :return: list of object (:py:class:`luigi.task.Task`)\n        "
        return [StreamsHdfs(date) for date in self.date_interval]

    def app_options(self):
        if False:
            for i in range(10):
                print('nop')
        return [','.join([p.path for p in self.input()]), self.output().path]

class Top10Artists(luigi.Task):
    """
    This task runs over the target data returned by :py:meth:`~/.AggregateArtists.output` or
    :py:meth:`~/.AggregateArtistsSpark.output` in case :py:attr:`~/.Top10Artists.use_spark` is set and
    writes the result into its :py:meth:`~.Top10Artists.output` target (a file in local filesystem).
    """
    date_interval = luigi.DateIntervalParameter()
    use_spark = luigi.BoolParameter()

    def requires(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        This task's dependencies:\n\n        * :py:class:`~.AggregateArtists` or\n        * :py:class:`~.AggregateArtistsSpark` if :py:attr:`~/.Top10Artists.use_spark` is set.\n\n        :return: object (:py:class:`luigi.task.Task`)\n        "
        if self.use_spark:
            return AggregateArtistsSpark(self.date_interval)
        else:
            return AggregateArtists(self.date_interval)

    def output(self):
        if False:
            print('Hello World!')
        '\n        Returns the target output for this task.\n        In this case, a successful execution of this task will create a file on the local filesystem.\n\n        :return: the target output for this task.\n        :rtype: object (:py:class:`luigi.target.Target`)\n        '
        return luigi.LocalTarget('data/top_artists_%s.tsv' % self.date_interval)

    def run(self):
        if False:
            while True:
                i = 10
        top_10 = nlargest(10, self._input_iterator())
        with self.output().open('w') as out_file:
            for (streams, artist) in top_10:
                out_line = '\t'.join([str(self.date_interval.date_a), str(self.date_interval.date_b), artist, str(streams)])
                out_file.write(out_line + '\n')

    def _input_iterator(self):
        if False:
            while True:
                i = 10
        with self.input().open('r') as in_file:
            for line in in_file:
                (artist, streams) = line.strip().split()
                yield (int(streams), artist)

class ArtistToplistToDatabase(luigi.contrib.postgres.CopyToTable):
    """
    This task runs a :py:class:`luigi.contrib.postgres.CopyToTable` task
    over the target data returned by :py:meth:`~/.Top10Artists.output` and
    writes the result into its :py:meth:`~.ArtistToplistToDatabase.output` target which,
    by default, is :py:class:`luigi.contrib.postgres.PostgresTarget` (a table in PostgreSQL).

    This class uses :py:meth:`luigi.contrib.postgres.CopyToTable.run`
    and :py:meth:`luigi.contrib.postgres.CopyToTable.output`.
    """
    date_interval = luigi.DateIntervalParameter()
    use_spark = luigi.BoolParameter()
    host = 'localhost'
    database = 'toplists'
    user = 'luigi'
    password = 'abc123'
    table = 'top10'
    columns = [('date_from', 'DATE'), ('date_to', 'DATE'), ('artist', 'TEXT'), ('streams', 'INT')]

    def requires(self):
        if False:
            print('Hello World!')
        "\n        This task's dependencies:\n\n        * :py:class:`~.Top10Artists`\n\n        :return: list of object (:py:class:`luigi.task.Task`)\n        "
        return Top10Artists(self.date_interval, self.use_spark)
if __name__ == '__main__':
    luigi.run()