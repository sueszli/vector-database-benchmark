"""
A module for I/O between Vaex and Google BigQuery.

Requires:
 - google.cloud.bigquery
 - google.cloud.bigquery_storage
"""
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from vaex.docstrings import docsubst
import vaex.utils
google = vaex.utils.optional_import('google', modules=['google.cloud.bigquery', 'google.cloud.bigquery_storage'])

def from_query(query, client_project=None, credentials=None):
    if False:
        return 10
    'Make a query to Google BigQuery and get the result as a Vaex DataFrame.\n\n    :param str query: The SQL query.\n    :param str client_project: The ID of the project that executes the query. Will be passed when creating a job. If `None`, falls back to the default inferred from the environment.\n    :param credentials: The authorization credentials to attach to requests. See google.auth.credentials.Credentials for more details.\n    :rtype: DataFrame\n\n    Example\n\n    >>> import os\n    os.environ[\'GOOGLE_APPLICATION_CREDENTIALS\'] = \'../path/to/project_access_key.json\'\n    >>> from vaex.contrib.io.gbq import from_query\n\n    >>> query = """\n        select * from `bigquery-public-data.ml_datasets.iris`\n        where species = "virginica"\n    """\n\n    >>> df = from_query(query=query)\n    >>> df.head(3)\n    #    sepal_length    sepal_width    petal_length    petal_width  species\n    0             4.9            2.5             4.5            1.7  virginica\n    1             5.7            2.5             5              2    virginica\n    2             6              2.2             5              1.5  virginica\n\n    '
    client = google.cloud.bigquery.Client(project=client_project, credentials=credentials)
    job = client.query(query=query)
    return vaex.from_arrow_table(job.to_arrow())

@docsubst
def from_table(project, dataset, table, columns=None, condition=None, export=None, fs=None, fs_options=None, client_project=None, credentials=None):
    if False:
        return 10
    'Download (stream) an entire Google BigQuery table locally.\n\n    :param str project: The Google BigQuery project that owns the table.\n    :param str dataset: The dataset the table is part of.\n    :param str table: The name of the table\n    :param list columns: A list of columns (field names) to download. If None, all columns will be downloaded.\n    :param str condition: SQL text filtering statement, similar to a WHERE clause in a query. Aggregates are not supported.\n    :param str export: Pass an filename or path to download the table as an Apache Arrow file, and leverage memory mapping. If `None` the DataFrame is in memory.\n    :param fs: Valid if export is not None. {fs}\n    :param fs: Valid if export is not None. {fs_options}\n    :param str client_project: The ID of the project that executes the query. Will be passed when creating a job. If `None`, it will be set with the same value as `project`.\n    :param credentials: The authorization credentials to attach to requests. See google.auth.credentials.Credentials for more details.\n    :rtype: DataFrame\n\n    Example:\n\n    >>> import os\n    >>> os.environ[\'GOOGLE_APPLICATION_CREDENTIALS\'] = \'../path/to/project_access_key.json\'\n    >>> from vaex.contrib.io.gbq import from_table\n\n    >>> client_project = \'my_project_id\'\n    >>> project = \'bigquery-public-data\'\n    >>> dataset = \'ml_datasets\'\n    >>> table = \'iris\'\n    >>> columns = [\'species\', \'sepal_width\', \'petal_width\']\n    >>> conditions = \'species = "virginica"\'\n    >>> df = from_table(project=project,\n                                            dataset=dataset,\n                                            table=table,\n                                            columns=columns,\n                                            condition=conditions,\n                                            client_project=client_project)\n    >>> df.head(3)\n    #    sepal_width    petal_width  species\n    0            2.5            1.7  virginica\n    1            2.5            2    virginica\n    2            2.2            1.5  virginica\n    >>>\n\n    '
    bq_table = f'projects/{project}/datasets/{dataset}/tables/{table}'
    req_sess = google.cloud.bigquery_storage.types.ReadSession(table=bq_table, data_format=google.cloud.bigquery_storage.types.DataFormat.ARROW)
    req_sess.read_options.selected_fields = columns
    req_sess.read_options.row_restriction = condition
    client = google.cloud.bigquery_storage.BigQueryReadClient(credentials=credentials)
    parent = f'projects/{client_project or project}'
    session = client.create_read_session(parent=parent, read_session=req_sess, max_stream_count=1)
    reader = client.read_rows(session.streams[0].name)
    if export is None:
        arrow_table = reader.to_arrow(session)
        return vaex.from_arrow_table(arrow_table)
    else:
        pages = reader.rows(session).pages
        first_batch = pages.__next__().to_arrow()
        schema = first_batch.schema
        with vaex.file.open(path=export, mode='wb', fs=fs, fs_options=fs_options) as sink:
            with pa.RecordBatchStreamWriter(sink, schema) as writer:
                writer.write_batch(first_batch)
                for page in pages:
                    batch = page.to_arrow()
                    writer.write_batch(batch)
        return vaex.open(export)

def to_table(df, dataset, table, job_config=None, client_project=None, credentials=None, chunk_size=None, progress=None):
    if False:
        for i in range(10):
            print('nop')
    "Upload a Vaex DataFrame to a Google BigQuery Table.\n\n    Note that the upload creates a temporary parquet file on the local disk, which is then upload to\n    Google BigQuery.\n\n    :param DataFrame df: The Vaex DataFrame to be uploaded.\n    :param str dataset: The name of the dataset to which the table belongs\n    :param str table: The name of the table\n    :param job_config: Optional, an instance of google.cloud.bigquery.job.load.LoadJobConfig\n    :param str client_project: The ID of the project that executes the query. Will be passed when creating a job. If `None`, falls back to the default inferred from the environment.\n    :param credentials: The authorization credentials to attach to requests. See google.auth.credentials.Credentials for more details.\n    :param chunk_size: In case the local disk space is limited, export the dataset in chunks.\n                       This is considerably slower than a single file upload and it should be avoided.\n    :param progress: Valid only if chunk_size is not None. A callable that takes one argument (a floating point value between 0 and 1) indicating the progress, calculations are cancelled when this callable returns False\n\n    Example:\n\n    >>> import os\n    >>> os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../path/to/project_access_key.json'\n    >>> import vaex\n    >>> from vaex.contrib.io.gbq import to_table\n\n    >>> df = vaex.example()\n    >>> dataset = 'my_dataset'\n    >>> table = 'my_table'\n\n    >>> to_table(df=df, dataset=dataset, table=table)\n\n    "
    client = google.cloud.bigquery.Client(project=client_project, credentials=credentials)
    if job_config is not None:
        assert isinstance(job_config, google.cloud.bigquery.job.load.LoadJobConfig)
        job_config.source_format = google.cloud.bigquery.SourceFormat.PARQUET
    else:
        job_config = google.cloud.bigquery.LoadJobConfig(source_format=google.cloud.bigquery.SourceFormat.PARQUET)
    table_bq = f'{dataset}.{table}'
    if chunk_size is None:
        with tempfile.TemporaryFile(suffix='.parquet') as tmp:
            df.export_parquet('tmp.parquet')
            with open('tmp.parquet', 'rb') as source_file:
                job = client.load_table_from_file(source_file, table_bq, job_config=job_config)
            job.result()
    else:
        progressbar = vaex.utils.progressbars(progress)
        n_samples = len(df)
        for (i1, i2, table) in df.to_arrow_table(chunk_size=chunk_size):
            progressbar(i1 / n_samples)
            with tempfile.TemporaryFile(suffix='.parquet') as tmp:
                pq.write_table(table, 'tmp.parquet')
                with open('tmp.parquet', 'rb') as source_file:
                    job = client.load_table_from_file(source_file, table_bq, job_config=job_config)
                job.result()
        progressbar(1.0)