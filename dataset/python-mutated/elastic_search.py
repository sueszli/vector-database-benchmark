from bigdl.orca import OrcaContext
from bigdl.dllib.nncontext import init_nncontext

class elastic_search:
    """
    Primary DataFrame-based loading data from elastic search interface,
    defining API to read data from ES to DataFrame.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def read_df(esConfig, esResource, schema=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Read the data from elastic search into DataFrame.\n\n        :param esConfig: Dictionary which represents configuration for\n               elastic search(eg. ip, port etc).\n        :param esResource: resource file in elastic search.\n        :param schema: Optional. Defines the schema of Spark dataframe.\n                If each column in Es is single value, don't need set schema.\n        :return: Spark DataFrame. Each row represents a document in ES.\n        "
        sc = init_nncontext()
        spark = OrcaContext.get_spark_session()
        reader = spark.read.format('org.elasticsearch.spark.sql')
        for key in esConfig:
            reader.option(key, esConfig[key])
        if schema:
            reader.schema(schema)
        df = reader.load(esResource)
        return df

    @staticmethod
    def flatten_df(df):
        if False:
            return 10
        fields = elastic_search.flatten(df.schema)
        flatten_df = df.select(fields)
        return flatten_df

    @staticmethod
    def flatten(schema, prefix=None):
        if False:
            while True:
                i = 10
        from pyspark.sql.types import StructType
        fields = []
        for field in schema.fields:
            name = prefix + '.' + field.name if prefix else field.name
            dtype = field.dataType
            if isinstance(dtype, StructType):
                fields += elastic_search.flatten(dtype, prefix=name)
            else:
                fields.append(name)
        return fields

    @staticmethod
    def write_df(esConfig, esResource, df):
        if False:
            print('Hello World!')
        '\n        Write the Spark DataFrame to elastic search.\n\n        :param esConfig: Dictionary which represents configuration for\n               elastic search(eg. ip, port etc).\n        :param esResource: resource file in elastic search.\n        :param df: Spark DataFrame that will be saved.\n        '
        wdf = df.write.format('org.elasticsearch.spark.sql').option('es.resource', esResource)
        for key in esConfig:
            wdf.option(key, esConfig[key])
        wdf.save()

    @staticmethod
    def read_rdd(esConfig, esResource=None, filter=None, esQuery=None):
        if False:
            while True:
                i = 10
        '\n        Read the data from elastic search into Spark RDD.\n\n        :param esConfig: Dictionary which represents configuration for\n               elastic search(eg. ip, port, es query etc).\n        :param esResource: Optional. resource file in elastic search.\n               It also can be set in esConfig\n        :param filter: Optional. Request only those fields from Elasticsearch\n        :param esQuery: Optional. es query\n        :return: Spark RDD\n        '
        sc = init_nncontext()
        if 'es.resource' not in esConfig:
            esConfig['es.resource'] = esResource
        if filter is not None:
            esConfig['es.read.source.filter'] = filter
        if esQuery is not None:
            esConfig['es.query'] = esQuery
        rdd = sc.newAPIHadoopRDD('org.elasticsearch.hadoop.mr.EsInputFormat', 'org.apache.hadoop.io.NullWritable', 'org.elasticsearch.hadoop.mr.LinkedMapWritable', conf=esConfig)
        return rdd