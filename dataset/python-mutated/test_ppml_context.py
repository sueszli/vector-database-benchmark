import unittest
import os
import random
import shutil
from bigdl.ppml.ppml_context import *
from pyspark.sql.types import StructType, StructField, StringType
resource_path = os.path.join(os.path.dirname(__file__), 'resources')

class TestPPMLContext(unittest.TestCase):
    app_id = ''.join([str(random.randint(0, 9)) for i in range(12)])
    app_key = ''.join([str(random.randint(0, 9)) for j in range(12)])
    df = None
    data_content = None
    sc = None

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            print('Hello World!')
        if not os.path.exists(resource_path):
            os.mkdir(resource_path)
        primary_key_path = os.path.join(resource_path, 'primaryKey')
        conf = {'spark.app.name': 'PPML TEST', 'spark.hadoop.io.compression.codecs': 'com.intel.analytics.bigdl.ppml.crypto.CryptoCodec', 'spark.bigdl.primaryKey.defaultKey.kms.type': 'SimpleKeyManagementService', 'spark.bigdl.primaryKey.defaultKey.kms.appId': cls.app_id, 'spark.bigdl.primaryKey.defaultKey.kms.apiKey': cls.app_key, 'spark.bigdl.primaryKey.defaultKey.material': primary_key_path}
        init_spark_on_local(conf=conf)
        init_keys(cls.app_id, cls.app_key, primary_key_path)
        args = {'kms_type': 'SimpleKeyManagementService', 'app_id': cls.app_id, 'api_key': cls.app_key, 'primary_key_material': primary_key_path}
        cls.sc = PPMLContext('testApp', args)
        data = [('Java', '20000'), ('Python', '100000'), ('Scala', '3000')]
        cls.df = cls.sc.spark.createDataFrame(data).toDF('language', 'user')
        cls.df = cls.df.repartition(1)
        cls.data_content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in cls.df.orderBy('language').collect()])

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            return 10
        if os.path.exists(resource_path):
            shutil.rmtree(resource_path)

    def test_schema_by_write_and_read_csv(self):
        if False:
            while True:
                i = 10
        path = os.path.join(resource_path, 'schema_csv/plain')
        test_schema = StructType([StructField('language', StringType()), StructField('user', StringType())])
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT).mode('overwrite').option('header', True).csv(path)
        df = self.sc.read(CryptoMode.PLAIN_TEXT).option('header', 'true').schema(test_schema).csv(path)
        csv_content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in df.orderBy('language').collect()])
        self.assertEqual(csv_content, self.data_content)

    def test_sql_by_write_and_read_encrypted_csv(self):
        if False:
            while True:
                i = 10
        path = os.path.join(resource_path, 'sql_csv/plain')
        test_schema = StructType([StructField('language', StringType()), StructField('user', StringType())])
        self.sc.write(self.df, CryptoMode.AES_CBC_PKCS5PADDING).mode('overwrite').option('header', True).csv(path)
        df = self.sc.read(CryptoMode.AES_CBC_PKCS5PADDING).option('header', 'true').schema(test_schema).csv(path)
        df.createOrReplaceTempView('test')
        sqlTest = 'select language, user from test'
        data = self.sc.sql(sqlTest)
        csv_content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in data.orderBy('language').collect()])
        self.assertEqual(csv_content, self.data_content)

    def test_write_and_read_plain_csv(self):
        if False:
            return 10
        path = os.path.join(resource_path, 'csv/plain')
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT).mode('overwrite').option('header', True).csv(path)
        df = self.sc.read(CryptoMode.PLAIN_TEXT).option('header', 'true').csv(path)
        csv_content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in df.orderBy('language').collect()])
        self.assertEqual(csv_content, self.data_content)

    def test_write_and_read_encrypted_csv(self):
        if False:
            i = 10
            return i + 15
        path = os.path.join(resource_path, 'csv/encrypted')
        self.sc.write(self.df, CryptoMode.AES_CBC_PKCS5PADDING).mode('overwrite').option('header', True).csv(path)
        df = self.sc.read(CryptoMode.AES_CBC_PKCS5PADDING).option('header', 'true').csv(path)
        csv_content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in df.orderBy('language').collect()])
        self.assertEqual(csv_content, self.data_content)

    def test_write_and_read_plain_parquet(self):
        if False:
            i = 10
            return i + 15
        parquet_path = os.path.join(resource_path, 'parquet/plain-parquet')
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT).mode('overwrite').parquet(parquet_path)
        df_from_parquet = self.sc.read(CryptoMode.PLAIN_TEXT).parquet(parquet_path)
        content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in df_from_parquet.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)

    def test_write_and_read_encrypted_parquet(self):
        if False:
            print('Hello World!')
        parquet_path = os.path.join(resource_path, 'parquet/en-parquet')
        self.sc.write(self.df, CryptoMode.AES_GCM_CTR_V1).mode('overwrite').parquet(parquet_path)
        df_from_parquet = self.sc.read(CryptoMode.AES_GCM_CTR_V1).parquet(parquet_path)
        content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in df_from_parquet.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)

    def test_plain_text_file(self):
        if False:
            i = 10
            return i + 15
        path = os.path.join(resource_path, 'csv/plain')
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT).mode('overwrite').option('header', True).csv(path)
        rdd = self.sc.textfile(path)
        rdd_content = '\n'.join([line for line in rdd.collect()])
        self.assertEqual(rdd_content, 'language,user\n' + self.data_content)

    def test_encrypted_text_file(self):
        if False:
            return 10
        path = os.path.join(resource_path, 'csv/encrypted')
        self.sc.write(self.df, CryptoMode.AES_CBC_PKCS5PADDING).mode('overwrite').option('header', True).csv(path)
        rdd = self.sc.textfile(path=path, crypto_mode=CryptoMode.AES_CBC_PKCS5PADDING)
        rdd_content = '\n'.join([line for line in rdd.collect()])
        self.assertEqual(rdd_content, 'language,user\n' + self.data_content)

    def test_write_and_read_plain_json(self):
        if False:
            while True:
                i = 10
        json_path = os.path.join(resource_path, 'json/plain-json')
        self.sc.write(self.df, CryptoMode.PLAIN_TEXT).mode('overwrite').json(json_path)
        df_from_json = self.sc.read(CryptoMode.PLAIN_TEXT).json(json_path)
        content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in df_from_json.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)

    def test_write_and_read_encrypted_json(self):
        if False:
            return 10
        json_path = os.path.join(resource_path, 'json/en-json')
        self.sc.write(self.df, CryptoMode.AES_CBC_PKCS5PADDING).mode('overwrite').json(json_path)
        df_from_json = self.sc.read(CryptoMode.AES_CBC_PKCS5PADDING).json(json_path)
        content = '\n'.join([str(v['language']) + ',' + str(v['user']) for v in df_from_json.orderBy('language').collect()])
        self.assertEqual(content, self.data_content)
if __name__ == '__main__':
    unittest.main()