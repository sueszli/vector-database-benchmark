"""This is the implementation of SAR."""
import logging
import pandas as pd
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pysarplus import SARModel
import pyspark.sql.functions as F
SIM_COOCCUR = 'cooccurrence'
SIM_JACCARD = 'jaccard'
SIM_LIFT = 'lift'
logging.basicConfig(level=logging.INFO)
log = logging.getLogger('sarplus')

class SARPlus:
    """SAR implementation for PySpark."""

    def __init__(self, spark, col_user='userID', col_item='itemID', col_rating='rating', col_timestamp='timestamp', table_prefix='', similarity_type='jaccard', time_decay_coefficient=30, time_now=None, timedecay_formula=False, threshold=1, cache_path=None):
        if False:
            i = 10
            return i + 15
        "Initialize model parameters\n        Args:\n            spark (pyspark.sql.SparkSession): Spark session\n            col_user (str): user column name\n            col_item (str): item column name\n            col_rating (str): rating column name\n            col_timestamp (str): timestamp column name\n            table_prefix (str): name prefix of the generated tables\n            similarity_type (str): ['cooccurrence', 'jaccard', 'lift']\n                option for computing item-item similarity\n            time_decay_coefficient (float): number of days till\n                ratings are decayed by 1/2.  denominator in time\n                decay.  Zero makes time decay irrelevant\n            time_now (int | None): current time for time decay\n                calculation\n            timedecay_formula (bool): flag to apply time decay\n            threshold (int): item-item co-occurrences below this\n                threshold will be removed\n            cache_path (str): user specified local cache directory for\n                recommend_k_items().  If specified,\n                recommend_k_items() will do C++ based fast\n                predictions.\n        "
        assert threshold > 0
        self.spark = spark
        self.header = {'col_user': col_user, 'col_item': col_item, 'col_rating': col_rating, 'col_timestamp': col_timestamp, 'prefix': table_prefix, 'time_now': time_now, 'time_decay_half_life': time_decay_coefficient * 24 * 60 * 60, 'threshold': threshold}
        if similarity_type not in [SIM_COOCCUR, SIM_JACCARD, SIM_LIFT]:
            raise ValueError('Similarity type must be one of ["cooccurrence" | "jaccard" | "lift"]')
        self.similarity_type = similarity_type
        self.timedecay_formula = timedecay_formula
        self.item_similarity = None
        self.item_frequencies = None
        self.cache_path = cache_path

    def _format(self, string, **kwargs):
        if False:
            i = 10
            return i + 15
        return string.format(**self.header, **kwargs)

    def fit(self, df):
        if False:
            for i in range(10):
                print('nop')
        'Main fit method for SAR.\n\n        Expects the dataframes to have row_id, col_id columns which\n        are indexes, i.e. contain the sequential integer index of the\n        original alphanumeric user and item IDs.  Dataframe also\n        contains rating and timestamp as floats; timestamp is in\n        seconds since Epoch by default.\n\n        Arguments:\n            df (pySpark.DataFrame): input dataframe which contains the\n                index of users and items.\n        '
        df.createOrReplaceTempView(self._format('{prefix}df_train_input'))
        if self.timedecay_formula:
            if self.header['time_now'] is None:
                query = self._format('\n                    SELECT CAST(MAX(`{col_timestamp}`) AS long)\n                    FROM `{prefix}df_train_input`\n                ')
                self.header['time_now'] = self.spark.sql(query).first()[0]
            query = self._format('\n                SELECT `{col_user}`,\n                       `{col_item}`,\n                       SUM(\n                           `{col_rating}` *\n                           POW(2, (CAST(`{col_timestamp}` AS LONG) - {time_now}) / {time_decay_half_life})\n                          ) AS `{col_rating}`\n                FROM `{prefix}df_train_input`\n                GROUP BY `{col_user}`, `{col_item}`\n                CLUSTER BY `{col_user}`\n            ')
            df = self.spark.sql(query)
        elif self.header['col_timestamp'] in df.columns:
            query = self._format('\n                    SELECT `{col_user}`, `{col_item}`, `{col_rating}`\n                    FROM (\n                          SELECT `{col_user}`,\n                                 `{col_item}`,\n                                 `{col_rating}`,\n                                 ROW_NUMBER() OVER user_item_win AS latest\n                          FROM `{prefix}df_train_input`\n                         ) AS reverse_chrono_table\n                    WHERE reverse_chrono_table.latest = 1\n                    WINDOW user_item_win AS (\n                        PARTITION BY `{col_user}`,`{col_item}`\n                        ORDER BY `{col_timestamp}` DESC)\n                ')
            df = self.spark.sql(query)
        df.createOrReplaceTempView(self._format('{prefix}df_train'))
        log.info('sarplus.fit 1/2: compute item cooccurrences...')
        query = self._format('\n            SELECT a.`{col_item}` AS i1,\n                   b.`{col_item}` AS i2,\n                   COUNT(*) AS value\n            FROM `{prefix}df_train` AS a\n            INNER JOIN `{prefix}df_train` AS b\n            ON a.`{col_user}` = b.`{col_user}` AND a.`{col_item}` <= b.`{col_item}`\n            GROUP BY i1, i2\n            HAVING value >= {threshold}\n            CLUSTER BY i1, i2\n        ')
        item_cooccurrence = self.spark.sql(query)
        item_cooccurrence.write.mode('overwrite').saveAsTable(self._format('{prefix}item_cooccurrence'))
        self.item_frequencies = item_cooccurrence.filter(F.col('i1') == F.col('i2')).select(F.col('i1').alias('item_id'), F.col('value').alias('frequency'))
        if self.similarity_type == SIM_LIFT or self.similarity_type == SIM_JACCARD:
            query = self._format('\n                SELECT i1 AS i, value AS margin\n                FROM `{prefix}item_cooccurrence`\n                WHERE i1 = i2\n            ')
            item_marginal = self.spark.sql(query)
            item_marginal.createOrReplaceTempView(self._format('{prefix}item_marginal'))
        if self.similarity_type == SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif self.similarity_type == SIM_JACCARD:
            query = self._format('\n                SELECT i1, i2, value / (m1.margin + m2.margin - value) AS value\n                FROM `{prefix}item_cooccurrence` AS a\n                INNER JOIN `{prefix}item_marginal` AS m1 ON a.i1 = m1.i\n                INNER JOIN `{prefix}item_marginal` AS m2 ON a.i2 = m2.i\n                CLUSTER BY i1, i2\n            ')
            self.item_similarity = self.spark.sql(query)
        elif self.similarity_type == SIM_LIFT:
            query = self._format('\n                SELECT i1, i2, value / (m1.margin * m2.margin) AS value\n                FROM `{prefix}item_cooccurrence` AS a\n                INNER JOIN `{prefix}item_marginal` AS m1 ON a.i1 = m1.i\n                INNER JOIN `{prefix}item_marginal` AS m2 ON a.i2 = m2.i\n                CLUSTER BY i1, i2\n            ')
            self.item_similarity = self.spark.sql(query)
        else:
            raise ValueError('Unknown similarity type: {0}'.format(self.similarity_type))
        log.info('sarplus.fit 2/2: compute similarity metric %s...' % self.similarity_type)
        self.item_similarity.write.mode('overwrite').saveAsTable(self._format('{prefix}item_similarity_upper'))
        query = self._format('\n            SELECT i1, i2, value\n            FROM (\n                  (\n                   SELECT i1, i2, value\n                   FROM `{prefix}item_similarity_upper`\n                  )\n                  UNION ALL\n                  (\n                   SELECT i2 AS i1, i1 AS i2, value\n                   FROM `{prefix}item_similarity_upper`\n                   WHERE i1 <> i2\n                  )\n                 )\n            CLUSTER BY i1\n        ')
        self.item_similarity = self.spark.sql(query)
        self.item_similarity.write.mode('overwrite').saveAsTable(self._format('{prefix}item_similarity'))
        self.spark.sql(self._format('DROP TABLE `{prefix}item_cooccurrence`'))
        self.spark.sql(self._format('DROP TABLE `{prefix}item_similarity_upper`'))
        self.item_similarity = self.spark.table(self._format('{prefix}item_similarity'))

    def get_user_affinity(self, test):
        if False:
            for i in range(10):
                print('nop')
        'Prepare test set for C++ SAR prediction code.\n        Find all items the test users have seen in the past.\n\n        Arguments:\n            test (pySpark.DataFrame): input dataframe which contains test users.\n        '
        test.createOrReplaceTempView(self._format('{prefix}df_test'))
        query = self._format('\n            SELECT DISTINCT `{col_user}`\n            FROM `{prefix}df_test`\n            CLUSTER BY `{col_user}`\n        ')
        df_test_users = self.spark.sql(query)
        df_test_users.write.mode('overwrite').saveAsTable(self._format('{prefix}df_test_users'))
        query = self._format('\n            SELECT a.`{col_user}`,\n                   a.`{col_item}`,\n                   CAST(a.`{col_rating}` AS double) AS `{col_rating}`\n            FROM `{prefix}df_train` AS a\n            INNER JOIN `{prefix}df_test_users` AS b\n            ON a.`{col_user}` = b.`{col_user}`\n            DISTRIBUTE BY `{col_user}`\n            SORT BY `{col_user}`, `{col_item}`\n        ')
        return self.spark.sql(query)

    def _recommend_k_items_fast(self, test, top_k=10, remove_seen=True, n_user_prediction_partitions=200):
        if False:
            while True:
                i = 10
        assert self.cache_path is not None
        log.info('sarplus.recommend_k_items 1/3: create item index')
        query = self._format('\n            SELECT i1, ROW_NUMBER() OVER(ORDER BY i1)-1 AS idx\n            FROM (\n                  SELECT DISTINCT i1\n                  FROM `{prefix}item_similarity`\n                 )\n            CLUSTER BY i1\n        ')
        self.spark.sql(query).write.mode('overwrite').saveAsTable(self._format('{prefix}item_mapping'))
        query = self._format('\n            SELECT a.idx AS i1, b.idx AS i2, is.value\n            FROM `{prefix}item_similarity` AS is,\n                 `{prefix}item_mapping` AS a,\n                 `{prefix}item_mapping` AS b\n            WHERE is.i1 = a.i1 AND i2 = b.i1\n        ')
        self.spark.sql(query).write.mode('overwrite').saveAsTable(self._format('{prefix}item_similarity_mapped'))
        cache_path_output = self.cache_path
        if self.cache_path.startswith('dbfs:'):
            cache_path_input = '/dbfs' + self.cache_path[5:]
        elif self.cache_path.startswith('synfs:'):
            cache_path_input = '/synfs' + self.cache_path[6:]
        else:
            cache_path_input = self.cache_path
        log.info('sarplus.recommend_k_items 2/3: prepare similarity matrix')
        query = self._format('\n            SELECT i1, i2, CAST(value AS DOUBLE) AS value\n            FROM `{prefix}item_similarity_mapped`\n            ORDER BY i1, i2\n        ')
        self.spark.sql(query).coalesce(1).write.format('com.microsoft.sarplus').mode('overwrite').save(cache_path_output)
        self.get_user_affinity(test).createOrReplaceTempView(self._format('{prefix}user_affinity'))
        query = self._format('\n            SELECT `{col_user}`, idx, rating\n            FROM (\n                  SELECT `{col_user}`, b.idx, `{col_rating}` AS rating\n                  FROM `{prefix}user_affinity`\n                  JOIN `{prefix}item_mapping` AS b\n                  ON `{col_item}` = b.i1 \n                 )\n            CLUSTER BY `{col_user}`\n        ')
        pred_input = self.spark.sql(query)
        schema = StructType([StructField('userID', pred_input.schema[self.header['col_user']].dataType, True), StructField('itemID', IntegerType(), True), StructField('score', FloatType(), True)])
        local_header = self.header

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def sar_predict_udf(df):
            if False:
                print('Hello World!')
            model = SARModel(cache_path_input)
            preds = model.predict(df['idx'].values, df['rating'].values, top_k, remove_seen)
            user = df[local_header['col_user']].iloc[0]
            preds_ret = pd.DataFrame([(user, x.id, x.score) for x in preds], columns=range(3))
            return preds_ret
        log.info('sarplus.recommend_k_items 3/3: compute recommendations')
        df_preds = pred_input.repartition(n_user_prediction_partitions, self.header['col_user']).groupby(self.header['col_user']).apply(sar_predict_udf)
        df_preds.createOrReplaceTempView(self._format('{prefix}predictions'))
        query = self._format('\n            SELECT userID AS `{col_user}`, b.i1 AS `{col_item}`, score\n            FROM `{prefix}predictions` AS p, `{prefix}item_mapping` AS b\n            WHERE p.itemID = b.idx\n        ')
        return self.spark.sql(query)

    def _recommend_k_items_slow(self, test, top_k=10, remove_seen=True):
        if False:
            while True:
                i = 10
        'Recommend top K items for all users which are in the test set.\n\n        Args:\n            test: test Spark dataframe\n            top_k: top n items to return\n            remove_seen: remove items test users have already seen in\n                the past from the recommended set.\n        '
        if remove_seen:
            raise ValueError('Not implemented')
        self.get_user_affinity(test).write.mode('overwrite').saveAsTable(self._format('{prefix}user_affinity'))
        query = self._format('\n            SELECT `{col_user}`, `{col_item}`, score\n            FROM (\n                  SELECT df.`{col_user}`,\n                         s.i2 AS `{col_item}`,\n                         SUM(df.`{col_rating}` * s.value) AS score,\n                         ROW_NUMBER() OVER w AS rank\n                  FROM `{prefix}user_affinity` AS df,\n                       `{prefix}item_similarity` AS s\n                  WHERE df.`{col_item}` = s.i1\n                  GROUP BY df.`{col_user}`, s.i2\n                  WINDOW w AS (\n                      PARTITION BY `{col_user}`\n                      ORDER BY SUM(df.`{col_rating}` * s.value) DESC)\n                 )\n            WHERE rank <= {top_k}\n        ', top_k=top_k)
        return self.spark.sql(query)

    def recommend_k_items(self, test, top_k=10, remove_seen=True, use_cache=False, n_user_prediction_partitions=200):
        if False:
            while True:
                i = 10
        'Recommend top K items for all users which are in the test set.\n\n        Args:\n            test (pyspark.sql.DataFrame): test Spark dataframe.\n            top_k (int): top n items to return.\n            remove_seen (bool): remove items test users have already\n                seen in the past from the recommended set.\n            use_cache (bool): use specified local directory stored in\n                `self.cache_path` as cache for C++ based fast\n                predictions.\n            n_user_prediction_partitions (int): prediction partitions.\n\n        Returns:\n            pyspark.sql.DataFrame: Spark dataframe with recommended items\n        '
        if not use_cache:
            return self._recommend_k_items_slow(test, top_k, remove_seen)
        elif self.cache_path is not None:
            return self._recommend_k_items_fast(test, top_k, remove_seen, n_user_prediction_partitions)
        else:
            raise ValueError('No cache_path specified')

    def get_topk_most_similar_users(self, test, user, top_k=10):
        if False:
            print('Hello World!')
        'Based on user affinity towards items, calculate the top k most\n            similar users from test dataframe to the given user.\n\n        Args:\n            test (pyspark.sql.DataFrame): test Spark dataframe.\n            user (int): user to retrieve most similar users for.\n            top_k (int): number of top items to recommend.\n\n        Returns:\n            pyspark.sql.DataFrame: Spark dataframe with top k most similar users\n            from test and their similarity scores in descending order.\n        '
        if len(test.filter(test['user_id'].contains(user)).collect()) == 0:
            raise ValueError('Target user must exist in the input dataframe')
        test_affinity = self.get_user_affinity(test).alias('matrix')
        num_test_users = test_affinity.select('user_id').distinct().count() - 1
        if num_test_users < top_k:
            log.warning('Number of users is less than top_k, limiting top_k to number of users')
        k = min(top_k, num_test_users)
        user_affinity = test_affinity.where(F.col('user_id') == user).alias('user')
        df_similar_users = test_affinity.join(user_affinity, test_affinity['item_id'] == user_affinity['item_id'], 'outer').withColumn('prod', F.when(F.col('matrix.user_id') == user, -float('inf')).when(F.col('user.rating').isNotNull(), F.col('matrix.rating') * F.col('user.rating')).otherwise(0.0)).groupBy('matrix.user_id').agg(F.sum('prod').alias('similarity')).orderBy('similarity', ascending=False).limit(k)
        return df_similar_users

    def get_popularity_based_topk(self, top_k=10, items=True):
        if False:
            i = 10
            return i + 15
        'Get top K most frequently occurring items across all users.\n\n        Args:\n            top_k (int): number of top items to recommend.\n            items (bool): if false, return most frequent users instead.\n\n        Returns:\n            pyspark.sql.DataFrame: Spark dataframe with top k most popular items\n            and their frequencies in descending order.\n        '
        if not items:
            raise ValueError('Not implemented')
        return self.item_frequencies.orderBy('frequency', ascending=False).limit(top_k)