#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from bigdl.orca import init_orca_context, stop_orca_context
from bigdl.friesian.feature import FeatureTable
from pyspark.ml.linalg import DenseVector, VectorUDT
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from bigdl.dllib.nnframes.tree_model import *
import argparse
import time
from bigdl.dllib.utils.log4Error import *


spark_conf = {"spark.network.timeout": "10000000",
              "spark.sql.broadcastTimeout": "7200",
              "spark.sql.shuffle.partitions": "500",
              "spark.locality.wait": "0s",
              "spark.sql.hive.filesourcePartitionFileCacheSize": "4096000000",
              "spark.sql.crossJoin.enabled": "true",
              "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
              "spark.kryo.unsafe": "true",
              "spark.kryoserializer.buffer.max": "1024m",
              "spark.task.cpus": "4",
              "spark.executor.heartbeatInterval": "200s",
              "spark.driver.maxResultSize": "40G",
              "spark.app.name": "recsys-xgb"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XGBoost Training')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn or standalone.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=4,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="160g",
                        help='The executor memory.')
    parser.add_argument('--num_executor', type=int, default=4,
                        help='The number of executor.')
    parser.add_argument('--driver_cores', type=int, default=4,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="36g",
                        help='The driver memory.')
    parser.add_argument('--model_dir', default='snapshot', type=str,
                        help='nativeModel directory name (default: nativeModel)')
    parser.add_argument('--data_dir', type=str, help='data directory')

    args = parser.parse_args()

    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores,
                               memory=args.executor_memory, conf=spark_conf)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executor, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory,
                               conf=spark_conf)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        invalidInputError(False,
                          "cluster_mode should be one of 'local', 'yarn' and"
                          " 'spark-submit', but got " + args.cluster_mode)

    num_cols = ["enaging_user_follower_count", 'enaging_user_following_count',
                "engaged_with_user_follower_count", "engaged_with_user_following_count",
                "len_hashtags", "len_domains", "len_links"]
    cat_cols = ["engaged_with_user_is_verified", "enaging_user_is_verified", "present_media",
                "tweet_type", "language", 'present_media_language']
    embed_cols = ["enaging_user_id", "engaged_with_user_id", "hashtags", "present_links",
                  "present_domains"]

    features = num_cols + [col + "_te_label" for col in cat_cols] + \
        [col + "_te_label" for col in embed_cols]
    begin = time.time()
    train_tbl = FeatureTable.read_parquet(args.data_dir + "/train_parquet")\
        .drop("tweet_timestamp", "enaging_user_account_creation", "reply_timestamp", "text_tokens",
              "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")
    test_tbl = FeatureTable.read_parquet(args.data_dir + "/test_parquet")\
        .drop("tweet_timestamp", "enaging_user_account_creation", "reply_timestamp", "text_tokens",
              "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")

    train_tbl.cache()
    test_tbl.cache()
    full = train_tbl.concat(test_tbl)
    full, target_codes = full.target_encode(cat_cols=cat_cols + embed_cols, target_cols=["label"])
    for code in target_codes:
        code.cache()

    train = train_tbl\
        .encode_target(target_cols="label", targets=target_codes)\
        .merge_cols(features, "features") \
        .select(["label", "features"])\
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT())
    train.show(5, False)

    test = test_tbl\
        .encode_target(target_cols="label", targets=target_codes) \
        .merge_cols(features, "features") \
        .select(["label", "features"]) \
        .apply("features", "features", lambda x: DenseVector(x), VectorUDT())

    test.show(5, False)
    train = train.cache()
    test = test.cache()
    print("training size:", train.size())
    print("test size:", test.size())
    train_tbl.uncache()
    test_tbl.uncache()
    for code in target_codes:
        code.uncache()

    preprocess = time.time()
    print("feature preprocessing time: %.2f" % (preprocess - begin))

    params = {"tree_method": 'hist', "eta": 0.1, "gamma": 0.1,
              "min_child_weight": 30, "reg_lambda": 1, "scale_pos_weight": 2,
              "subsample": 1, "objective": "binary:logistic"}

    for eta in [0.1]:
        for max_depth in [6]:
            for num_round in [200]:
                params.update({"eta": eta, "max_depth": max_depth, "num_round": num_round})
                classifier = XGBClassifier(params)
                xgbmodel = classifier.fit(train.df)
                xgbmodel.saveModel(args.model_dir)
                xgbmodel = XGBClassifierModel.loadModel(args.model_dir, 2)
                xgbmodel.setFeaturesCol("features")
                predicts = xgbmodel.transform(test.df).drop("features")
                predicts.cache()
                predicts.show(5, False)

                evaluator = BinaryClassificationEvaluator(labelCol="label",
                                                          rawPredictionCol="rawPrediction")
                auc = evaluator.evaluate(predicts, {evaluator.metricName: "areaUnderROC"})

                evaluator2 = MulticlassClassificationEvaluator(labelCol="label",
                                                               predictionCol="prediction")
                acc = evaluator2.evaluate(predicts, {evaluator2.metricName: "accuracy"})
                print(params)
                print("AUC: %.2f" % (auc * 100.0))
                print("Accuracy: %.2f" % (acc * 100.0))

                predicts.unpersist(blocking=True)

    end = time.time()
    print("training time: %.2f" % (end - preprocess))
    print(end - begin)
    stop_orca_context()
