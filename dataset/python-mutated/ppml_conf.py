import os
from pyspark import SparkConf

class PPMLConf:

    def __init__(self, k8s_enabled=True, sgx_enabled=True):
        if False:
            while True:
                i = 10
        self.spark_conf = self.init_spark_on_k8s_conf(SparkConf(), k8s_enabled, sgx_enabled)

    def set(self, key, value):
        if False:
            return 10
        self.spark_conf = self.spark_conf.set(key, value)
        return self

    def setAppName(self, app_name):
        if False:
            i = 10
            return i + 15
        self.spark_conf = self.spark_conf.setAppName(app_name)
        return self

    def conf(self):
        if False:
            while True:
                i = 10
        return self.spark_conf

    def init_spark_on_k8s_conf(self, spark_conf, k8s_enabled, sgx_enabled):
        if False:
            print('Hello World!')
        if not k8s_enabled:
            spark_conf = spark_conf.setMaster('local[4]').set('spark.python.use.daemon', 'false').set('park.python.worker.reuse', 'false')
            return spark_conf
        master = os.getenv('RUNTIME_SPARK_MASTER')
        image = os.getenv('RUNTIME_K8S_SPARK_IMAGE')
        driver_ip = os.getenv('RUNTIME_DRIVER_HOST')
        print('k8s master url is ' + str(master))
        print('executor image is ' + str(image))
        print('driver ip is ' + str(driver_ip))
        secure_password = os.getenv('secure_password')
        spark_conf = spark_conf.setMaster(master).set('spark.submit.deployMode', 'client').set('spark.kubernetes.container.image', image).set('spark.driver.host', driver_ip).set('spark.kubernetes.driver.podTemplateFile', '/ppml/spark-driver-template.yaml').set('spark.kubernetes.executor.podTemplateFile', '/ppml/spark-executor-template.yaml').set('spark.kubernetes.authenticate.driver.serviceAccountName', 'spark').set('spark.kubernetes.executor.deleteOnTermination', 'false').set('spark.network.timeout', '10000000').set('spark.executor.heartbeatInterval', '10000000').set('spark.python.use.daemon', 'false').set('spark.python.worker.reuse', 'false').set('spark.authenticate', 'true').set('spark.authenticate.secret', secure_password).set('spark.kubernetes.executor.secretKeyRef.SPARK_AUTHENTICATE_SECRET', 'spark-secret:secret').set('spark.kubernetes.driver.secretKeyRef.SPARK_AUTHENTICATE_SECRET', 'spark-secret:secret')
        if sgx_enabled:
            spark_conf = spark_conf.set('spark.kubernetes.sgx.enabled', 'true').set('spark.kubernetes.sgx.driver.jvm.mem', '1g').set('spark.kubernetes.sgx.executor.jvm.mem', '3g')
        return spark_conf