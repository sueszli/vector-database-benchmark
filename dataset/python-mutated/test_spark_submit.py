from __future__ import annotations
import os
from io import StringIO
from unittest.mock import call, patch
import pytest
from airflow.exceptions import AirflowException
from airflow.models import Connection
from airflow.providers.apache.spark.hooks.spark_submit import SparkSubmitHook
from airflow.utils import db
pytestmark = pytest.mark.db_test

class TestSparkSubmitHook:
    _spark_job_file = 'test_application.py'
    _config = {'conf': {'parquet.compression': 'SNAPPY'}, 'conn_id': 'default_spark', 'files': 'hive-site.xml', 'py_files': 'sample_library.py', 'archives': 'sample_archive.zip#SAMPLE', 'jars': 'parquet.jar', 'packages': 'com.databricks:spark-avro_2.11:3.2.0', 'exclude_packages': 'org.bad.dependency:1.0.0', 'repositories': 'http://myrepo.org', 'total_executor_cores': 4, 'executor_cores': 4, 'executor_memory': '22g', 'keytab': 'privileged_user.keytab', 'principal': 'user/spark@airflow.org', 'proxy_user': 'sample_user', 'name': 'spark-job', 'num_executors': 10, 'verbose': True, 'driver_memory': '3g', 'java_class': 'com.foo.bar.AppMain', 'application_args': ['-f', 'foo', '--bar', 'bar', '--with-spaces', 'args should keep embedded spaces', 'baz'], 'use_krb5ccache': True}

    @staticmethod
    def cmd_args_to_dict(list_cmd):
        if False:
            while True:
                i = 10
        return_dict = {}
        for (arg1, arg2) in zip(list_cmd, list_cmd[1:]):
            if arg1.startswith('--'):
                return_dict[arg1] = arg2
        return return_dict

    def setup_method(self):
        if False:
            return 10
        db.merge_conn(Connection(conn_id='spark_yarn_cluster', conn_type='spark', host='yarn://yarn-master', extra='{"queue": "root.etl", "deploy-mode": "cluster"}'))
        db.merge_conn(Connection(conn_id='spark_k8s_cluster', conn_type='spark', host='k8s://https://k8s-master', extra='{"deploy-mode": "cluster", "namespace": "mynamespace"}'))
        db.merge_conn(Connection(conn_id='spark_default_mesos', conn_type='spark', host='mesos://host', port=5050))
        db.merge_conn(Connection(conn_id='spark_binary_set', conn_type='spark', host='yarn', extra='{"spark-binary": "spark2-submit"}'))
        db.merge_conn(Connection(conn_id='spark_binary_set_spark3_submit', conn_type='spark', host='yarn', extra='{"spark-binary": "spark3-submit"}'))
        db.merge_conn(Connection(conn_id='spark_custom_binary_set', conn_type='spark', host='yarn', extra='{"spark-binary": "spark-other-submit"}'))
        db.merge_conn(Connection(conn_id='spark_home_set', conn_type='spark', host='yarn', extra='{"spark-home": "/custom/spark-home/path"}'))
        db.merge_conn(Connection(conn_id='spark_standalone_cluster', conn_type='spark', host='spark://spark-standalone-master:6066', extra='{"deploy-mode": "cluster"}'))
        db.merge_conn(Connection(conn_id='spark_standalone_cluster_client_mode', conn_type='spark', host='spark://spark-standalone-master:6066', extra='{"deploy-mode": "client"}'))

    @patch('airflow.providers.apache.spark.hooks.spark_submit.os.getenv', return_value='/tmp/airflow_krb5_ccache')
    def test_build_spark_submit_command(self, mock_get_env):
        if False:
            for i in range(10):
                print('nop')
        hook = SparkSubmitHook(**self._config)
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        expected_build_cmd = ['spark-submit', '--master', 'yarn', '--conf', 'parquet.compression=SNAPPY', '--files', 'hive-site.xml', '--py-files', 'sample_library.py', '--archives', 'sample_archive.zip#SAMPLE', '--jars', 'parquet.jar', '--packages', 'com.databricks:spark-avro_2.11:3.2.0', '--exclude-packages', 'org.bad.dependency:1.0.0', '--repositories', 'http://myrepo.org', '--num-executors', '10', '--total-executor-cores', '4', '--executor-cores', '4', '--executor-memory', '22g', '--driver-memory', '3g', '--keytab', 'privileged_user.keytab', '--principal', 'user/spark@airflow.org', '--conf', 'spark.kerberos.renewal.credentials=ccache', '--proxy-user', 'sample_user', '--name', 'spark-job', '--class', 'com.foo.bar.AppMain', '--verbose', 'test_application.py', '-f', 'foo', '--bar', 'bar', '--with-spaces', 'args should keep embedded spaces', 'baz']
        assert expected_build_cmd == cmd
        mock_get_env.assert_called_with('KRB5CCNAME')

    @patch('airflow.configuration.conf.get_mandatory_value')
    def test_resolve_spark_submit_env_vars_use_krb5ccache_missing_principal(self, mock_get_madantory_value):
        if False:
            for i in range(10):
                print('nop')
        mock_principle = 'airflow'
        mock_get_madantory_value.return_value = mock_principle
        hook = SparkSubmitHook(conn_id='spark_yarn_cluster', principal=None, use_krb5ccache=True)
        mock_get_madantory_value.assert_called_with('kerberos', 'principal')
        assert hook._principal == mock_principle

    def test_resolve_spark_submit_env_vars_use_krb5ccache_missing_KRB5CCNAME_env(self):
        if False:
            for i in range(10):
                print('nop')
        hook = SparkSubmitHook(conn_id='spark_yarn_cluster', principal='user/spark@airflow.org', use_krb5ccache=True)
        with pytest.raises(AirflowException, match='KRB5CCNAME environment variable required to use ticket ccache is missing.'):
            hook._build_spark_submit_command(self._spark_job_file)

    def test_build_track_driver_status_command(self):
        if False:
            print('Hello World!')
        hook_spark_standalone_cluster = SparkSubmitHook(conn_id='spark_standalone_cluster')
        hook_spark_standalone_cluster._driver_id = 'driver-20171128111416-0001'
        hook_spark_yarn_cluster = SparkSubmitHook(conn_id='spark_yarn_cluster')
        hook_spark_yarn_cluster._driver_id = 'driver-20171128111417-0001'
        build_track_driver_status_spark_standalone_cluster = hook_spark_standalone_cluster._build_track_driver_status_command()
        build_track_driver_status_spark_yarn_cluster = hook_spark_yarn_cluster._build_track_driver_status_command()
        expected_spark_standalone_cluster = ['/usr/bin/curl', '--max-time', '30', 'http://spark-standalone-master:6066/v1/submissions/status/driver-20171128111416-0001']
        expected_spark_yarn_cluster = ['spark-submit', '--master', 'yarn://yarn-master', '--status', 'driver-20171128111417-0001']
        assert expected_spark_standalone_cluster == build_track_driver_status_spark_standalone_cluster
        assert expected_spark_yarn_cluster == build_track_driver_status_spark_yarn_cluster

    @patch('airflow.providers.apache.spark.hooks.spark_submit.subprocess.Popen')
    def test_spark_process_runcmd(self, mock_popen):
        if False:
            while True:
                i = 10
        mock_popen.return_value.stdout = StringIO('stdout')
        mock_popen.return_value.stderr = StringIO('stderr')
        mock_popen.return_value.wait.return_value = 0
        hook = SparkSubmitHook(conn_id='')
        hook.submit()
        assert mock_popen.mock_calls[0] == call(['spark-submit', '--master', 'yarn', '--name', 'default-name', ''], stderr=-2, stdout=-1, universal_newlines=True, bufsize=-1)

    def test_resolve_should_track_driver_status(self):
        if False:
            print('Hello World!')
        hook_default = SparkSubmitHook(conn_id='')
        hook_spark_yarn_cluster = SparkSubmitHook(conn_id='spark_yarn_cluster')
        hook_spark_k8s_cluster = SparkSubmitHook(conn_id='spark_k8s_cluster')
        hook_spark_default_mesos = SparkSubmitHook(conn_id='spark_default_mesos')
        hook_spark_binary_set = SparkSubmitHook(conn_id='spark_binary_set')
        hook_spark_standalone_cluster = SparkSubmitHook(conn_id='spark_standalone_cluster')
        should_track_driver_status_default = hook_default._resolve_should_track_driver_status()
        should_track_driver_status_spark_yarn_cluster = hook_spark_yarn_cluster._resolve_should_track_driver_status()
        should_track_driver_status_spark_k8s_cluster = hook_spark_k8s_cluster._resolve_should_track_driver_status()
        should_track_driver_status_spark_default_mesos = hook_spark_default_mesos._resolve_should_track_driver_status()
        should_track_driver_status_spark_binary_set = hook_spark_binary_set._resolve_should_track_driver_status()
        should_track_driver_status_spark_standalone_cluster = hook_spark_standalone_cluster._resolve_should_track_driver_status()
        assert should_track_driver_status_default is False
        assert should_track_driver_status_spark_yarn_cluster is False
        assert should_track_driver_status_spark_k8s_cluster is False
        assert should_track_driver_status_spark_default_mesos is False
        assert should_track_driver_status_spark_binary_set is False
        assert should_track_driver_status_spark_standalone_cluster is True

    def test_resolve_connection_yarn_default(self):
        if False:
            for i in range(10):
                print('nop')
        hook = SparkSubmitHook(conn_id='')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        dict_cmd = self.cmd_args_to_dict(cmd)
        expected_spark_connection = {'master': 'yarn', 'spark_binary': 'spark-submit', 'deploy_mode': None, 'queue': None, 'namespace': None}
        assert connection == expected_spark_connection
        assert dict_cmd['--master'] == 'yarn'

    def test_resolve_connection_yarn_default_connection(self):
        if False:
            print('Hello World!')
        hook = SparkSubmitHook(conn_id='spark_default')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        dict_cmd = self.cmd_args_to_dict(cmd)
        expected_spark_connection = {'master': 'yarn', 'spark_binary': 'spark-submit', 'deploy_mode': None, 'queue': 'root.default', 'namespace': None}
        assert connection == expected_spark_connection
        assert dict_cmd['--master'] == 'yarn'
        assert dict_cmd['--queue'] == 'root.default'

    def test_resolve_connection_mesos_default_connection(self):
        if False:
            return 10
        hook = SparkSubmitHook(conn_id='spark_default_mesos')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        dict_cmd = self.cmd_args_to_dict(cmd)
        expected_spark_connection = {'master': 'mesos://host:5050', 'spark_binary': 'spark-submit', 'deploy_mode': None, 'queue': None, 'namespace': None}
        assert connection == expected_spark_connection
        assert dict_cmd['--master'] == 'mesos://host:5050'

    def test_resolve_connection_spark_yarn_cluster_connection(self):
        if False:
            while True:
                i = 10
        hook = SparkSubmitHook(conn_id='spark_yarn_cluster')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        dict_cmd = self.cmd_args_to_dict(cmd)
        expected_spark_connection = {'master': 'yarn://yarn-master', 'spark_binary': 'spark-submit', 'deploy_mode': 'cluster', 'queue': 'root.etl', 'namespace': None}
        assert connection == expected_spark_connection
        assert dict_cmd['--master'] == 'yarn://yarn-master'
        assert dict_cmd['--queue'] == 'root.etl'
        assert dict_cmd['--deploy-mode'] == 'cluster'

    def test_resolve_connection_spark_k8s_cluster_connection(self):
        if False:
            print('Hello World!')
        hook = SparkSubmitHook(conn_id='spark_k8s_cluster')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        dict_cmd = self.cmd_args_to_dict(cmd)
        expected_spark_connection = {'queue': None, 'spark_binary': 'spark-submit', 'master': 'k8s://https://k8s-master', 'deploy_mode': 'cluster', 'namespace': 'mynamespace'}
        assert connection == expected_spark_connection
        assert dict_cmd['--master'] == 'k8s://https://k8s-master'
        assert dict_cmd['--deploy-mode'] == 'cluster'

    def test_resolve_connection_spark_k8s_cluster_ns_conf(self):
        if False:
            for i in range(10):
                print('nop')
        conf = {'spark.kubernetes.namespace': 'airflow'}
        hook = SparkSubmitHook(conn_id='spark_k8s_cluster', conf=conf)
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        dict_cmd = self.cmd_args_to_dict(cmd)
        expected_spark_connection = {'queue': None, 'spark_binary': 'spark-submit', 'master': 'k8s://https://k8s-master', 'deploy_mode': 'cluster', 'namespace': 'airflow'}
        assert connection == expected_spark_connection
        assert dict_cmd['--master'] == 'k8s://https://k8s-master'
        assert dict_cmd['--deploy-mode'] == 'cluster'
        assert dict_cmd['--conf'] == 'spark.kubernetes.namespace=airflow'

    def test_resolve_connection_spark_binary_set_connection(self):
        if False:
            for i in range(10):
                print('nop')
        hook = SparkSubmitHook(conn_id='spark_binary_set')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        expected_spark_connection = {'master': 'yarn', 'spark_binary': 'spark2-submit', 'deploy_mode': None, 'queue': None, 'namespace': None}
        assert connection == expected_spark_connection
        assert cmd[0] == 'spark2-submit'

    def test_resolve_connection_spark_binary_spark3_submit_set_connection(self):
        if False:
            return 10
        hook = SparkSubmitHook(conn_id='spark_binary_set_spark3_submit')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        expected_spark_connection = {'master': 'yarn', 'spark_binary': 'spark3-submit', 'deploy_mode': None, 'queue': None, 'namespace': None}
        assert connection == expected_spark_connection
        assert cmd[0] == 'spark3-submit'

    def test_resolve_connection_custom_spark_binary_allowed_in_hook(self):
        if False:
            for i in range(10):
                print('nop')
        SparkSubmitHook(conn_id='spark_binary_set', spark_binary='another-custom-spark-submit')

    def test_resolve_connection_spark_binary_extra_not_allowed_runtime_error(self):
        if False:
            print('Hello World!')
        with pytest.raises(RuntimeError):
            SparkSubmitHook(conn_id='spark_custom_binary_set')

    def test_resolve_connection_spark_home_not_allowed_runtime_error(self):
        if False:
            return 10
        with pytest.raises(RuntimeError):
            SparkSubmitHook(conn_id='spark_home_set')

    def test_resolve_connection_spark_binary_default_value_override(self):
        if False:
            print('Hello World!')
        hook = SparkSubmitHook(conn_id='spark_binary_set', spark_binary='spark3-submit')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        expected_spark_connection = {'master': 'yarn', 'spark_binary': 'spark3-submit', 'deploy_mode': None, 'queue': None, 'namespace': None}
        assert connection == expected_spark_connection
        assert cmd[0] == 'spark3-submit'

    def test_resolve_connection_spark_binary_default_value(self):
        if False:
            print('Hello World!')
        hook = SparkSubmitHook(conn_id='spark_default')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        expected_spark_connection = {'master': 'yarn', 'spark_binary': 'spark-submit', 'deploy_mode': None, 'queue': 'root.default', 'namespace': None}
        assert connection == expected_spark_connection
        assert cmd[0] == 'spark-submit'

    def test_resolve_connection_spark_standalone_cluster_connection(self):
        if False:
            return 10
        hook = SparkSubmitHook(conn_id='spark_standalone_cluster')
        connection = hook._resolve_connection()
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        expected_spark_connection = {'master': 'spark://spark-standalone-master:6066', 'spark_binary': 'spark-submit', 'deploy_mode': 'cluster', 'queue': None, 'namespace': None}
        assert connection == expected_spark_connection
        assert cmd[0] == 'spark-submit'

    def test_resolve_spark_submit_env_vars_standalone_client_mode(self):
        if False:
            i = 10
            return i + 15
        hook = SparkSubmitHook(conn_id='spark_standalone_cluster_client_mode', env_vars={'bar': 'foo'})
        hook._build_spark_submit_command(self._spark_job_file)
        assert hook._env == {'bar': 'foo'}

    def test_resolve_spark_submit_env_vars_standalone_cluster_mode(self):
        if False:
            return 10

        def env_vars_exception_in_standalone_cluster_mode():
            if False:
                i = 10
                return i + 15
            hook = SparkSubmitHook(conn_id='spark_standalone_cluster', env_vars={'bar': 'foo'})
            hook._build_spark_submit_command(self._spark_job_file)
        with pytest.raises(AirflowException):
            env_vars_exception_in_standalone_cluster_mode()

    def test_resolve_spark_submit_env_vars_yarn(self):
        if False:
            i = 10
            return i + 15
        hook = SparkSubmitHook(conn_id='spark_yarn_cluster', env_vars={'bar': 'foo'})
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        assert cmd[4] == 'spark.yarn.appMasterEnv.bar=foo'
        assert hook._env == {'bar': 'foo'}

    def test_resolve_spark_submit_env_vars_k8s(self):
        if False:
            while True:
                i = 10
        hook = SparkSubmitHook(conn_id='spark_k8s_cluster', env_vars={'bar': 'foo'})
        cmd = hook._build_spark_submit_command(self._spark_job_file)
        assert cmd[4] == 'spark.kubernetes.driverEnv.bar=foo'

    def test_process_spark_submit_log_yarn(self):
        if False:
            print('Hello World!')
        hook = SparkSubmitHook(conn_id='spark_yarn_cluster')
        log_lines = ['SPARK_MAJOR_VERSION is set to 2, using Spark2', 'WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable', 'WARN DomainSocketFactory: The short-circuit local reads feature cannot be used because libhadoop cannot be loaded.', 'INFO Client: Requesting a new application from cluster with 10 NodeManagers', 'INFO Client: Submitting application application_1486558679801_1820 to ResourceManager']
        hook._process_spark_submit_log(log_lines)
        assert hook._yarn_application_id == 'application_1486558679801_1820'

    def test_process_spark_submit_log_k8s(self):
        if False:
            for i in range(10):
                print('nop')
        hook = SparkSubmitHook(conn_id='spark_k8s_cluster')
        log_lines = ['INFO  LoggingPodStatusWatcherImpl:54 - State changed, new state:pod name: spark-pi-edf2ace37be7353a958b38733a12f8e6-drivernamespace: defaultlabels: spark-app-selector -> spark-465b868ada474bda82ccb84ab2747fcd,spark-role -> driverpod uid: ba9c61f6-205f-11e8-b65f-d48564c88e42creation time: 2018-03-05T10:26:55Zservice account name: sparkvolumes: spark-init-properties, download-jars-volume,download-files-volume, spark-token-2vmlmnode name: N/Astart time: N/Acontainer images: N/Aphase: Pendingstatus: []2018-03-05 11:26:56 INFO  LoggingPodStatusWatcherImpl:54 - State changed, new state:pod name: spark-pi-edf2ace37be7353a958b38733a12f8e6-drivernamespace: defaultExit code: 999']
        hook._process_spark_submit_log(log_lines)
        assert hook._kubernetes_driver_pod == 'spark-pi-edf2ace37be7353a958b38733a12f8e6-driver'
        assert hook._spark_exit_code == 999

    def test_process_spark_submit_log_k8s_spark_3(self):
        if False:
            print('Hello World!')
        hook = SparkSubmitHook(conn_id='spark_k8s_cluster')
        log_lines = ['exit code: 999']
        hook._process_spark_submit_log(log_lines)
        assert hook._spark_exit_code == 999

    def test_process_spark_submit_log_standalone_cluster(self):
        if False:
            print('Hello World!')
        hook = SparkSubmitHook(conn_id='spark_standalone_cluster')
        log_lines = ['Running Spark using the REST application submission protocol.', '17/11/28 11:14:15 INFO RestSubmissionClient: Submitting a request to launch an application in spark://spark-standalone-master:6066', '17/11/28 11:14:15 INFO RestSubmissionClient: Submission successfully created as driver-20171128111415-0001. Polling submission state...']
        hook._process_spark_submit_log(log_lines)
        assert hook._driver_id == 'driver-20171128111415-0001'

    def test_process_spark_driver_status_log(self):
        if False:
            i = 10
            return i + 15
        hook = SparkSubmitHook(conn_id='spark_standalone_cluster')
        log_lines = ['Submitting a request for the status of submission driver-20171128111415-0001 in spark://spark-standalone-master:6066', '17/11/28 11:15:37 INFO RestSubmissionClient: Server responded with SubmissionStatusResponse:', '{', '"action" : "SubmissionStatusResponse",', '"driverState" : "RUNNING",', '"serverSparkVersion" : "1.6.0",', '"submissionId" : "driver-20171128111415-0001",', '"success" : true,', '"workerHostPort" : "172.18.0.7:38561",', '"workerId" : "worker-20171128110741-172.18.0.7-38561"', '}']
        hook._process_spark_status_log(log_lines)
        assert hook._driver_status == 'RUNNING'

    def test_process_spark_driver_status_log_bad_response(self):
        if False:
            return 10
        hook = SparkSubmitHook(conn_id='spark_standalone_cluster')
        log_lines = ['curl: Failed to connect to http://spark-standalone-master:6066This is an invalid Spark response', 'Timed out']
        hook._process_spark_status_log(log_lines)
        assert hook._driver_status is None

    @patch('airflow.providers.apache.spark.hooks.spark_submit.renew_from_kt')
    @patch('airflow.providers.apache.spark.hooks.spark_submit.subprocess.Popen')
    def test_yarn_process_on_kill(self, mock_popen, mock_renew_from_kt):
        if False:
            i = 10
            return i + 15
        mock_popen.return_value.stdout = StringIO('stdout')
        mock_popen.return_value.stderr = StringIO('stderr')
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.wait.return_value = 0
        log_lines = ['SPARK_MAJOR_VERSION is set to 2, using Spark2', 'WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable', 'WARN DomainSocketFactory: The short-circuit local reads feature cannot be used because libhadoop cannot be loaded.', 'INFO Client: Requesting a new application from cluster with 10 NodeManagerapplication_1486558679801_1820s', 'INFO Client: Submitting application application_1486558679801_1820 to ResourceManager']
        env = {'PATH': 'hadoop/bin'}
        hook = SparkSubmitHook(conn_id='spark_yarn_cluster', env_vars=env)
        hook._process_spark_submit_log(log_lines)
        hook.submit()
        hook.on_kill()
        assert call(['yarn', 'application', '-kill', 'application_1486558679801_1820'], env={**os.environ, **env}, stderr=-1, stdout=-1) in mock_popen.mock_calls
        mock_popen.reset_mock()
        hook = SparkSubmitHook(conn_id='spark_yarn_cluster', keytab='privileged_user.keytab', principal='user/spark@airflow.org')
        hook._process_spark_submit_log(log_lines)
        hook.submit()
        hook.on_kill()
        expected_env = os.environ.copy()
        expected_env['KRB5CCNAME'] = '/tmp/airflow_krb5_ccache'
        assert call(['yarn', 'application', '-kill', 'application_1486558679801_1820'], env=expected_env, stderr=-1, stdout=-1) in mock_popen.mock_calls

    def test_standalone_cluster_process_on_kill(self):
        if False:
            print('Hello World!')
        log_lines = ['Running Spark using the REST application submission protocol.', '17/11/28 11:14:15 INFO RestSubmissionClient: Submitting a request to launch an application in spark://spark-standalone-master:6066', '17/11/28 11:14:15 INFO RestSubmissionClient: Submission successfully created as driver-20171128111415-0001. Polling submission state...']
        hook = SparkSubmitHook(conn_id='spark_standalone_cluster')
        hook._process_spark_submit_log(log_lines)
        kill_cmd = hook._build_spark_driver_kill_command()
        assert kill_cmd[0] == 'spark-submit'
        assert kill_cmd[1] == '--master'
        assert kill_cmd[2] == 'spark://spark-standalone-master:6066'
        assert kill_cmd[3] == '--kill'
        assert kill_cmd[4] == 'driver-20171128111415-0001'

    @patch('airflow.providers.cncf.kubernetes.kube_client.get_kube_client')
    @patch('airflow.providers.apache.spark.hooks.spark_submit.subprocess.Popen')
    def test_k8s_process_on_kill(self, mock_popen, mock_client_method):
        if False:
            return 10
        mock_popen.return_value.stdout = StringIO('stdout')
        mock_popen.return_value.stderr = StringIO('stderr')
        mock_popen.return_value.poll.return_value = None
        mock_popen.return_value.wait.return_value = 0
        client = mock_client_method.return_value
        hook = SparkSubmitHook(conn_id='spark_k8s_cluster')
        log_lines = ['INFO  LoggingPodStatusWatcherImpl:54 - State changed, new state:pod name: spark-pi-edf2ace37be7353a958b38733a12f8e6-drivernamespace: defaultlabels: spark-app-selector -> spark-465b868ada474bda82ccb84ab2747fcd,spark-role -> driverpod uid: ba9c61f6-205f-11e8-b65f-d48564c88e42creation time: 2018-03-05T10:26:55Zservice account name: sparkvolumes: spark-init-properties, download-jars-volume,download-files-volume, spark-token-2vmlmnode name: N/Astart time: N/Acontainer images: N/Aphase: Pendingstatus: []2018-03-05 11:26:56 INFO  LoggingPodStatusWatcherImpl:54 - State changed, new state:pod name: spark-pi-edf2ace37be7353a958b38733a12f8e6-drivernamespace: defaultExit code: 0']
        hook._process_spark_submit_log(log_lines)
        hook.submit()
        hook.on_kill()
        import kubernetes
        kwargs = {'pretty': True, 'body': kubernetes.client.V1DeleteOptions()}
        client.delete_namespaced_pod.assert_called_once_with('spark-pi-edf2ace37be7353a958b38733a12f8e6-driver', 'mynamespace', **kwargs)

    @pytest.mark.parametrize('command, expected', [(('spark-submit', 'foo', '--bar', 'baz', "--password='secret'", '--foo', 'bar'), "spark-submit foo --bar baz --password='******' --foo bar"), (('spark-submit', 'foo', '--bar', 'baz', "--password='secret'"), "spark-submit foo --bar baz --password='******'"), (('spark-submit', 'foo', '--bar', 'baz', '--password="secret"'), 'spark-submit foo --bar baz --password="******"'), (('spark-submit', 'foo', '--bar', 'baz', '--password=secret'), 'spark-submit foo --bar baz --password=******'), (('spark-submit', 'foo', '--bar', 'baz', "--password 'secret'"), "spark-submit foo --bar baz --password '******'"), (('spark-submit', 'foo', '--bar', 'baz', '--password=\'sec"ret\''), "spark-submit foo --bar baz --password='******'"), (('spark-submit', 'foo', '--bar', 'baz', '--password="sec\'ret"'), 'spark-submit foo --bar baz --password="******"'), (('spark-submit',), 'spark-submit')])
    def test_masks_passwords(self, command: str, expected: str) -> None:
        if False:
            return 10
        hook = SparkSubmitHook()
        command_masked = hook._mask_cmd(command)
        assert command_masked == expected