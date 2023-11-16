from __future__ import annotations
import contextlib
import os
import re
import subprocess
import time
from typing import Any, Iterator
from airflow.configuration import conf as airflow_conf
from airflow.exceptions import AirflowException
from airflow.hooks.base import BaseHook
from airflow.security.kerberos import renew_from_kt
from airflow.utils.log.logging_mixin import LoggingMixin
with contextlib.suppress(ImportError, NameError):
    from airflow.providers.cncf.kubernetes import kube_client
ALLOWED_SPARK_BINARIES = ['spark-submit', 'spark2-submit', 'spark3-submit']

class SparkSubmitHook(BaseHook, LoggingMixin):
    """
    Wrap the spark-submit binary to kick off a spark-submit job; requires "spark-submit" binary in the PATH.

    :param conf: Arbitrary Spark configuration properties
    :param spark_conn_id: The :ref:`spark connection id <howto/connection:spark>` as configured
        in Airflow administration. When an invalid connection_id is supplied, it will default
        to yarn.
    :param files: Upload additional files to the executor running the job, separated by a
        comma. Files will be placed in the working directory of each executor.
        For example, serialized objects.
    :param py_files: Additional python files used by the job, can be .zip, .egg or .py.
    :param archives: Archives that spark should unzip (and possibly tag with #ALIAS) into
        the application working directory.
    :param driver_class_path: Additional, driver-specific, classpath settings.
    :param jars: Submit additional jars to upload and place them in executor classpath.
    :param java_class: the main class of the Java application
    :param packages: Comma-separated list of maven coordinates of jars to include on the
        driver and executor classpaths
    :param exclude_packages: Comma-separated list of maven coordinates of jars to exclude
        while resolving the dependencies provided in 'packages'
    :param repositories: Comma-separated list of additional remote repositories to search
        for the maven coordinates given with 'packages'
    :param total_executor_cores: (Standalone & Mesos only) Total cores for all executors
        (Default: all the available cores on the worker)
    :param executor_cores: (Standalone, YARN and Kubernetes only) Number of cores per
        executor (Default: 2)
    :param executor_memory: Memory per executor (e.g. 1000M, 2G) (Default: 1G)
    :param driver_memory: Memory allocated to the driver (e.g. 1000M, 2G) (Default: 1G)
    :param keytab: Full path to the file that contains the keytab
    :param principal: The name of the kerberos principal used for keytab
    :param proxy_user: User to impersonate when submitting the application
    :param name: Name of the job (default airflow-spark)
    :param num_executors: Number of executors to launch
    :param status_poll_interval: Seconds to wait between polls of driver status in cluster
        mode (Default: 1)
    :param application_args: Arguments for the application being submitted
    :param env_vars: Environment variables for spark-submit. It
        supports yarn and k8s mode too.
    :param verbose: Whether to pass the verbose flag to spark-submit process for debugging
    :param spark_binary: The command to use for spark submit.
                         Some distros may use spark2-submit or spark3-submit.
    :param use_krb5ccache: if True, configure spark to use ticket cache instead of relying
        on keytab for Kerberos login
    """
    conn_name_attr = 'conn_id'
    default_conn_name = 'spark_default'
    conn_type = 'spark'
    hook_name = 'Spark'

    @staticmethod
    def get_ui_field_behaviour() -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return custom field behaviour.'
        return {'hidden_fields': ['schema', 'login', 'password'], 'relabeling': {}}

    def __init__(self, conf: dict[str, Any] | None=None, conn_id: str='spark_default', files: str | None=None, py_files: str | None=None, archives: str | None=None, driver_class_path: str | None=None, jars: str | None=None, java_class: str | None=None, packages: str | None=None, exclude_packages: str | None=None, repositories: str | None=None, total_executor_cores: int | None=None, executor_cores: int | None=None, executor_memory: str | None=None, driver_memory: str | None=None, keytab: str | None=None, principal: str | None=None, proxy_user: str | None=None, name: str='default-name', num_executors: int | None=None, status_poll_interval: int=1, application_args: list[Any] | None=None, env_vars: dict[str, Any] | None=None, verbose: bool=False, spark_binary: str | None=None, *, use_krb5ccache: bool=False) -> None:
        if False:
            return 10
        super().__init__()
        self._conf = conf or {}
        self._conn_id = conn_id
        self._files = files
        self._py_files = py_files
        self._archives = archives
        self._driver_class_path = driver_class_path
        self._jars = jars
        self._java_class = java_class
        self._packages = packages
        self._exclude_packages = exclude_packages
        self._repositories = repositories
        self._total_executor_cores = total_executor_cores
        self._executor_cores = executor_cores
        self._executor_memory = executor_memory
        self._driver_memory = driver_memory
        self._keytab = keytab
        self._principal = self._resolve_kerberos_principal(principal) if use_krb5ccache else principal
        self._use_krb5ccache = use_krb5ccache
        self._proxy_user = proxy_user
        self._name = name
        self._num_executors = num_executors
        self._status_poll_interval = status_poll_interval
        self._application_args = application_args
        self._env_vars = env_vars
        self._verbose = verbose
        self._submit_sp: Any | None = None
        self._yarn_application_id: str | None = None
        self._kubernetes_driver_pod: str | None = None
        self.spark_binary = spark_binary
        self._connection = self._resolve_connection()
        self._is_yarn = 'yarn' in self._connection['master']
        self._is_kubernetes = 'k8s' in self._connection['master']
        if self._is_kubernetes and kube_client is None:
            raise RuntimeError(f"{self._connection['master']} specified by kubernetes dependencies are not installed!")
        self._should_track_driver_status = self._resolve_should_track_driver_status()
        self._driver_id: str | None = None
        self._driver_status: str | None = None
        self._spark_exit_code: int | None = None
        self._env: dict[str, Any] | None = None

    def _resolve_should_track_driver_status(self) -> bool:
        if False:
            return 10
        'Check if we should track the driver status.\n\n        If so, we should send subsequent spark-submit status requests after the\n        initial spark-submit request.\n\n        :return: if the driver status should be tracked\n        '
        return 'spark://' in self._connection['master'] and self._connection['deploy_mode'] == 'cluster'

    def _resolve_connection(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        conn_data = {'master': 'yarn', 'queue': None, 'deploy_mode': None, 'spark_binary': self.spark_binary or 'spark-submit', 'namespace': None}
        try:
            conn = self.get_connection(self._conn_id)
            if conn.port:
                conn_data['master'] = f'{conn.host}:{conn.port}'
            else:
                conn_data['master'] = conn.host
            extra = conn.extra_dejson
            conn_data['queue'] = extra.get('queue')
            conn_data['deploy_mode'] = extra.get('deploy-mode')
            if not self.spark_binary:
                self.spark_binary = extra.get('spark-binary', 'spark-submit')
                if self.spark_binary is not None and self.spark_binary not in ALLOWED_SPARK_BINARIES:
                    raise RuntimeError(f'The spark-binary extra can be on of {ALLOWED_SPARK_BINARIES} and it was `{self.spark_binary}`. Please make sure your spark binary is one of the allowed ones and that it is available on the PATH')
            conn_spark_home = extra.get('spark-home')
            if conn_spark_home:
                raise RuntimeError(f'The `spark-home` extra is not allowed any more. Please make sure one of {ALLOWED_SPARK_BINARIES} is available on the PATH, and set `spark-binary` if needed.')
            conn_data['spark_binary'] = self.spark_binary
            conn_data['namespace'] = extra.get('namespace')
        except AirflowException:
            self.log.info('Could not load connection string %s, defaulting to %s', self._conn_id, conn_data['master'])
        if 'spark.kubernetes.namespace' in self._conf:
            conn_data['namespace'] = self._conf['spark.kubernetes.namespace']
        return conn_data

    def get_conn(self) -> Any:
        if False:
            return 10
        pass

    def _get_spark_binary_path(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        return [self._connection['spark_binary']]

    def _mask_cmd(self, connection_cmd: str | list[str]) -> str:
        if False:
            for i in range(10):
                print('nop')
        connection_cmd_masked = re.sub('(\\S*?(?:secret|password)\\S*?(?:=|\\s+)([\'\\"]?))(?:(?!\\2\\s).)*(\\2)', '\\1******\\3', ' '.join(connection_cmd), flags=re.I)
        return connection_cmd_masked

    def _build_spark_submit_command(self, application: str) -> list[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct the spark-submit command to execute.\n\n        :param application: command to append to the spark-submit command\n        :return: full command to be executed\n        '
        connection_cmd = self._get_spark_binary_path()
        connection_cmd += ['--master', self._connection['master']]
        for key in self._conf:
            connection_cmd += ['--conf', f'{key}={self._conf[key]}']
        if self._env_vars and (self._is_kubernetes or self._is_yarn):
            if self._is_yarn:
                tmpl = 'spark.yarn.appMasterEnv.{}={}'
                self._env = self._env_vars
            else:
                tmpl = 'spark.kubernetes.driverEnv.{}={}'
            for key in self._env_vars:
                connection_cmd += ['--conf', tmpl.format(key, str(self._env_vars[key]))]
        elif self._env_vars and self._connection['deploy_mode'] != 'cluster':
            self._env = self._env_vars
        elif self._env_vars and self._connection['deploy_mode'] == 'cluster':
            raise AirflowException('SparkSubmitHook env_vars is not supported in standalone-cluster mode.')
        if self._is_kubernetes and self._connection['namespace']:
            connection_cmd += ['--conf', f"spark.kubernetes.namespace={self._connection['namespace']}"]
        if self._files:
            connection_cmd += ['--files', self._files]
        if self._py_files:
            connection_cmd += ['--py-files', self._py_files]
        if self._archives:
            connection_cmd += ['--archives', self._archives]
        if self._driver_class_path:
            connection_cmd += ['--driver-class-path', self._driver_class_path]
        if self._jars:
            connection_cmd += ['--jars', self._jars]
        if self._packages:
            connection_cmd += ['--packages', self._packages]
        if self._exclude_packages:
            connection_cmd += ['--exclude-packages', self._exclude_packages]
        if self._repositories:
            connection_cmd += ['--repositories', self._repositories]
        if self._num_executors:
            connection_cmd += ['--num-executors', str(self._num_executors)]
        if self._total_executor_cores:
            connection_cmd += ['--total-executor-cores', str(self._total_executor_cores)]
        if self._executor_cores:
            connection_cmd += ['--executor-cores', str(self._executor_cores)]
        if self._executor_memory:
            connection_cmd += ['--executor-memory', self._executor_memory]
        if self._driver_memory:
            connection_cmd += ['--driver-memory', self._driver_memory]
        if self._keytab:
            connection_cmd += ['--keytab', self._keytab]
        if self._principal:
            connection_cmd += ['--principal', self._principal]
        if self._use_krb5ccache:
            if not os.getenv('KRB5CCNAME'):
                raise AirflowException('KRB5CCNAME environment variable required to use ticket ccache is missing.')
            connection_cmd += ['--conf', 'spark.kerberos.renewal.credentials=ccache']
        if self._proxy_user:
            connection_cmd += ['--proxy-user', self._proxy_user]
        if self._name:
            connection_cmd += ['--name', self._name]
        if self._java_class:
            connection_cmd += ['--class', self._java_class]
        if self._verbose:
            connection_cmd += ['--verbose']
        if self._connection['queue']:
            connection_cmd += ['--queue', self._connection['queue']]
        if self._connection['deploy_mode']:
            connection_cmd += ['--deploy-mode', self._connection['deploy_mode']]
        connection_cmd += [application]
        if self._application_args:
            connection_cmd += self._application_args
        self.log.info('Spark-Submit cmd: %s', self._mask_cmd(connection_cmd))
        return connection_cmd

    def _build_track_driver_status_command(self) -> list[str]:
        if False:
            while True:
                i = 10
        '\n        Construct the command to poll the driver status.\n\n        :return: full command to be executed\n        '
        curl_max_wait_time = 30
        spark_host = self._connection['master']
        if spark_host.endswith(':6066'):
            spark_host = spark_host.replace('spark://', 'http://')
            connection_cmd = ['/usr/bin/curl', '--max-time', str(curl_max_wait_time), f'{spark_host}/v1/submissions/status/{self._driver_id}']
            self.log.info(connection_cmd)
            if not self._driver_id:
                raise AirflowException('Invalid status: attempted to poll driver status but no driver id is known. Giving up.')
        else:
            connection_cmd = self._get_spark_binary_path()
            connection_cmd += ['--master', self._connection['master']]
            if self._driver_id:
                connection_cmd += ['--status', self._driver_id]
            else:
                raise AirflowException('Invalid status: attempted to poll driver status but no driver id is known. Giving up.')
        self.log.debug('Poll driver status cmd: %s', connection_cmd)
        return connection_cmd

    def _resolve_kerberos_principal(self, principal: str | None) -> str:
        if False:
            while True:
                i = 10
        'Resolve kerberos principal if airflow > 2.8.\n\n        TODO: delete when min airflow version >= 2.8 and import directly from airflow.security.kerberos\n        '
        from packaging.version import Version
        from airflow.version import version
        if Version(version) < Version('2.8'):
            from airflow.utils.net import get_hostname
            return principal or airflow_conf.get_mandatory_value('kerberos', 'principal').replace('_HOST', get_hostname())
        else:
            from airflow.security.kerberos import get_kerberos_principle
            return get_kerberos_principle(principal)

    def submit(self, application: str='', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Remote Popen to execute the spark-submit job.\n\n        :param application: Submitted application, jar or py file\n        :param kwargs: extra arguments to Popen (see subprocess.Popen)\n        '
        spark_submit_cmd = self._build_spark_submit_command(application)
        if self._env:
            env = os.environ.copy()
            env.update(self._env)
            kwargs['env'] = env
        self._submit_sp = subprocess.Popen(spark_submit_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=-1, universal_newlines=True, **kwargs)
        self._process_spark_submit_log(iter(self._submit_sp.stdout))
        returncode = self._submit_sp.wait()
        if returncode or (self._is_kubernetes and self._spark_exit_code != 0):
            if self._is_kubernetes:
                raise AirflowException(f'Cannot execute: {self._mask_cmd(spark_submit_cmd)}. Error code is: {returncode}. Kubernetes spark exit code is: {self._spark_exit_code}')
            else:
                raise AirflowException(f'Cannot execute: {self._mask_cmd(spark_submit_cmd)}. Error code is: {returncode}.')
        self.log.debug('Should track driver: %s', self._should_track_driver_status)
        if self._should_track_driver_status:
            if self._driver_id is None:
                raise AirflowException('No driver id is known: something went wrong when executing the spark submit command')
            self._driver_status = 'SUBMITTED'
            self._start_driver_status_tracking()
            if self._driver_status != 'FINISHED':
                raise AirflowException(f'ERROR : Driver {self._driver_id} badly exited with status {self._driver_status}')

    def _process_spark_submit_log(self, itr: Iterator[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Process the log files and extract useful information out of it.\n\n        If the deploy-mode is 'client', log the output of the submit command as those\n        are the output logs of the Spark worker directly.\n\n        Remark: If the driver needs to be tracked for its status, the log-level of the\n        spark deploy needs to be at least INFO (log4j.logger.org.apache.spark.deploy=INFO)\n\n        :param itr: An iterator which iterates over the input of the subprocess\n        "
        for line in itr:
            line = line.strip()
            if self._is_yarn and self._connection['deploy_mode'] == 'cluster':
                match = re.search('application[0-9_]+', line)
                if match:
                    self._yarn_application_id = match.group(0)
                    self.log.info('Identified spark driver id: %s', self._yarn_application_id)
            elif self._is_kubernetes:
                match = re.search('\\s*pod name: ((.+?)-([a-z0-9]+)-driver)', line)
                if match:
                    self._kubernetes_driver_pod = match.group(1)
                    self.log.info('Identified spark driver pod: %s', self._kubernetes_driver_pod)
                match_exit_code = re.search('\\s*[eE]xit code: (\\d+)', line)
                if match_exit_code:
                    self._spark_exit_code = int(match_exit_code.group(1))
            elif self._should_track_driver_status and (not self._driver_id):
                match_driver_id = re.search('driver-[0-9\\-]+', line)
                if match_driver_id:
                    self._driver_id = match_driver_id.group(0)
                    self.log.info('identified spark driver id: %s', self._driver_id)
            self.log.info(line)

    def _process_spark_status_log(self, itr: Iterator[Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Parse the logs of the spark driver status query process.\n\n        :param itr: An iterator which iterates over the input of the subprocess\n        '
        driver_found = False
        valid_response = False
        for line in itr:
            line = line.strip()
            if 'submissionId' in line:
                valid_response = True
            if 'driverState' in line:
                self._driver_status = line.split(' : ')[1].replace(',', '').replace('"', '').strip()
                driver_found = True
            self.log.debug('spark driver status log: %s', line)
        if valid_response and (not driver_found):
            self._driver_status = 'UNKNOWN'

    def _start_driver_status_tracking(self) -> None:
        if False:
            return 10
        '\n        Poll the driver based on self._driver_id to get the status.\n\n        Finish successfully when the status is FINISHED.\n        Finish failed when the status is ERROR/UNKNOWN/KILLED/FAILED.\n\n        Possible status:\n\n        SUBMITTED\n            Submitted but not yet scheduled on a worker\n        RUNNING\n            Has been allocated to a worker to run\n        FINISHED\n            Previously ran and exited cleanly\n        RELAUNCHING\n            Exited non-zero or due to worker failure, but has not yet\n            started running again\n        UNKNOWN\n            The status of the driver is temporarily not known due to\n            master failure recovery\n        KILLED\n            A user manually killed this driver\n        FAILED\n            The driver exited non-zero and was not supervised\n        ERROR\n            Unable to run or restart due to an unrecoverable error\n            (e.g. missing jar file)\n        '
        missed_job_status_reports = 0
        max_missed_job_status_reports = 10
        while self._driver_status not in ['FINISHED', 'UNKNOWN', 'KILLED', 'FAILED', 'ERROR']:
            time.sleep(self._status_poll_interval)
            self.log.debug('polling status of spark driver with id %s', self._driver_id)
            poll_drive_status_cmd = self._build_track_driver_status_command()
            status_process: Any = subprocess.Popen(poll_drive_status_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=-1, universal_newlines=True)
            self._process_spark_status_log(iter(status_process.stdout))
            returncode = status_process.wait()
            if returncode:
                if missed_job_status_reports < max_missed_job_status_reports:
                    missed_job_status_reports += 1
                else:
                    raise AirflowException(f'Failed to poll for the driver status {max_missed_job_status_reports} times: returncode = {returncode}')

    def _build_spark_driver_kill_command(self) -> list[str]:
        if False:
            print('Hello World!')
        '\n        Construct the spark-submit command to kill a driver.\n\n        :return: full command to kill a driver\n        '
        connection_cmd = [self._connection['spark_binary']]
        connection_cmd += ['--master', self._connection['master']]
        if self._driver_id:
            connection_cmd += ['--kill', self._driver_id]
        self.log.debug('Spark-Kill cmd: %s', connection_cmd)
        return connection_cmd

    def on_kill(self) -> None:
        if False:
            return 10
        'Kill Spark submit command.'
        self.log.debug('Kill Command is being called')
        if self._should_track_driver_status and self._driver_id:
            self.log.info('Killing driver %s on cluster', self._driver_id)
            kill_cmd = self._build_spark_driver_kill_command()
            with subprocess.Popen(kill_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as driver_kill:
                self.log.info('Spark driver %s killed with return code: %s', self._driver_id, driver_kill.wait())
        if self._submit_sp and self._submit_sp.poll() is None:
            self.log.info('Sending kill signal to %s', self._connection['spark_binary'])
            self._submit_sp.kill()
            if self._yarn_application_id:
                kill_cmd = f'yarn application -kill {self._yarn_application_id}'.split()
                env = {**os.environ, **(self._env or {})}
                if self._keytab is not None and self._principal is not None:
                    renew_from_kt(self._principal, self._keytab, exit_on_fail=False)
                    env = os.environ.copy()
                    ccacche = airflow_conf.get_mandatory_value('kerberos', 'ccache')
                    env['KRB5CCNAME'] = ccacche
                with subprocess.Popen(kill_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as yarn_kill:
                    self.log.info('YARN app killed with return code: %s', yarn_kill.wait())
            if self._kubernetes_driver_pod:
                self.log.info('Killing pod %s on Kubernetes', self._kubernetes_driver_pod)
                try:
                    import kubernetes
                    client = kube_client.get_kube_client()
                    api_response = client.delete_namespaced_pod(self._kubernetes_driver_pod, self._connection['namespace'], body=kubernetes.client.V1DeleteOptions(), pretty=True)
                    self.log.info('Spark on K8s killed with response: %s', api_response)
                except kube_client.ApiException:
                    self.log.exception('Exception when attempting to kill Spark on K8s')