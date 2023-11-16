import collections
import logging
import os
import re
import sys
import tempfile
import shutil
import importlib
import tarfile
import inspect
import pickle
from luigi.contrib.external_program import ExternalProgramTask
from luigi import configuration
logger = logging.getLogger('luigi-interface')

class SparkSubmitTask(ExternalProgramTask):
    """
    Template task for running a Spark job

    Supports running jobs on Spark local, standalone, Mesos or Yarn

    See http://spark.apache.org/docs/latest/submitting-applications.html
    for more information

    """
    name = None
    entry_class = None
    app = None
    always_log_stderr = False
    stream_for_searching_tracking_url = 'stderr'

    @property
    def tracking_url_pattern(self):
        if False:
            return 10
        if self.deploy_mode == 'cluster':
            return 'tracking URL: (https?://.*)\\s'
        else:
            return 'Bound (?:.*) to (?:.*), and started at (https?://.*)\\s'

    def app_options(self):
        if False:
            return 10
        "\n        Subclass this method to map your task parameters to the app's arguments\n\n        "
        return []

    @property
    def pyspark_python(self):
        if False:
            while True:
                i = 10
        return None

    @property
    def pyspark_driver_python(self):
        if False:
            i = 10
            return i + 15
        return None

    @property
    def hadoop_user_name(self):
        if False:
            while True:
                i = 10
        return None

    @property
    def spark_version(self):
        if False:
            i = 10
            return i + 15
        return 'spark'

    @property
    def spark_submit(self):
        if False:
            i = 10
            return i + 15
        return configuration.get_config().get(self.spark_version, 'spark-submit', 'spark-submit')

    @property
    def master(self):
        if False:
            while True:
                i = 10
        return configuration.get_config().get(self.spark_version, 'master', None)

    @property
    def deploy_mode(self):
        if False:
            return 10
        return configuration.get_config().get(self.spark_version, 'deploy-mode', None)

    @property
    def jars(self):
        if False:
            for i in range(10):
                print('nop')
        return self._list_config(configuration.get_config().get(self.spark_version, 'jars', None))

    @property
    def packages(self):
        if False:
            print('Hello World!')
        return self._list_config(configuration.get_config().get(self.spark_version, 'packages', None))

    @property
    def py_files(self):
        if False:
            i = 10
            return i + 15
        return self._list_config(configuration.get_config().get(self.spark_version, 'py-files', None))

    @property
    def files(self):
        if False:
            while True:
                i = 10
        return self._list_config(configuration.get_config().get(self.spark_version, 'files', None))

    @property
    def _conf(self):
        if False:
            return 10
        conf = collections.OrderedDict(self.conf or {})
        if self.pyspark_python:
            conf['spark.pyspark.python'] = self.pyspark_python
        if self.pyspark_driver_python:
            conf['spark.pyspark.driver.python'] = self.pyspark_driver_python
        return conf

    @property
    def conf(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dict_config(configuration.get_config().get(self.spark_version, 'conf', None))

    @property
    def properties_file(self):
        if False:
            i = 10
            return i + 15
        return configuration.get_config().get(self.spark_version, 'properties-file', None)

    @property
    def driver_memory(self):
        if False:
            print('Hello World!')
        return configuration.get_config().get(self.spark_version, 'driver-memory', None)

    @property
    def driver_java_options(self):
        if False:
            while True:
                i = 10
        return configuration.get_config().get(self.spark_version, 'driver-java-options', None)

    @property
    def driver_library_path(self):
        if False:
            i = 10
            return i + 15
        return configuration.get_config().get(self.spark_version, 'driver-library-path', None)

    @property
    def driver_class_path(self):
        if False:
            print('Hello World!')
        return configuration.get_config().get(self.spark_version, 'driver-class-path', None)

    @property
    def executor_memory(self):
        if False:
            for i in range(10):
                print('nop')
        return configuration.get_config().get(self.spark_version, 'executor-memory', None)

    @property
    def driver_cores(self):
        if False:
            i = 10
            return i + 15
        return configuration.get_config().get(self.spark_version, 'driver-cores', None)

    @property
    def supervise(self):
        if False:
            return 10
        return bool(configuration.get_config().get(self.spark_version, 'supervise', False))

    @property
    def total_executor_cores(self):
        if False:
            for i in range(10):
                print('nop')
        return configuration.get_config().get(self.spark_version, 'total-executor-cores', None)

    @property
    def executor_cores(self):
        if False:
            return 10
        return configuration.get_config().get(self.spark_version, 'executor-cores', None)

    @property
    def queue(self):
        if False:
            print('Hello World!')
        return configuration.get_config().get(self.spark_version, 'queue', None)

    @property
    def num_executors(self):
        if False:
            i = 10
            return i + 15
        return configuration.get_config().get(self.spark_version, 'num-executors', None)

    @property
    def archives(self):
        if False:
            i = 10
            return i + 15
        return self._list_config(configuration.get_config().get(self.spark_version, 'archives', None))

    @property
    def hadoop_conf_dir(self):
        if False:
            while True:
                i = 10
        return configuration.get_config().get(self.spark_version, 'hadoop-conf-dir', None)

    def get_environment(self):
        if False:
            return 10
        env = os.environ.copy()
        for prop in ('HADOOP_CONF_DIR', 'HADOOP_USER_NAME'):
            var = getattr(self, prop.lower(), None)
            if var:
                env[prop] = var
        return env

    def program_environment(self):
        if False:
            i = 10
            return i + 15
        return self.get_environment()

    def program_args(self):
        if False:
            print('Hello World!')
        return self.spark_command() + self.app_command()

    def spark_command(self):
        if False:
            for i in range(10):
                print('nop')
        command = [self.spark_submit]
        command += self._text_arg('--master', self.master)
        command += self._text_arg('--deploy-mode', self.deploy_mode)
        command += self._text_arg('--name', self.name)
        command += self._text_arg('--class', self.entry_class)
        command += self._list_arg('--jars', self.jars)
        command += self._list_arg('--packages', self.packages)
        command += self._list_arg('--py-files', self.py_files)
        command += self._list_arg('--files', self.files)
        command += self._list_arg('--archives', self.archives)
        command += self._dict_arg('--conf', self._conf)
        command += self._text_arg('--properties-file', self.properties_file)
        command += self._text_arg('--driver-memory', self.driver_memory)
        command += self._text_arg('--driver-java-options', self.driver_java_options)
        command += self._text_arg('--driver-library-path', self.driver_library_path)
        command += self._text_arg('--driver-class-path', self.driver_class_path)
        command += self._text_arg('--executor-memory', self.executor_memory)
        command += self._text_arg('--driver-cores', self.driver_cores)
        command += self._flag_arg('--supervise', self.supervise)
        command += self._text_arg('--total-executor-cores', self.total_executor_cores)
        command += self._text_arg('--executor-cores', self.executor_cores)
        command += self._text_arg('--queue', self.queue)
        command += self._text_arg('--num-executors', self.num_executors)
        return command

    def app_command(self):
        if False:
            return 10
        if not self.app:
            raise NotImplementedError('subclass should define an app (.jar or .py file)')
        return [self.app] + self.app_options()

    def _list_config(self, config):
        if False:
            while True:
                i = 10
        if config and isinstance(config, str):
            return list(map(lambda x: x.strip(), config.split(',')))

    def _dict_config(self, config):
        if False:
            i = 10
            return i + 15
        if config and isinstance(config, str):
            return dict(map(lambda i: i.split('=', 1), config.split('|')))

    def _text_arg(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        if value:
            return [name, value]
        return []

    def _list_arg(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        if value and isinstance(value, (list, tuple)):
            return [name, ','.join(value)]
        return []

    def _dict_arg(self, name, value):
        if False:
            for i in range(10):
                print('nop')
        command = []
        if value and isinstance(value, dict):
            for (prop, value) in value.items():
                command += [name, '{0}={1}'.format(prop, value)]
        return command

    def _flag_arg(self, name, value):
        if False:
            while True:
                i = 10
        if value:
            return [name]
        return []

class PySparkTask(SparkSubmitTask):
    """
    Template task for running an inline PySpark job

    Simply implement the ``main`` method in your subclass

    You can optionally define package names to be distributed to the cluster
    with ``py_packages`` (uses luigi's global py-packages configuration by default)

    """
    app = os.path.join(os.path.dirname(__file__), 'pyspark_runner.py')

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__.__name__

    @property
    def py_packages(self):
        if False:
            i = 10
            return i + 15
        packages = configuration.get_config().get('spark', 'py-packages', None)
        if packages:
            return map(lambda s: s.strip(), packages.split(','))

    @property
    def files(self):
        if False:
            return 10
        if self.deploy_mode == 'cluster':
            return [self.run_pickle]

    @property
    def pickle_protocol(self):
        if False:
            i = 10
            return i + 15
        return configuration.get_config().getint('spark', 'pickle-protocol', pickle.DEFAULT_PROTOCOL)

    def setup(self, conf):
        if False:
            i = 10
            return i + 15
        '\n        Called by the pyspark_runner with a SparkConf instance that will be used to instantiate the SparkContext\n\n        :param conf: SparkConf\n        '

    def setup_remote(self, sc):
        if False:
            print('Hello World!')
        self._setup_packages(sc)

    def main(self, sc, *args):
        if False:
            return 10
        '\n        Called by the pyspark_runner with a SparkContext and any arguments returned by ``app_options()``\n\n        :param sc: SparkContext\n        :param args: arguments list\n        '
        raise NotImplementedError('subclass should define a main method')

    def app_command(self):
        if False:
            for i in range(10):
                print('nop')
        if self.deploy_mode == 'cluster':
            pickle_loc = os.path.basename(self.run_pickle)
        else:
            pickle_loc = self.run_pickle
        return [self.app, pickle_loc] + self.app_options()

    def run(self):
        if False:
            return 10
        path_name_fragment = re.sub('[^\\w]', '_', self.name)
        self.run_path = tempfile.mkdtemp(prefix=path_name_fragment)
        self.run_pickle = os.path.join(self.run_path, '.'.join([path_name_fragment, 'pickle']))
        with open(self.run_pickle, 'wb') as fd:
            module_path = os.path.abspath(inspect.getfile(self.__class__))
            shutil.copy(module_path, os.path.join(self.run_path, '.'))
            self._dump(fd)
        try:
            super(PySparkTask, self).run()
        finally:
            shutil.rmtree(self.run_path)

    def _dump(self, fd):
        if False:
            for i in range(10):
                print('nop')
        with self.no_unpicklable_properties():
            if self.__module__ == '__main__':
                d = pickle.dumps(self, protocol=self.pickle_protocol)
                module_name = os.path.basename(sys.argv[0]).rsplit('.', 1)[0]
                d = d.replace(b'c__main__', b'c' + module_name.encode('ascii'))
                fd.write(d)
            else:
                pickle.dump(self, fd, protocol=self.pickle_protocol)

    def _setup_packages(self, sc):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method compresses and uploads packages to the cluster\n\n        '
        packages = self.py_packages
        if not packages:
            return
        for package in packages:
            mod = importlib.import_module(package)
            try:
                mod_path = mod.__path__[0]
            except AttributeError:
                mod_path = mod.__file__
            os.makedirs(self.run_path, exist_ok=True)
            tar_path = os.path.join(self.run_path, package + '.tar.gz')
            tar = tarfile.open(tar_path, 'w:gz')
            tar.add(mod_path, os.path.basename(mod_path))
            tar.close()
            sc.addPyFile(tar_path)