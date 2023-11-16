from bigdl.dllib.utils.common import *
from bigdl.dllib.utils.file_utils import callZooFunc
from bigdl.dllib.utils.utils import set_python_home
from bigdl.dllib.utils.log4Error import *
import warnings
import multiprocessing
import os
import threading
import sys

def init_spark_on_local(cores=2, conf=None, python_location=None, spark_log_level='WARN', redirect_spark_log=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a SparkContext with BigDL configurations on the local machine.\n\n    :param cores: The number of cores for Spark local. Default to be 2. You can also set it to "*"\n           to use all the available cores. i.e `init_spark_on_local(cores="*")`\n    :param conf: You can append extra conf for Spark in key-value format.\n           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.\n           Default to be None.\n    :param python_location: The path to your running Python executable. If not specified, the\n           default Python interpreter in effect would be used.\n    :param spark_log_level: The log level for Spark. Default to be \'WARN\'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n\n    :return: An instance of SparkContext.\n    '
    from bigdl.dllib.utils.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level, redirect_spark_log=redirect_spark_log)
    set_python_home()
    return runner.init_spark_on_local(cores=cores, conf=conf, python_location=python_location)

def init_spark_on_yarn(hadoop_conf, conda_name, num_executors, executor_cores, executor_memory='2g', driver_cores=4, driver_memory='2g', extra_executor_memory_for_ray=None, extra_python_lib=None, penv_archive=None, additional_archive=None, hadoop_user_name=None, spark_yarn_archive=None, spark_log_level='WARN', redirect_spark_log=True, jars=None, conf=None, py_files=None):
    if False:
        i = 10
        return i + 15
    '\n    Create a SparkContext with BigDL configurations on Yarn cluster for yarn-client mode.\n    You only need to create a conda environment and install the python dependencies in that\n    environment beforehand on the driver machine. These dependencies would be automatically\n    packaged and distributed to the whole Yarn cluster.\n\n    :param hadoop_conf: The path to the yarn configuration folder.\n    :param conda_name: The name of the conda environment.\n    :param num_executors: The number of Spark executors.\n    :param executor_cores: The number of cores for each executor.\n    :param executor_memory: The memory for each executor. Default to be \'2g\'.\n    :param driver_cores: The number of cores for the Spark driver. Default to be 4.\n    :param driver_memory: The memory for the Spark driver. Default to be \'1g\'.\n    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.\n    :param extra_python_lib: Extra python files or packages needed for distribution.\n           Default to be None.\n    :param penv_archive: Ideally, the program would auto-pack the conda environment specified by\n           \'conda_name\', but you can also pass the path to a packed file in "tar.gz" format here.\n           Default to be None.\n    :param additional_archive: Comma-separated list of additional archives to be uploaded and\n           unpacked on executors. Default to be None.\n    :param hadoop_user_name: The user name for running the yarn cluster. Default to be None.\n           The default None means reading from env, the value of os.environ["HADOOP_USER_NAME"],\n           or current user if HADOOP_USER_NAME is unset.\n    :param spark_yarn_archive: Conf value for setting spark.yarn.archive. Default to be None.\n    :param spark_log_level: The log level for Spark. Default to be \'WARN\'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n    :param jars: Comma-separated list of jars to be included on driver and executor\'s classpath.\n           Default to be None.\n    :param conf: You can append extra conf for Spark in key-value format.\n           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.\n           Default to be None.\n\n    :return: An instance of SparkContext.\n    '
    from bigdl.dllib.utils.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level, redirect_spark_log=redirect_spark_log)
    set_python_home()
    sc = runner.init_spark_on_yarn(hadoop_conf=hadoop_conf, conda_name=conda_name, num_executors=num_executors, executor_cores=executor_cores, executor_memory=executor_memory, driver_cores=driver_cores, driver_memory=driver_memory, extra_executor_memory_for_ray=extra_executor_memory_for_ray, extra_python_lib=extra_python_lib, penv_archive=penv_archive, additional_archive=additional_archive, hadoop_user_name=hadoop_user_name, spark_yarn_archive=spark_yarn_archive, jars=jars, conf=conf, py_files=py_files)
    return sc

def init_spark_on_yarn_cluster(hadoop_conf, conda_name, num_executors, executor_cores, executor_memory='2g', driver_cores=4, driver_memory='2g', extra_executor_memory_for_ray=None, extra_python_lib=None, penv_archive=None, additional_archive=None, hadoop_user_name=None, spark_yarn_archive=None, spark_log_level='WARN', redirect_spark_log=True, jars=None, conf=None, py_files=None):
    if False:
        return 10
    '\n    Create a SparkContext with BigDL configurations on Yarn cluster for yarn-cluster mode.\n    You only need to create a conda environment and install the python dependencies in that\n    environment beforehand on the driver machine. These dependencies would be automatically\n    packaged and distributed to the whole Yarn cluster.\n\n    :param hadoop_conf: The path to the yarn configuration folder.\n    :param conda_name: The name of the conda environment.\n    :param num_executors: The number of Spark executors.\n    :param executor_cores: The number of cores for each executor.\n    :param executor_memory: The memory for each executor. Default to be \'2g\'.\n    :param driver_cores: The number of cores for the Spark driver. Default to be 4.\n    :param driver_memory: The memory for the Spark driver. Default to be \'1g\'.\n    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.\n    :param extra_python_lib: Extra python files or packages needed for distribution.\n           Default to be None.\n    :param penv_archive: Ideally, the program would auto-pack the conda environment specified by\n           \'conda_name\', but you can also pass the path to a packed file in "tar.gz" format here.\n           Default to be None.\n    :param additional_archive: Comma-separated list of additional archives to be uploaded and\n           unpacked on executors. Default to be None.\n    :param hadoop_user_name: The user name for running the yarn cluster. Default to be None.\n           The default None means reading from env, the value of os.environ["HADOOP_USER_NAME"],\n           or current user if HADOOP_USER_NAME is unset.\n    :param spark_yarn_archive: Conf value for setting spark.yarn.archive. Default to be None.\n    :param spark_log_level: The log level for Spark. Default to be \'WARN\'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n    :param jars: Comma-separated list of jars to be included on driver and executor\'s classpath.\n           Default to be None.\n    :param conf: You can append extra conf for Spark in key-value format.\n           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.\n           Default to be None.\n\n    :return: An instance of SparkContext.\n    '
    if os.environ.get('OnAppMaster', 'False') == 'True':
        sc = init_internal_nncontext()
        return sc
    else:
        from bigdl.dllib.utils.spark import SparkRunner
        runner = SparkRunner(spark_log_level=spark_log_level, redirect_spark_log=redirect_spark_log)
        return_value = runner.init_spark_on_yarn_cluster(hadoop_conf=hadoop_conf, conda_name=conda_name, num_executors=num_executors, executor_cores=executor_cores, executor_memory=executor_memory, driver_cores=driver_cores, driver_memory=driver_memory, extra_executor_memory_for_ray=extra_executor_memory_for_ray, extra_python_lib=extra_python_lib, penv_archive=penv_archive, additional_archive=additional_archive, hadoop_user_name=hadoop_user_name, spark_yarn_archive=spark_yarn_archive, jars=jars, conf=conf, py_files=py_files)
    sys.exit(return_value)

def init_spark_standalone(num_executors, executor_cores, executor_memory='2g', driver_cores=4, driver_memory='2g', master=None, extra_executor_memory_for_ray=None, extra_python_lib=None, spark_log_level='WARN', redirect_spark_log=True, conf=None, jars=None, python_location=None, enable_numa_binding=False):
    if False:
        while True:
            i = 10
    '\n    Create a SparkContext with BigDL configurations on Spark standalone cluster.\n\n    You need to specify master if you already have a Spark standalone cluster. For a\n    standalone cluster with multiple nodes, make sure that BigDL is installed via\n    pip in the Python environment on every node.\n    If master is not specified, a new Spark standalone cluster on the current single node\n    would be started first and the SparkContext would use its master address. You need to\n    call `stop_spark_standalone` after your program finishes to shutdown the cluster.\n\n    :param num_executors: The number of Spark executors.\n    :param executor_cores: The number of cores for each executor.\n    :param executor_memory: The memory for each executor. Default to be \'2g\'.\n    :param driver_cores: The number of cores for the Spark driver. Default to be 4.\n    :param driver_memory: The memory for the Spark driver. Default to be \'1g\'.\n    :param master: The master URL of an existing Spark standalone cluster: \'spark://master:port\'.\n    You only need to specify this if you have already started a standalone cluster.\n    Default to be None and a new standalone cluster would be started in this case.\n    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.\n    :param extra_python_lib: Extra python files or packages needed for distribution.\n           Default to be None.\n    :param spark_log_level: The log level for Spark. Default to be \'WARN\'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n    :param jars: Comma-separated list of jars to be included on driver and executor\'s classpath.\n           Default to be None.\n    :param conf: You can append extra conf for Spark in key-value format.\n           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.\n           Default to be None.\n    :param python_location: The path to your running Python executable. If not specified, the\n           default Python interpreter in effect would be used.\n    :param enable_numa_binding: Whether to use numactl to start spark worker in order to bind\n           different worker processes to different cpus and memory areas. This is may lead to\n           better performance on a multi-sockets machine. Defaults to False.\n\n    :return: An instance of SparkContext.\n    '
    from bigdl.dllib.utils.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level, redirect_spark_log=redirect_spark_log)
    set_python_home()
    sc = runner.init_spark_standalone(num_executors=num_executors, executor_cores=executor_cores, executor_memory=executor_memory, driver_cores=driver_cores, driver_memory=driver_memory, master=master, extra_executor_memory_for_ray=extra_executor_memory_for_ray, extra_python_lib=extra_python_lib, conf=conf, jars=jars, python_location=python_location, enable_numa_binding=enable_numa_binding)
    return sc

def init_spark_on_k8s(master, container_image, conda_name, num_executors, executor_cores, executor_memory='2g', driver_memory='2g', driver_cores=4, extra_executor_memory_for_ray=None, extra_python_lib=None, penv_archive=None, spark_log_level='WARN', redirect_spark_log=True, jars=None, conf=None, python_location=None):
    if False:
        print('Hello World!')
    '\n    Create a SparkContext with BigDL configurations on Kubernetes cluster for k8s client\n    mode. You are recommended to use the Docker image intelanalytics/bigdl-k8s:latest.\n    You can refer to https://github.com/intel-analytics/BigDL/tree/main/docker/bigdl-k8s\n    to build your own Docker image.\n\n    :param master: The master address of your k8s cluster.\n    :param container_image: The name of the docker container image for Spark executors.\n           For example, intelanalytics/bigdl-k8s:latest\n    :param conda_name: The name of the conda environment.\n    :param num_executors: The number of Spark executors.\n    :param executor_cores: The number of cores for each executor.\n    :param executor_memory: The memory for each executor. Default to be \'2g\'.\n    :param driver_cores: The number of cores for the Spark driver. Default to be 4.\n    :param driver_memory: The memory for the Spark driver. Default to be \'1g\'.\n    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.\n    :param extra_python_lib: Extra python files or packages needed for distribution.\n           Default to be None.\n    :param penv_archive: Ideally, the program would auto-pack the conda environment specified by\n           \'conda_name\', but you can also pass the path to a packed file in "tar.gz" format here.\n           Default to be None.\n    :param spark_log_level: The log level for Spark. Default to be \'WARN\'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n    :param jars: Comma-separated list of jars to be included on driver and executor\'s classpath.\n           Default to be None.\n    :param conf: You can append extra conf for Spark in key-value format.\n           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.\n           Default to be None.\n    :param python_location: The path to your running Python executable. If not specified, the\n           default Python interpreter in effect would be used.\n\n    :return: An instance of SparkContext.\n    '
    from bigdl.dllib.utils.spark import SparkRunner
    runner = SparkRunner(spark_log_level=spark_log_level, redirect_spark_log=redirect_spark_log)
    sc = runner.init_spark_on_k8s(master=master, container_image=container_image, conda_name=conda_name, num_executors=num_executors, executor_cores=executor_cores, executor_memory=executor_memory, driver_memory=driver_memory, driver_cores=driver_cores, extra_executor_memory_for_ray=extra_executor_memory_for_ray, extra_python_lib=extra_python_lib, penv_archive=penv_archive, jars=jars, conf=conf, python_location=python_location)
    return sc

def init_spark_on_k8s_cluster(master, container_image, num_executors, executor_cores, executor_memory='2g', driver_memory='2g', driver_cores=4, extra_executor_memory_for_ray=None, extra_python_lib=None, penv_archive=None, spark_log_level='WARN', redirect_spark_log=True, jars=None, conf=None, python_location=None):
    if False:
        i = 10
        return i + 15
    '\n    Create a SparkContext with BigDL configurations on Kubernetes cluster for k8s cluster\n    mode. You are recommended to use the Docker image intelanalytics/bigdl-k8s:latest.\n    You can refer to https://github.com/intel-analytics/BigDL/tree/main/docker/bigdl-k8s\n    to build your own Docker image.\n    :param master: The master address of your k8s cluster.\n    :param container_image: The name of the docker container image for Spark executors.\n           For example, intelanalytics/bigdl-k8s:latest\n    :param executor_cores: The number of cores for each executor.\n    :param executor_memory: The memory for each executor. Default to be \'2g\'.\n    :param driver_cores: The number of cores for the Spark driver. Default to be 4.\n    :param driver_memory: The memory for the Spark driver. Default to be \'1g\'.\n    :param extra_executor_memory_for_ray: The extra memory for Ray services. Default to be None.\n    :param extra_python_lib: Extra python files or packages needed for distribution.\n           Default to be None.\n    :param penv_archive: the path to a packed conda file in "tar.gz" format here. The path should be\n           that k8s pod can access.\n           Default to be None.\n    :param spark_log_level: The log level for Spark. Default to be \'WARN\'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n    :param jars: Comma-separated list of jars to be included on driver and executor\'s classpath.\n           Default to be None.\n    :param conf: You can append extra conf for Spark in key-value format.\n           i.e conf={"spark.executor.extraJavaOptions": "-XX:+PrintGCDetails"}.\n           Default to be None.\n    :param python_location: The path to your running Python executable. If not specified, the\n           default Python interpreter in effect would be used.\n    :return: An instance of SparkContext.\n    '
    if os.environ.get('onDriver', 'False') == 'True':
        sc = init_internal_nncontext()
        return sc
    else:
        from bigdl.dllib.utils.spark import SparkRunner
        runner = SparkRunner(spark_log_level=spark_log_level, redirect_spark_log=redirect_spark_log)
        return_value = runner.init_spark_on_k8s_cluster(master=master, container_image=container_image, num_executors=num_executors, executor_cores=executor_cores, executor_memory=executor_memory, driver_memory=driver_memory, driver_cores=driver_cores, extra_executor_memory_for_ray=extra_executor_memory_for_ray, extra_python_lib=extra_python_lib, penv_archive=penv_archive, jars=jars, conf=conf, python_location=python_location)
        sys.exit(return_value)

def stop_spark_standalone():
    if False:
        i = 10
        return i + 15
    '\n    Stop the Spark standalone cluster created from init_spark_standalone (master not specified).\n    '
    from bigdl.dllib.utils.spark import SparkRunner
    SparkRunner.stop_spark_standalone()

class ZooContextMeta(type):
    _log_output = False
    _barrier_mode = True

    @property
    def log_output(cls):
        if False:
            for i in range(10):
                print('nop')
        "\n        Whether to redirect Spark driver JVM's stdout and stderr to the current\n        python process. This is useful when running BigDL in jupyter notebook.\n        Default to be False. Needs to be set before initializing SparkContext.\n        "
        return cls._log_output

    @log_output.setter
    def log_output(cls, value):
        if False:
            for i in range(10):
                print('nop')
        invalidInputError(isinstance(value, bool), 'log_output should either be True or False')
        cls._log_output = value

    @property
    def barrier_mode(cls):
        if False:
            while True:
                i = 10
        '\n        Whether to use Spark barrier mode to launch Ray, which is supported in Spark 2.4+ and when\n        dynamic allocation is disabled.\n        Default to be True.\n        '
        return cls._barrier_mode

    @barrier_mode.setter
    def barrier_mode(cls, value):
        if False:
            i = 10
            return i + 15
        invalidInputError(isinstance(value, bool), 'barrier_mode should either be True or False')
        cls._barrier_mode = value

class ZooContext(metaclass=ZooContextMeta):
    pass

def _read_stream(fd, fn):
    if False:
        return 10
    'Reads bytes from a file descriptor, utf-8 decodes them, and passes them\n    to the provided callback function on the next IOLoop tick.\n    Assumes fd.read will block and should be used in a thread.\n    Parameters\n    ----------\n    fd : file\n        File descriptor to read\n    fn : callable(str) -> None\n        Callback function that handles chunks of text\n    '
    while True:
        buff = fd.read(8192)
        if buff:
            fn(buff)

def init_nncontext(conf=None, cluster_mode='spark-submit', spark_log_level='WARN', redirect_spark_log=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates or gets a SparkContext with optimized configurations for BigDL performance.\n    This method will also initialize the BigDL engine.\n\n    Note: If you use spark-shell or Jupyter notebook, as the SparkContext is created\n    before your code, you have to set the Spark configurations through command line options\n    or the properties file before calling this method. In this case, you are recommended\n    to use the launch scripts we provide:\n    https://github.com/intel-analytics/BigDL/tree/main/scripts.\n\n    :param conf: An instance of SparkConf. If not specified, a new SparkConf with\n           BigDL configurations would be created and used.\n           You can also input a string here to indicate the name of the application.\n    :param cluster_mode: The mode for the Spark cluster. One of "local", "yarn-client",\n       "yarn-cluster", "k8s-client", "standalone" and "spark-submit". Default to be "local".\n\n       For "spark-submit", you are supposed to use spark-submit to submit the application.\n       In this case, please set the Spark configurations through command line options or\n       the properties file. You need to use "spark-submit" for yarn-cluster or k8s-cluster mode.\n       To make things easier, you are recommended to use the launch scripts we provide:\n       https://github.com/intel-analytics/BigDL/tree/main/scripts.\n\n       For other cluster modes, you are recommended to install and run BigDL through\n       pip, which is more convenient.\n    :param spark_log_level: The log level for Spark. Default to be \'WARN\'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n\n    :return: An instance of SparkContext.\n    '
    cluster_mode = cluster_mode.lower()
    memory = '2g'
    cores = 2
    num_nodes = 1
    spark_args = {}
    spark_args['spark_log_level'] = spark_log_level
    spark_args['redirect_spark_log'] = redirect_spark_log
    if conf and (not isinstance(conf, six.string_types)):
        memory = conf.get('spark.executor.memory', '2g')
        if conf.get('spark.executor.cores'):
            cores = conf.get('spark.executor.cores')
        if conf.get('spark.executor.instances'):
            num_nodes = conf.get('spark.executor.instances')
        spark_args.update(conf.getAll())
    if cluster_mode == 'spark-submit':
        sc = init_internal_nncontext(conf, spark_log_level, redirect_spark_log)
    elif cluster_mode == 'local':
        if conf:
            os.environ['SPARK_DRIVER_MEMORY'] = conf.get('spark.driver.memory')
        else:
            os.environ['SPARK_DRIVER_MEMORY'] = memory
        python_location = None
        if 'python_location' in kwargs:
            python_location = kwargs['python_location']
        sc = init_spark_on_local(2, spark_args, python_location, spark_log_level, redirect_spark_log)
    elif cluster_mode in ('yarn-client', 'yarn-cluster'):
        hadoop_conf = os.environ.get('HADOOP_CONF_DIR')
        if not hadoop_conf:
            invalidInputError('hadoop_conf' in kwargs, 'Directory path to hadoop conf not found for yarn-client mode. Please either specify argument hadoop_conf orset the environment variable HADOOP_CONF_DIR')
            hadoop_conf = kwargs['hadoop_conf']
        from bigdl.dllib.utils.utils import detect_conda_env_name
        conda_env_name = detect_conda_env_name()
        for key in ['driver_cores', 'driver_memory', 'extra_executor_memory_for_ray', 'extra_python_lib', 'penv_archive', 'additional_archive', 'hadoop_user_name', 'spark_yarn_archive', 'jars']:
            if key in kwargs:
                spark_args[key] = kwargs[key]
        if cluster_mode == 'yarn-client':
            from bigdl.dllib.nncontext import init_spark_on_yarn
            sc = init_spark_on_yarn(hadoop_conf=hadoop_conf, conda_name=conda_env_name, num_executors=num_nodes, executor_cores=cores, executor_memory=memory, conf=spark_args)
        else:
            sc = init_spark_on_yarn_cluster(hadoop_conf=hadoop_conf, conda_name=conda_env_name, num_executors=num_nodes, executor_cores=cores, executor_memory=memory, conf=spark_args)
    elif cluster_mode.startswith('k8s'):
        invalidInputError('master' in kwargs, 'master is not set in k8s mode', 'Please specify master for k8s mode')
        invalidInputError('container_image' in kwargs, 'container_image is not set in k8s mode', 'Please specify container_image for k8s mode')
        for key in ['driver_cores', 'driver_memory', 'extra_executor_memory_for_ray', 'extra_python_lib', 'penv_archive', 'jars', 'python_location']:
            if key in kwargs:
                spark_args[key] = kwargs[key]
        from bigdl.dllib.nncontext import init_spark_on_k8s, init_spark_on_k8s_cluster
        if cluster_mode == 'k8s-cluster':
            sc = init_spark_on_k8s_cluster(master=kwargs['master'], container_image=kwargs['container_image'], num_executors=num_nodes, executor_cores=cores, executor_memory=memory, **spark_args)
        else:
            from bigdl.dllib.utils.utils import detect_conda_env_name
            conda_env_name = detect_conda_env_name()
            sc = init_spark_on_k8s(master=kwargs['master'], container_image=kwargs['container_image'], conda_name=conda_env_name, num_executors=num_nodes, executor_cores=cores, executor_memory=memory, **spark_args)
    elif cluster_mode == 'standalone':
        for key in ['driver_cores', 'driver_memory', 'extra_executor_memory_for_ray', 'extra_python_lib', 'jars', 'master', 'python_location', 'enable_numa_binding']:
            if key in kwargs:
                spark_args[key] = kwargs[key]
        from bigdl.dllib.nncontext import init_spark_standalone
        sc = init_spark_standalone(num_executors=num_nodes, executor_cores=cores, executor_memory=memory, **spark_args)
    else:
        invalidInputError(False, 'cluster_mode can only be local, yarn-client, yarn-cluster, standalone or spark-submit, but got: %s'.format(cluster_mode))
    return sc

def init_internal_nncontext(conf=None, spark_log_level='WARN', redirect_spark_log=True):
    if False:
        return 10
    "\n    Creates or gets a SparkContext with optimized configurations for BigDL performance.\n    This method will also initialize the BigDL engine.\n\n    Note: If you use spark-shell or Jupyter notebook, as the SparkContext is created\n    before your code, you have to set the Spark configurations through command line options\n    or the properties file before calling this method. In this case, you are recommended\n    to use the launch scripts we provide:\n    https://github.com/intel-analytics/BigDL/tree/main/scripts.\n\n    :param conf: An instance of SparkConf. If not specified, a new SparkConf with\n           BigDL configurations would be created and used.\n           You can also input a string here to indicate the name of the application.\n    :param spark_log_level: The log level for Spark. Default to be 'WARN'.\n    :param redirect_spark_log: Whether to redirect the Spark log to local file. Default to be True.\n\n    :return: An instance of SparkContext.\n    "
    has_activate_sc = SparkContext._active_spark_context is not None
    if isinstance(conf, six.string_types):
        sc = getOrCreateSparkContext(conf=None, appName=conf)
    else:
        sc = getOrCreateSparkContext(conf=conf)
    sc.setLogLevel(spark_log_level)
    if ZooContext.log_output:
        import uuid
        uuidStr = str(uuid.uuid4())
        log_path = 'bigdl' + uuidStr + '.log'
        abs_path = '/tmp/' + log_path
        logger.info(f'log path {abs_path}')
        stderr_reader = threading.Thread(target=_read_stream, daemon=True, kwargs=dict(fd=open(abs_path, 'w+'), fn=sys.stdout.write))
        stderr_reader.start()
        sc.setSystemProperty('logFilename', log_path)
    check_version()
    if redirect_spark_log:
        redire_spark_logs()
        show_bigdl_info_logs()
    init_engine()
    set_python_home()
    return sc

def getOrCreateSparkContext(conf=None, appName=None):
    if False:
        print('Hello World!')
    '\n    Get the current active SparkContext or create a new SparkContext.\n    :param conf: An instance of SparkConf. If not specified, a new SparkConf with\n           BigDL configurations would be created and used.\n    :param appName: The name of the application if any.\n\n    :return: An instance of SparkContext.\n    '
    with SparkContext._lock:
        if SparkContext._active_spark_context is None:
            spark_conf = init_spark_conf() if conf is None else conf
            if appName:
                spark_conf.setAppName(appName)
            return SparkContext.getOrCreate(spark_conf)
        else:
            return SparkContext.getOrCreate()

def get_analytics_zoo_conf():
    if False:
        return 10
    zoo_conf_file = 'spark-bigdl.conf'
    zoo_python_wrapper = 'python-api.zip'
    for p in sys.path:
        if zoo_conf_file in p and os.path.isfile(p):
            with open(p) if sys.version_info < (3,) else open(p, encoding='latin-1') as conf_file:
                return load_conf(conf_file.read())
        if zoo_python_wrapper in p and os.path.isfile(p):
            import zipfile
            with zipfile.ZipFile(p, 'r') as zip_conf:
                if zoo_conf_file in zip_conf.namelist():
                    content = zip_conf.read(zoo_conf_file)
                    if sys.version_info >= (3,):
                        content = str(content, 'latin-1')
                    return load_conf(content)
    return {}

def init_env(conf):
    if False:
        for i in range(10):
            print('nop')
    kmp_affinity = 'granularity=fine,compact,1,0'
    kmp_settings = '1'
    omp_num_threads = '1'
    kmp_blocktime = '0'
    if 'KMP_AFFINITY' in os.environ:
        kmp_affinity = os.environ['KMP_AFFINITY']
    if 'KMP_SETTINGS' in os.environ:
        kmp_settings = os.environ['KMP_SETTINGS']
    if 'ZOO_NUM_MKLTHREADS' in os.environ:
        if os.environ['ZOO_NUM_MKLTHREADS'].lower() == 'all':
            omp_num_threads = conf.get('spark.executor.cores', str(multiprocessing.cpu_count()))
        else:
            omp_num_threads = os.environ['ZOO_NUM_MKLTHREADS']
    elif 'OMP_NUM_THREADS' in os.environ:
        omp_num_threads = os.environ['OMP_NUM_THREADS']
    if 'KMP_BLOCKTIME' in os.environ:
        kmp_blocktime = os.environ['KMP_BLOCKTIME']
    conf.set('spark.executorEnv.KMP_AFFINITY', kmp_affinity)
    conf.set('spark.executorEnv.KMP_SETTINGS', kmp_settings)
    conf.set('spark.executorEnv.KMP_BLOCKTIME', kmp_blocktime)
    conf.set('spark.executorEnv.OMP_NUM_THREADS', omp_num_threads)
    os.environ['KMP_AFFINITY'] = kmp_affinity
    os.environ['KMP_SETTINGS'] = kmp_settings
    os.environ['OMP_NUM_THREADS'] = omp_num_threads
    os.environ['KMP_BLOCKTIME'] = kmp_blocktime

def init_spark_conf(conf=None):
    if False:
        while True:
            i = 10
    spark_conf = SparkConf()
    if conf:
        spark_conf.setAll(conf.items())
    init_env(spark_conf)
    zoo_conf = get_analytics_zoo_conf()
    if conf and 'spark.driver.extraJavaOptions' in conf:
        extraJavaOptions = conf['spark.driver.extraJavaOptions']
        spark_conf.setAll(zoo_conf.items())
        concatJavaOptions = extraJavaOptions + ' ' + zoo_conf.get('spark.driver.extraJavaOptions')
        spark_conf.set('spark.driver.extraJavaOptions', concatJavaOptions)
    else:
        spark_conf.setAll(zoo_conf.items())
    if os.environ.get('BIGDL_JARS', None) and (not is_spark_below_2_2()):
        if 'PYSPARK_SUBMIT_ARGS' in os.environ:
            submit_args = os.environ['PYSPARK_SUBMIT_ARGS']
            start = submit_args.find('pyspark-shell')
            submit_args = submit_args[:start] + '--driver-class-path ' + os.environ['BIGDL_JARS'] + submit_args[start:]
        else:
            submit_args = f" --driver-class-path {os.environ['BIGDL_JARS']} pyspark-shell "
        print('pyspark_submit_args is:', submit_args)
        os.environ['PYSPARK_SUBMIT_ARGS'] = submit_args
    python_lib = os.environ.get('PYSPARK_FILES', None)
    if python_lib:
        existing_py_files = spark_conf.get('spark.submit.pyFiles')
        if existing_py_files:
            spark_conf.set(key='spark.submit.pyFiles', value='%s,%s' % (python_lib, existing_py_files))
        else:
            spark_conf.set(key='spark.submit.pyFiles', value=python_lib)
    return spark_conf

def check_version():
    if False:
        i = 10
        return i + 15
    sc = getOrCreateSparkContext()
    conf = sc._conf
    if conf.get('spark.analytics.zoo.versionCheck', 'False').lower() == 'true':
        report_warn = conf.get('spark.analytics.zoo.versionCheck.warning', 'False').lower() == 'true'
        _check_spark_version(sc, report_warn)

def _split_full_version(version):
    if False:
        print('Hello World!')
    parts = version.split('.')
    major = parts[0]
    feature = parts[1]
    maintenance = parts[2]
    return (major, feature, maintenance)

def _check_spark_version(sc, report_warn):
    if False:
        return 10
    version_info = _get_bigdl_verion_conf()
    (c_major, c_feature, c_maintenance) = _split_full_version(version_info['spark_version'])
    (r_major, r_feature, r_maintenance) = _split_full_version(sc.version)
    error_message = '\n        The compile time spark version is not compatible with the spark runtime version.\n        Compile time version is %s, runtime version is %s. If you want bypass this check,\n        please set spark.analytics.zoo.versionCheck to false, and if you want to only report\n        an warning message, please set spark.analytics.zoo.versionCheck.warning to true.\n        ' % (version_info['spark_version'], sc.version)
    if c_major != r_major:
        if not report_warn:
            invalidInputError(False, error_message)
        else:
            warnings.warn(error_message)
    elif not (c_maintenance == r_maintenance and c_feature == r_feature):
        warnings.warn('The compile time spark version may not compatible with ' + 'the Spark runtime version. ' + 'Compile time version is %s, ' % version_info['spark_version'] + 'runtime version is %s' % sc.version)

def _get_bigdl_verion_conf():
    if False:
        while True:
            i = 10
    bigdl_build_file = 'zoo-version-info.properties'
    bigdl_python_wrapper = 'python-api.zip'
    for p in sys.path:
        if bigdl_build_file in p and os.path.isfile(p):
            with open(p) if sys.version_info < (3,) else open(p, encoding='latin-1') as conf_file:
                return load_conf(conf_file.read(), '=')
        if bigdl_python_wrapper in p and os.path.isfile(p):
            import zipfile
            with zipfile.ZipFile(p, 'r') as zip_conf:
                if bigdl_build_file in zip_conf.namelist():
                    content = zip_conf.read(bigdl_build_file)
                    if sys.version_info >= (3,):
                        content = str(content, 'latin-1')
                    return load_conf(content, '=')
    invalidInputError(False, 'Error while locating file zoo-version-info.properties, please make sure the mvn generate-resources phase is executed and a zoo-version-info.properties file is located in zoo/target/extra-resources')

def load_conf(conf_str, split_char=None):
    if False:
        print('Hello World!')
    return dict((line.split(split_char) for line in conf_str.split('\n') if '#' not in line and line.strip()))

def set_optimizer_version(optimizerVersion, bigdl_type='float'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Set DistriOptimizer version.\n    param optimizerVersion: should be "OptimizerV1" or "OptimizerV2".\n    '
    callZooFunc(bigdl_type, 'setOptimizerVersion', optimizerVersion)

def get_optimizer_version(bigdl_type='float'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get DistriOptimizer version.\n    return optimizerVersion\n    '
    return callZooFunc(bigdl_type, 'getOptimizerVersion')