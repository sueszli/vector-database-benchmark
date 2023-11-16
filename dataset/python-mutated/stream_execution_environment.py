import os
import tempfile
from typing import List, Any, Optional, cast
from py4j.java_gateway import JavaObject
from pyflink.common import Configuration, WatermarkStrategy
from pyflink.common.execution_config import ExecutionConfig
from pyflink.common.io import InputFormat
from pyflink.common.job_client import JobClient
from pyflink.common.job_execution_result import JobExecutionResult
from pyflink.common.restart_strategy import RestartStrategies, RestartStrategyConfiguration
from pyflink.common.typeinfo import TypeInformation, Types
from pyflink.datastream import SlotSharingGroup
from pyflink.datastream.checkpoint_config import CheckpointConfig
from pyflink.datastream.checkpointing_mode import CheckpointingMode
from pyflink.datastream.connectors import Source
from pyflink.datastream.data_stream import DataStream
from pyflink.datastream.execution_mode import RuntimeExecutionMode
from pyflink.datastream.functions import SourceFunction
from pyflink.datastream.state_backend import _from_j_state_backend, StateBackend
from pyflink.datastream.time_characteristic import TimeCharacteristic
from pyflink.datastream.utils import ResultTypeQueryable
from pyflink.java_gateway import get_gateway
from pyflink.serializers import PickleSerializer
from pyflink.util.java_utils import load_java_class, add_jars_to_context_class_loader, invoke_method, get_field_value, is_local_deployment, get_j_env_configuration
__all__ = ['StreamExecutionEnvironment']

class StreamExecutionEnvironment(object):
    """
    The StreamExecutionEnvironment is the context in which a streaming program is executed. A
    *LocalStreamEnvironment* will cause execution in the attached JVM, a
    *RemoteStreamEnvironment* will cause execution on a remote setup.

    The environment provides methods to control the job execution (such as setting the parallelism
    or the fault tolerance/checkpointing parameters) and to interact with the outside world (data
    access).
    """

    def __init__(self, j_stream_execution_environment, serializer=PickleSerializer()):
        if False:
            while True:
                i = 10
        self._j_stream_execution_environment = j_stream_execution_environment
        self.serializer = serializer
        self._open()

    def get_config(self) -> ExecutionConfig:
        if False:
            return 10
        '\n        Gets the config object.\n\n        :return: The :class:`~pyflink.common.ExecutionConfig` object.\n        '
        return ExecutionConfig(self._j_stream_execution_environment.getConfig())

    def set_parallelism(self, parallelism: int) -> 'StreamExecutionEnvironment':
        if False:
            while True:
                i = 10
        '\n        Sets the parallelism for operations executed through this environment.\n        Setting a parallelism of x here will cause all operators (such as map,\n        batchReduce) to run with x parallel instances. This method overrides the\n        default parallelism for this environment. The\n        *LocalStreamEnvironment* uses by default a value equal to the\n        number of hardware contexts (CPU cores / threads). When executing the\n        program via the command line client from a JAR file, the default degree\n        of parallelism is the one configured for that setup.\n\n        :param parallelism: The parallelism.\n        :return: This object.\n        '
        self._j_stream_execution_environment = self._j_stream_execution_environment.setParallelism(parallelism)
        return self

    def set_max_parallelism(self, max_parallelism: int) -> 'StreamExecutionEnvironment':
        if False:
            print('Hello World!')
        '\n        Sets the maximum degree of parallelism defined for the program. The upper limit (inclusive)\n        is 32768.\n\n        The maximum degree of parallelism specifies the upper limit for dynamic scaling. It also\n        defines the number of key groups used for partitioned state.\n\n        :param max_parallelism: Maximum degree of parallelism to be used for the program,\n                                with 0 < maxParallelism <= 2^15.\n        :return: This object.\n        '
        self._j_stream_execution_environment = self._j_stream_execution_environment.setMaxParallelism(max_parallelism)
        return self

    def register_slot_sharing_group(self, slot_sharing_group: SlotSharingGroup) -> 'StreamExecutionEnvironment':
        if False:
            while True:
                i = 10
        "\n        Register a slot sharing group with its resource spec.\n\n        Note that a slot sharing group hints the scheduler that the grouped operators CAN be\n        deployed into a shared slot. There's no guarantee that the scheduler always deploy the\n        grouped operators together. In cases grouped operators are deployed into separate slots, the\n        slot resources will be derived from the specified group requirements.\n\n        :param slot_sharing_group: Which contains name and its resource spec.\n        :return: This object.\n        "
        self._j_stream_execution_environment = self._j_stream_execution_environment.registerSlotSharingGroup(slot_sharing_group.get_java_slot_sharing_group())
        return self

    def get_parallelism(self) -> int:
        if False:
            print('Hello World!')
        '\n        Gets the parallelism with which operation are executed by default.\n        Operations can individually override this value to use a specific\n        parallelism.\n\n        :return: The parallelism used by operations, unless they override that value.\n        '
        return self._j_stream_execution_environment.getParallelism()

    def get_max_parallelism(self) -> int:
        if False:
            return 10
        '\n        Gets the maximum degree of parallelism defined for the program.\n\n        The maximum degree of parallelism specifies the upper limit for dynamic scaling. It also\n        defines the number of key groups used for partitioned state.\n\n        :return: Maximum degree of parallelism.\n        '
        return self._j_stream_execution_environment.getMaxParallelism()

    def set_runtime_mode(self, execution_mode: RuntimeExecutionMode):
        if False:
            return 10
        "\n        Sets the runtime execution mode for the application\n        :class:`~pyflink.datastream.execution_mode.RuntimeExecutionMode`. This\n        is equivalent to setting the `execution.runtime-mode` in your application's\n        configuration file.\n\n        We recommend users to NOT use this method but set the `execution.runtime-mode` using\n        the command-line when submitting the application. Keeping the application code\n        configuration-free allows for more flexibility as the same application will be able to be\n        executed in any execution mode.\n\n        :param execution_mode: The desired execution mode.\n        :return: The execution environment of your application.\n\n        .. versionadded:: 1.13.0\n        "
        return self._j_stream_execution_environment.setRuntimeMode(execution_mode._to_j_execution_mode())

    def set_buffer_timeout(self, timeout_millis: int) -> 'StreamExecutionEnvironment':
        if False:
            return 10
        '\n        Sets the maximum time frequency (milliseconds) for the flushing of the\n        output buffers. By default the output buffers flush frequently to provide\n        low latency and to aid smooth developer experience. Setting the parameter\n        can result in three logical modes:\n\n        - A positive integer triggers flushing periodically by that integer\n        - 0 triggers flushing after every record thus minimizing latency\n        - -1 triggers flushing only when the output buffer is full thus maximizing throughput\n\n        :param timeout_millis: The maximum time between two output flushes.\n        :return: This object.\n        '
        self._j_stream_execution_environment = self._j_stream_execution_environment.setBufferTimeout(timeout_millis)
        return self

    def get_buffer_timeout(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the maximum time frequency (milliseconds) for the flushing of the\n        output buffers. For clarification on the extremal values see\n        :func:`set_buffer_timeout`.\n\n        :return: The timeout of the buffer.\n        '
        return self._j_stream_execution_environment.getBufferTimeout()

    def disable_operator_chaining(self) -> 'StreamExecutionEnvironment':
        if False:
            i = 10
            return i + 15
        '\n        Disables operator chaining for streaming operators. Operator chaining\n        allows non-shuffle operations to be co-located in the same thread fully\n        avoiding serialization and de-serialization.\n\n        :return: This object.\n        '
        self._j_stream_execution_environment = self._j_stream_execution_environment.disableOperatorChaining()
        return self

    def is_chaining_enabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Returns whether operator chaining is enabled.\n\n        :return: True if chaining is enabled, false otherwise.\n        '
        return self._j_stream_execution_environment.isChainingEnabled()

    def is_chaining_of_operators_with_different_max_parallelism_enabled(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns whether operators that have a different max parallelism can be chained.\n\n        :return: True if chaining is enabled, false otherwise\n        '
        return self._j_stream_execution_environment.isChainingOfOperatorsWithDifferentMaxParallelismEnabled()

    def get_checkpoint_config(self) -> CheckpointConfig:
        if False:
            return 10
        '\n        Gets the checkpoint config, which defines values like checkpoint interval, delay between\n        checkpoints, etc.\n\n        :return: The :class:`~pyflink.datastream.CheckpointConfig`.\n        '
        j_checkpoint_config = self._j_stream_execution_environment.getCheckpointConfig()
        return CheckpointConfig(j_checkpoint_config)

    def enable_checkpointing(self, interval: int, mode: CheckpointingMode=None) -> 'StreamExecutionEnvironment':
        if False:
            i = 10
            return i + 15
        '\n        Enables checkpointing for the streaming job. The distributed state of the streaming\n        dataflow will be periodically snapshotted. In case of a failure, the streaming\n        dataflow will be restarted from the latest completed checkpoint.\n\n        The job draws checkpoints periodically, in the given interval. The system uses the\n        given :class:`~pyflink.datastream.CheckpointingMode` for the checkpointing ("exactly once"\n        vs "at least once"). The state will be stored in the configured state backend.\n\n        .. note::\n            Checkpointing iterative streaming dataflows in not properly supported at\n            the moment. For that reason, iterative jobs will not be started if used\n            with enabled checkpointing.\n\n        Example:\n        ::\n\n            >>> env.enable_checkpointing(300000, CheckpointingMode.AT_LEAST_ONCE)\n\n        :param interval: Time interval between state checkpoints in milliseconds.\n        :param mode: The checkpointing mode, selecting between "exactly once" and "at least once"\n                     guaranteed.\n        :return: This object.\n        '
        if mode is None:
            self._j_stream_execution_environment = self._j_stream_execution_environment.enableCheckpointing(interval)
        else:
            j_checkpointing_mode = CheckpointingMode._to_j_checkpointing_mode(mode)
            self._j_stream_execution_environment.enableCheckpointing(interval, j_checkpointing_mode)
        return self

    def get_checkpoint_interval(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the checkpointing interval or -1 if checkpointing is disabled.\n\n        Shorthand for get_checkpoint_config().get_checkpoint_interval().\n\n        :return: The checkpointing interval or -1.\n        '
        return self._j_stream_execution_environment.getCheckpointInterval()

    def get_checkpointing_mode(self) -> CheckpointingMode:
        if False:
            return 10
        '\n        Returns the checkpointing mode (exactly-once vs. at-least-once).\n\n        Shorthand for get_checkpoint_config().get_checkpointing_mode().\n\n        :return: The :class:`~pyflink.datastream.CheckpointingMode`.\n        '
        j_checkpointing_mode = self._j_stream_execution_environment.getCheckpointingMode()
        return CheckpointingMode._from_j_checkpointing_mode(j_checkpointing_mode)

    def get_state_backend(self) -> StateBackend:
        if False:
            return 10
        '\n        Gets the state backend that defines how to store and checkpoint state.\n\n        .. seealso:: :func:`set_state_backend`\n\n        :return: The :class:`StateBackend`.\n        '
        j_state_backend = self._j_stream_execution_environment.getStateBackend()
        return _from_j_state_backend(j_state_backend)

    def set_state_backend(self, state_backend: StateBackend) -> 'StreamExecutionEnvironment':
        if False:
            i = 10
            return i + 15
        '\n        Sets the state backend that describes how to store and checkpoint operator state. It\n        defines both which data structures hold state during execution (for example hash tables,\n        RockDB, or other data stores) as well as where checkpointed data will be persisted.\n\n        The :class:`~pyflink.datastream.MemoryStateBackend` for example maintains the state in heap\n        memory, as objects. It is lightweight without extra dependencies, but can checkpoint only\n        small states(some counters).\n\n        In contrast, the :class:`~pyflink.datastream.FsStateBackend` stores checkpoints of the state\n        (also maintained as heap objects) in files. When using a replicated file system (like HDFS,\n        S3, Alluxio, etc) this will guarantee that state is not lost upon failures of\n        individual nodes and that streaming program can be executed highly available and strongly\n        consistent(assuming that Flink is run in high-availability mode).\n\n        The build-in state backend includes:\n            :class:`~pyflink.datastream.MemoryStateBackend`,\n            :class:`~pyflink.datastream.FsStateBackend`\n            and :class:`~pyflink.datastream.RocksDBStateBackend`.\n\n        .. seealso:: :func:`get_state_backend`\n\n        Example:\n        ::\n\n            >>> env.set_state_backend(EmbeddedRocksDBStateBackend())\n\n        :param state_backend: The :class:`StateBackend`.\n        :return: This object.\n        '
        self._j_stream_execution_environment = self._j_stream_execution_environment.setStateBackend(state_backend._j_state_backend)
        return self

    def enable_changelog_state_backend(self, enabled: bool) -> 'StreamExecutionEnvironment':
        if False:
            print('Hello World!')
        "\n        Enable the change log for current state backend. This change log allows operators to persist\n        state changes in a very fine-grained manner. Currently, the change log only applies to keyed\n        state, so non-keyed operator state and channel state are persisted as usual. The 'state'\n        here refers to 'keyed state'. Details are as follows:\n\n        * Stateful operators write the state changes to that log (logging the state), in addition         to applying them to the state tables in RocksDB or the in-mem Hashtable.\n        * An operator can acknowledge a checkpoint as soon as the changes in the log have reached         the durable checkpoint storage.\n        * The state tables are persisted periodically, independent of the checkpoints. We call         this the materialization of the state on the checkpoint storage.\n        * Once the state is materialized on checkpoint storage, the state changelog can be         truncated to the corresponding point.\n\n        It establish a way to drastically reduce the checkpoint interval for streaming\n        applications across state backends. For more details please check the FLIP-158.\n\n        If this method is not called explicitly, it means no preference for enabling the change\n        log. Configs for change log enabling will override in different config levels\n        (job/local/cluster).\n\n        .. seealso:: :func:`is_changelog_state_backend_enabled`\n\n\n        :param enabled: True if enable the change log for state backend explicitly, otherwise\n                        disable the change log.\n        :return: This object.\n\n        .. versionadded:: 1.14.0\n        "
        self._j_stream_execution_environment = self._j_stream_execution_environment.enableChangelogStateBackend(enabled)
        return self

    def is_changelog_state_backend_enabled(self) -> Optional[bool]:
        if False:
            return 10
        '\n        Gets the enable status of change log for state backend.\n\n        .. seealso:: :func:`enable_changelog_state_backend`\n\n        :return: An :class:`Optional[bool]` for the enable status of change log for state backend.\n                 Could be None if user never specify this by calling\n                 :func:`enable_changelog_state_backend`.\n\n        .. versionadded:: 1.14.0\n        '
        j_ternary_boolean = self._j_stream_execution_environment.isChangelogStateBackendEnabled()
        return j_ternary_boolean.getAsBoolean()

    def set_default_savepoint_directory(self, directory: str) -> 'StreamExecutionEnvironment':
        if False:
            i = 10
            return i + 15
        '\n        Sets the default savepoint directory, where savepoints will be written to if none\n        is explicitly provided when triggered.\n\n        Example:\n        ::\n\n            >>> env.set_default_savepoint_directory("hdfs://savepoints")\n\n        :param directory The savepoint directory\n        :return: This object.\n        '
        self._j_stream_execution_environment.setDefaultSavepointDirectory(directory)
        return self

    def get_default_savepoint_directory(self) -> Optional[str]:
        if False:
            return 10
        '\n        Gets the default savepoint directory for this Job.\n        '
        j_path = self._j_stream_execution_environment.getDefaultSavepointDirectory()
        if j_path is None:
            return None
        else:
            return j_path.toString()

    def set_restart_strategy(self, restart_strategy_configuration: RestartStrategyConfiguration):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the restart strategy configuration. The configuration specifies which restart strategy\n        will be used for the execution graph in case of a restart.\n\n        Example:\n        ::\n\n            >>> env.set_restart_strategy(RestartStrategies.no_restart())\n\n        :param restart_strategy_configuration: Restart strategy configuration to be set.\n        :return:\n        '
        self._j_stream_execution_environment.setRestartStrategy(restart_strategy_configuration._j_restart_strategy_configuration)

    def get_restart_strategy(self) -> RestartStrategyConfiguration:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the specified restart strategy configuration.\n\n        :return: The restart strategy configuration to be used.\n        '
        return RestartStrategies._from_j_restart_strategy(self._j_stream_execution_environment.getRestartStrategy())

    def add_default_kryo_serializer(self, type_class_name: str, serializer_class_name: str):
        if False:
            i = 10
            return i + 15
        '\n        Adds a new Kryo default serializer to the Runtime.\n\n        Example:\n        ::\n\n            >>> env.add_default_kryo_serializer("com.aaa.bbb.TypeClass", "com.aaa.bbb.Serializer")\n\n        :param type_class_name: The full-qualified java class name of the types serialized with the\n                                given serializer.\n        :param serializer_class_name: The full-qualified java class name of the serializer to use.\n        '
        type_clz = load_java_class(type_class_name)
        j_serializer_clz = load_java_class(serializer_class_name)
        self._j_stream_execution_environment.addDefaultKryoSerializer(type_clz, j_serializer_clz)

    def register_type_with_kryo_serializer(self, type_class_name: str, serializer_class_name: str):
        if False:
            print('Hello World!')
        '\n        Registers the given Serializer via its class as a serializer for the given type at the\n        KryoSerializer.\n\n        Example:\n        ::\n\n            >>> env.register_type_with_kryo_serializer("com.aaa.bbb.TypeClass",\n            ...                                        "com.aaa.bbb.Serializer")\n\n        :param type_class_name: The full-qualified java class name of the types serialized with\n                                the given serializer.\n        :param serializer_class_name: The full-qualified java class name of the serializer to use.\n        '
        type_clz = load_java_class(type_class_name)
        j_serializer_clz = load_java_class(serializer_class_name)
        self._j_stream_execution_environment.registerTypeWithKryoSerializer(type_clz, j_serializer_clz)

    def register_type(self, type_class_name: str):
        if False:
            i = 10
            return i + 15
        '\n        Registers the given type with the serialization stack. If the type is eventually\n        serialized as a POJO, then the type is registered with the POJO serializer. If the\n        type ends up being serialized with Kryo, then it will be registered at Kryo to make\n        sure that only tags are written.\n\n        Example:\n        ::\n\n            >>> env.register_type("com.aaa.bbb.TypeClass")\n\n        :param type_class_name: The full-qualified java class name of the type to register.\n        '
        type_clz = load_java_class(type_class_name)
        self._j_stream_execution_environment.registerType(type_clz)

    def set_stream_time_characteristic(self, characteristic: TimeCharacteristic):
        if False:
            print('Hello World!')
        '\n        Sets the time characteristic for all streams create from this environment, e.g., processing\n        time, event time, or ingestion time.\n\n        If you set the characteristic to IngestionTime of EventTime this will set a default\n        watermark update interval of 200 ms. If this is not applicable for your application\n        you should change it using\n        :func:`pyflink.common.ExecutionConfig.set_auto_watermark_interval`.\n\n        Example:\n        ::\n\n            >>> env.set_stream_time_characteristic(TimeCharacteristic.EventTime)\n\n        :param characteristic: The time characteristic, which could be\n                               :data:`TimeCharacteristic.ProcessingTime`,\n                               :data:`TimeCharacteristic.IngestionTime`,\n                               :data:`TimeCharacteristic.EventTime`.\n        '
        j_characteristic = TimeCharacteristic._to_j_time_characteristic(characteristic)
        self._j_stream_execution_environment.setStreamTimeCharacteristic(j_characteristic)

    def get_stream_time_characteristic(self) -> 'TimeCharacteristic':
        if False:
            i = 10
            return i + 15
        '\n        Gets the time characteristic.\n\n        .. seealso:: :func:`set_stream_time_characteristic`\n\n        :return: The :class:`TimeCharacteristic`.\n        '
        j_characteristic = self._j_stream_execution_environment.getStreamTimeCharacteristic()
        return TimeCharacteristic._from_j_time_characteristic(j_characteristic)

    def configure(self, configuration: Configuration):
        if False:
            print('Hello World!')
        '\n        Sets all relevant options contained in the :class:`~pyflink.common.Configuration`. such as\n        e.g. `pipeline.time-characteristic`. It will reconfigure\n        :class:`~pyflink.datastream.StreamExecutionEnvironment`,\n        :class:`~pyflink.common.ExecutionConfig` and :class:`~pyflink.datastream.CheckpointConfig`.\n\n        It will change the value of a setting only if a corresponding option was set in the\n        `configuration`. If a key is not present, the current value of a field will remain\n        untouched.\n\n        :param configuration: a configuration to read the values from.\n\n        .. versionadded:: 1.15.0\n        '
        self._j_stream_execution_environment.configure(configuration._j_configuration, get_gateway().jvm.Thread.currentThread().getContextClassLoader())

    def add_python_file(self, file_path: str):
        if False:
            print('Hello World!')
        '\n        Adds a python dependency which could be python files, python packages or\n        local directories. They will be added to the PYTHONPATH of the python UDF worker.\n        Please make sure that these dependencies can be imported.\n\n        :param file_path: The path of the python dependency.\n        '
        jvm = get_gateway().jvm
        env_config = jvm.org.apache.flink.python.util.PythonConfigUtil.getEnvironmentConfig(self._j_stream_execution_environment)
        python_files = env_config.getString(jvm.PythonOptions.PYTHON_FILES.key(), None)
        if python_files is not None:
            python_files = jvm.PythonDependencyUtils.FILE_DELIMITER.join([file_path, python_files])
        else:
            python_files = file_path
        env_config.setString(jvm.PythonOptions.PYTHON_FILES.key(), python_files)

    def set_python_requirements(self, requirements_file_path: str, requirements_cache_dir: str=None):
        if False:
            return 10
        '\n        Specifies a requirements.txt file which defines the third-party dependencies.\n        These dependencies will be installed to a temporary directory and added to the\n        PYTHONPATH of the python UDF worker.\n\n        For the dependencies which could not be accessed in the cluster, a directory which contains\n        the installation packages of these dependencies could be specified using the parameter\n        "requirements_cached_dir". It will be uploaded to the cluster to support offline\n        installation.\n\n        Example:\n        ::\n\n            # commands executed in shell\n            $ echo numpy==1.16.5 > requirements.txt\n            $ pip download -d cached_dir -r requirements.txt --no-binary :all:\n\n            # python code\n            >>> stream_env.set_python_requirements("requirements.txt", "cached_dir")\n\n        .. note::\n\n            Please make sure the installation packages matches the platform of the cluster\n            and the python version used. These packages will be installed using pip,\n            so also make sure the version of Pip (version >= 20.3) and the version of\n            SetupTools (version >= 37.0.0).\n\n        :param requirements_file_path: The path of "requirements.txt" file.\n        :param requirements_cache_dir: The path of the local directory which contains the\n                                       installation packages.\n        '
        jvm = get_gateway().jvm
        python_requirements = requirements_file_path
        if requirements_cache_dir is not None:
            python_requirements = jvm.PythonDependencyUtils.PARAM_DELIMITER.join([python_requirements, requirements_cache_dir])
        env_config = jvm.org.apache.flink.python.util.PythonConfigUtil.getEnvironmentConfig(self._j_stream_execution_environment)
        env_config.setString(jvm.PythonOptions.PYTHON_REQUIREMENTS.key(), python_requirements)

    def add_python_archive(self, archive_path: str, target_dir: str=None):
        if False:
            i = 10
            return i + 15
        '\n        Adds a python archive file. The file will be extracted to the working directory of\n        python UDF worker.\n\n        If the parameter "target_dir" is specified, the archive file will be extracted to a\n        directory named ${target_dir}. Otherwise, the archive file will be extracted to a\n        directory with the same name of the archive file.\n\n        If python UDF depends on a specific python version which does not exist in the cluster,\n        this method can be used to upload the virtual environment.\n        Note that the path of the python interpreter contained in the uploaded environment\n        should be specified via the method :func:`pyflink.table.TableConfig.set_python_executable`.\n\n        The files uploaded via this method are also accessible in UDFs via relative path.\n\n        Example:\n        ::\n\n            # command executed in shell\n            # assert the relative path of python interpreter is py_env/bin/python\n            $ zip -r py_env.zip py_env\n\n            # python code\n            >>> stream_env.add_python_archive("py_env.zip")\n            >>> stream_env.set_python_executable("py_env.zip/py_env/bin/python")\n\n            # or\n            >>> stream_env.add_python_archive("py_env.zip", "myenv")\n            >>> stream_env.set_python_executable("myenv/py_env/bin/python")\n\n            # the files contained in the archive file can be accessed in UDF\n            >>> def my_udf():\n            ...     with open("myenv/py_env/data/data.txt") as f:\n            ...         ...\n\n        .. note::\n\n            Please make sure the uploaded python environment matches the platform that the cluster\n            is running on and that the python version must be 3.6 or higher.\n\n        .. note::\n\n            Currently only zip-format is supported. i.e. zip, jar, whl, egg, etc.\n            The other archive formats such as tar, tar.gz, 7z, rar, etc are not supported.\n\n        :param archive_path: The archive file path.\n        :param target_dir: Optional, the target dir name that the archive file extracted to.\n        '
        jvm = get_gateway().jvm
        if target_dir is not None:
            archive_path = jvm.PythonDependencyUtils.PARAM_DELIMITER.join([archive_path, target_dir])
        env_config = jvm.org.apache.flink.python.util.PythonConfigUtil.getEnvironmentConfig(self._j_stream_execution_environment)
        python_archives = env_config.getString(jvm.PythonOptions.PYTHON_ARCHIVES.key(), None)
        if python_archives is not None:
            python_files = jvm.PythonDependencyUtils.FILE_DELIMITER.join([python_archives, archive_path])
        else:
            python_files = archive_path
        env_config.setString(jvm.PythonOptions.PYTHON_ARCHIVES.key(), python_files)

    def set_python_executable(self, python_exec: str):
        if False:
            return 10
        '\n        Sets the path of the python interpreter which is used to execute the python udf workers.\n\n        e.g. "/usr/local/bin/python3".\n\n        If python UDF depends on a specific python version which does not exist in the cluster,\n        the method :func:`pyflink.datastream.StreamExecutionEnvironment.add_python_archive` can be\n        used to upload a virtual environment. The path of the python interpreter contained in the\n        uploaded environment can be specified via this method.\n\n        Example:\n        ::\n\n            # command executed in shell\n            # assume that the relative path of python interpreter is py_env/bin/python\n            $ zip -r py_env.zip py_env\n\n            # python code\n            >>> stream_env.add_python_archive("py_env.zip")\n            >>> stream_env.set_python_executable("py_env.zip/py_env/bin/python")\n\n        .. note::\n\n            Please make sure the uploaded python environment matches the platform that the cluster\n            is running on and that the python version must be 3.7 or higher.\n\n        .. note::\n\n            The python udf worker depends on Apache Beam (version == 2.43.0).\n            Please ensure that the specified environment meets the above requirements.\n\n        :param python_exec: The path of python interpreter.\n        '
        jvm = get_gateway().jvm
        env_config = jvm.org.apache.flink.python.util.PythonConfigUtil.getEnvironmentConfig(self._j_stream_execution_environment)
        env_config.setString(jvm.PythonOptions.PYTHON_EXECUTABLE.key(), python_exec)

    def add_jars(self, *jars_path: str):
        if False:
            while True:
                i = 10
        '\n        Adds a list of jar files that will be uploaded to the cluster and referenced by the job.\n\n        :param jars_path: Path of jars.\n        '
        add_jars_to_context_class_loader(jars_path)
        jvm = get_gateway().jvm
        jars_key = jvm.org.apache.flink.configuration.PipelineOptions.JARS.key()
        env_config = jvm.org.apache.flink.python.util.PythonConfigUtil.getEnvironmentConfig(self._j_stream_execution_environment)
        old_jar_paths = env_config.getString(jars_key, None)
        joined_jars_path = ';'.join(jars_path)
        if old_jar_paths and old_jar_paths.strip():
            joined_jars_path = ';'.join([old_jar_paths, joined_jars_path])
        env_config.setString(jars_key, joined_jars_path)

    def add_classpaths(self, *classpaths: str):
        if False:
            print('Hello World!')
        '\n        Adds a list of URLs that are added to the classpath of each user code classloader of the\n        program. Paths must specify a protocol (e.g. file://) and be accessible on all nodes\n\n        :param classpaths: Classpaths that will be added.\n        '
        add_jars_to_context_class_loader(classpaths)
        jvm = get_gateway().jvm
        classpaths_key = jvm.org.apache.flink.configuration.PipelineOptions.CLASSPATHS.key()
        env_config = jvm.org.apache.flink.python.util.PythonConfigUtil.getEnvironmentConfig(self._j_stream_execution_environment)
        old_classpaths = env_config.getString(classpaths_key, None)
        joined_classpaths = ';'.join(list(classpaths))
        if old_classpaths and old_classpaths.strip():
            joined_classpaths = ';'.join([old_classpaths, joined_classpaths])
        env_config.setString(classpaths_key, joined_classpaths)

    def get_default_local_parallelism(self) -> int:
        if False:
            print('Hello World!')
        '\n        Gets the default parallelism that will be used for the local execution environment.\n\n        :return: The default local parallelism.\n        '
        return self._j_stream_execution_environment.getDefaultLocalParallelism()

    def set_default_local_parallelism(self, parallelism: int):
        if False:
            print('Hello World!')
        '\n        Sets the default parallelism that will be used for the local execution environment.\n\n        :param parallelism: The parallelism to use as the default local parallelism.\n        '
        self._j_stream_execution_environment.setDefaultLocalParallelism(parallelism)

    def execute(self, job_name: str=None) -> JobExecutionResult:
        if False:
            while True:
                i = 10
        '\n        Triggers the program execution. The environment will execute all parts of\n        the program that have resulted in a "sink" operation. Sink operations are\n        for example printing results or forwarding them to a message queue.\n\n        The program execution will be logged and displayed with the provided name\n\n        :param job_name: Desired name of the job, optional.\n        :return: The result of the job execution, containing elapsed time and accumulators.\n        '
        j_stream_graph = self._generate_stream_graph(clear_transformations=True, job_name=job_name)
        return JobExecutionResult(self._j_stream_execution_environment.execute(j_stream_graph))

    def execute_async(self, job_name: str='Flink Streaming Job') -> JobClient:
        if False:
            print('Hello World!')
        '\n        Triggers the program asynchronously. The environment will execute all parts of the program\n        that have resulted in a "sink" operation. Sink operations are for example printing results\n        or forwarding them to a message queue.\n        The program execution will be logged and displayed with a generated default name.\n\n        :param job_name: Desired name of the job.\n        :return: A JobClient that can be used to communicate with the submitted job, completed on\n                 submission succeeded.\n        '
        j_stream_graph = self._generate_stream_graph(clear_transformations=True, job_name=job_name)
        j_job_client = self._j_stream_execution_environment.executeAsync(j_stream_graph)
        return JobClient(j_job_client=j_job_client)

    def get_execution_plan(self) -> str:
        if False:
            print('Hello World!')
        '\n        Creates the plan with which the system will execute the program, and returns it as\n        a String using a JSON representation of the execution data flow graph.\n        Note that this needs to be called, before the plan is executed.\n\n        If the compiler could not be instantiated, or the master could not\n        be contacted to retrieve information relevant to the execution planning,\n        an exception will be thrown.\n\n        :return: The execution plan of the program, as a JSON String.\n        '
        j_stream_graph = self._generate_stream_graph(False)
        return j_stream_graph.getStreamingPlanAsJSON()

    def register_cached_file(self, file_path: str, name: str, executable: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers a file at the distributed cache under the given name. The file will be accessible\n        from any user-defined function in the (distributed) runtime under a local path. Files may be\n        local files (which will be distributed via BlobServer), or files in a distributed file\n        system. The runtime will copy the files temporarily to a local cache, if needed.\n\n        :param file_path: The path of the file, as a URI (e.g. "file:///some/path" or\n                         hdfs://host:port/and/path").\n        :param name: The name under which the file is registered.\n        :param executable: Flag indicating whether the file should be executable.\n\n        .. versionadded:: 1.16.0\n        '
        self._j_stream_execution_environment.registerCachedFile(file_path, name, executable)

    @staticmethod
    def get_execution_environment(configuration: Configuration=None) -> 'StreamExecutionEnvironment':
        if False:
            return 10
        '\n        Creates an execution environment that represents the context in which the\n        program is currently executed. If the program is invoked standalone, this\n        method returns a local execution environment.\n\n        When executed from the command line the given configuration is stacked on top of the\n        global configuration which comes from the flink-conf.yaml, potentially overriding\n        duplicated options.\n\n        :param configuration: The configuration to instantiate the environment with.\n        :return: The execution environment of the context in which the program is executed.\n        '
        gateway = get_gateway()
        JStreamExecutionEnvironment = gateway.jvm.org.apache.flink.streaming.api.environment.StreamExecutionEnvironment
        if configuration:
            j_stream_exection_environment = JStreamExecutionEnvironment.getExecutionEnvironment(configuration._j_configuration)
        else:
            j_stream_exection_environment = JStreamExecutionEnvironment.getExecutionEnvironment()
        return StreamExecutionEnvironment(j_stream_exection_environment)

    def create_input(self, input_format: InputFormat, type_info: Optional[TypeInformation]=None):
        if False:
            print('Hello World!')
        "\n        Create an input data stream with InputFormat.\n\n        If the input_format needs a well-defined type information (e.g. Avro's generic record), you\n        can either explicitly use type_info argument or use InputFormats implementing\n        ResultTypeQueryable.\n\n        :param input_format: The input format to read from.\n        :param type_info: Optional type information to explicitly declare output type.\n\n        .. versionadded:: 1.16.0\n        "
        input_type_info = type_info
        if input_type_info is None and isinstance(input_format, ResultTypeQueryable):
            input_type_info = cast(ResultTypeQueryable, input_format).get_produced_type()
        if input_type_info is None:
            j_data_stream = self._j_stream_execution_environment.createInput(input_format.get_java_object())
        else:
            j_data_stream = self._j_stream_execution_environment.createInput(input_format.get_java_object(), input_type_info.get_java_type_info())
        return DataStream(j_data_stream=j_data_stream)

    def add_source(self, source_func: SourceFunction, source_name: str='Custom Source', type_info: TypeInformation=None) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a data source to the streaming topology.\n\n        :param source_func: the user defined function.\n        :param source_name: name of the data source. Optional.\n        :param type_info: type of the returned stream. Optional.\n        :return: the data stream constructed.\n        '
        if type_info:
            j_type_info = type_info.get_java_type_info()
        else:
            j_type_info = None
        j_data_stream = self._j_stream_execution_environment.addSource(source_func.get_java_function(), source_name, j_type_info)
        return DataStream(j_data_stream=j_data_stream)

    def from_source(self, source: Source, watermark_strategy: WatermarkStrategy, source_name: str, type_info: TypeInformation=None) -> 'DataStream':
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a data :class:`~pyflink.datastream.connectors.Source` to the environment to get a\n        :class:`~pyflink.datastream.DataStream`.\n\n        The result will be either a bounded data stream (that can be processed in a batch way) or\n        an unbounded data stream (that must be processed in a streaming way), based on the\n        boundedness property of the source.\n\n        This method takes an explicit type information for the produced data stream, so that\n        callers can define directly what type/serializer will be used for the produced stream. For\n        sources that describe their produced type, the parameter type_info should not be specified\n        to avoid specifying the produced type redundantly.\n\n        .. versionadded:: 1.13.0\n        '
        if type_info:
            j_type_info = type_info.get_java_type_info()
        else:
            j_type_info = None
        j_data_stream = self._j_stream_execution_environment.fromSource(source.get_java_function(), watermark_strategy._j_watermark_strategy, source_name, j_type_info)
        return DataStream(j_data_stream=j_data_stream)

    def read_text_file(self, file_path: str, charset_name: str='UTF-8') -> DataStream:
        if False:
            i = 10
            return i + 15
        '\n        Reads the given file line-by-line and creates a DataStream that contains a string with the\n        contents of each such line. The charset with the given name will be used to read the files.\n\n        Note that this interface is not fault tolerant that is supposed to be used for test purpose.\n\n        :param file_path: The path of the file, as a URI (e.g., "file:///some/local/file" or\n                          "hdfs://host:port/file/path")\n        :param charset_name: The name of the character set used to read the file.\n        :return: The DataStream that represents the data read from the given file as text lines.\n        '
        return DataStream(self._j_stream_execution_environment.readTextFile(file_path, charset_name))

    def from_collection(self, collection: List[Any], type_info: TypeInformation=None) -> DataStream:
        if False:
            while True:
                i = 10
        '\n        Creates a data stream from the given non-empty collection. The type of the data stream is\n        that of the elements in the collection.\n\n        Note that this operation will result in a non-parallel data stream source, i.e. a data\n        stream source with parallelism one.\n\n        :param collection: The collection of elements to create the data stream from.\n        :param type_info: The TypeInformation for the produced data stream\n        :return: the data stream representing the given collection.\n        '
        if type_info is not None:
            collection = [type_info.to_internal_type(element) for element in collection]
        return self._from_collection(collection, type_info)

    def _from_collection(self, elements: List[Any], type_info: TypeInformation=None) -> DataStream:
        if False:
            for i in range(10):
                print('nop')
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=tempfile.mkdtemp())
        serializer = self.serializer
        try:
            with temp_file:
                serializer.serialize(elements, temp_file)
            gateway = get_gateway()
            if type_info is None:
                j_objs = gateway.jvm.PythonBridgeUtils.readPickledBytes(temp_file.name)
                out_put_type_info = Types.PICKLED_BYTE_ARRAY()
            else:
                j_objs = gateway.jvm.PythonBridgeUtils.readPythonObjects(temp_file.name)
                out_put_type_info = type_info
            PythonTypeUtils = gateway.jvm.org.apache.flink.streaming.api.utils.PythonTypeUtils
            execution_config = self._j_stream_execution_environment.getConfig()
            j_input_format = PythonTypeUtils.getCollectionInputFormat(j_objs, out_put_type_info.get_java_type_info(), execution_config)
            JInputFormatSourceFunction = gateway.jvm.org.apache.flink.streaming.api.functions.source.InputFormatSourceFunction
            JBoundedness = gateway.jvm.org.apache.flink.api.connector.source.Boundedness
            j_data_stream_source = invoke_method(self._j_stream_execution_environment, 'org.apache.flink.streaming.api.environment.StreamExecutionEnvironment', 'addSource', [JInputFormatSourceFunction(j_input_format, out_put_type_info.get_java_type_info()), 'Collection Source', out_put_type_info.get_java_type_info(), JBoundedness.BOUNDED], ['org.apache.flink.streaming.api.functions.source.SourceFunction', 'java.lang.String', 'org.apache.flink.api.common.typeinfo.TypeInformation', 'org.apache.flink.api.connector.source.Boundedness'])
            j_data_stream_source.forceNonParallel()
            return DataStream(j_data_stream=j_data_stream_source)
        finally:
            os.unlink(temp_file.name)

    def _generate_stream_graph(self, clear_transformations: bool=False, job_name: str=None) -> JavaObject:
        if False:
            i = 10
            return i + 15
        gateway = get_gateway()
        JPythonConfigUtil = gateway.jvm.org.apache.flink.python.util.PythonConfigUtil
        JPythonConfigUtil.configPythonOperator(self._j_stream_execution_environment)
        gateway.jvm.org.apache.flink.python.chain.PythonOperatorChainingOptimizer.apply(self._j_stream_execution_environment)
        JPythonConfigUtil.setPartitionCustomOperatorNumPartitions(get_field_value(self._j_stream_execution_environment, 'transformations'))
        j_stream_graph = self._j_stream_execution_environment.getStreamGraph(clear_transformations)
        if job_name is not None:
            j_stream_graph.setJobName(job_name)
        return j_stream_graph

    def _open(self):
        if False:
            for i in range(10):
                print('nop')
        j_configuration = get_j_env_configuration(self._j_stream_execution_environment)

        def startup_loopback_server():
            if False:
                for i in range(10):
                    print('nop')
            from pyflink.common import Configuration
            from pyflink.fn_execution.beam.beam_worker_pool_service import BeamFnLoopbackWorkerPoolServicer
            config = Configuration(j_configuration=j_configuration)
            config.set_string('python.loopback-server.address', BeamFnLoopbackWorkerPoolServicer().start())
        python_worker_execution_mode = os.environ.get('_python_worker_execution_mode')
        if python_worker_execution_mode is None:
            if is_local_deployment(j_configuration):
                startup_loopback_server()
        elif python_worker_execution_mode == 'loopback':
            if is_local_deployment(j_configuration):
                startup_loopback_server()
            else:
                raise ValueError("Loopback mode is enabled, however the job wasn't configured to run in local deployment mode")
        elif python_worker_execution_mode != 'process':
            raise ValueError("It only supports to execute the Python worker in 'loopback' mode and 'process' mode, unknown mode '%s' is configured" % python_worker_execution_mode)

    def is_unaligned_checkpoints_enabled(self):
        if False:
            return 10
        '\n        Returns whether Unaligned Checkpoints are enabled.\n        '
        return self._j_stream_execution_environment.isUnalignedCheckpointsEnabled()

    def is_force_unaligned_checkpoints(self):
        if False:
            print('Hello World!')
        '\n        Returns whether Unaligned Checkpoints are force-enabled.\n        '
        return self._j_stream_execution_environment.isForceUnalignedCheckpoints()

    def close(self):
        if False:
            while True:
                i = 10
        '\n        Close and clean up the execution environment. All the cached intermediate results will be\n        released physically.\n\n        .. versionadded:: 1.16.0\n        '
        self._j_stream_execution_environment.close()