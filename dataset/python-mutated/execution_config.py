import warnings
from typing import Dict, List
from pyflink.common.execution_mode import ExecutionMode
from pyflink.common.input_dependency_constraint import InputDependencyConstraint
from pyflink.common.restart_strategy import RestartStrategies, RestartStrategyConfiguration
from pyflink.java_gateway import get_gateway
from pyflink.util.java_utils import load_java_class
__all__ = ['ExecutionConfig']

class ExecutionConfig(object):
    """
    A config to define the behavior of the program execution. It allows to define (among other
    options) the following settings:

    - The default parallelism of the program, i.e., how many parallel tasks to use for
      all functions that do not define a specific value directly.

    - The number of retries in the case of failed executions.

    - The delay between execution retries.

    - The :class:`ExecutionMode` of the program: Batch or Pipelined.
      The default execution mode is :data:`ExecutionMode.PIPELINED`

    - Enabling or disabling the "closure cleaner". The closure cleaner pre-processes
      the implementations of functions. In case they are (anonymous) inner classes,
      it removes unused references to the enclosing class to fix certain serialization-related
      problems and to reduce the size of the closure.

    - The config allows to register types and serializers to increase the efficiency of
      handling *generic types* and *POJOs*. This is usually only needed
      when the functions return not only the types declared in their signature, but
      also subclasses of those types.

    :data:`PARALLELISM_DEFAULT`:

    The flag value indicating use of the default parallelism. This value can
    be used to reset the parallelism back to the default state.

    :data:`PARALLELISM_UNKNOWN`:

    The flag value indicating an unknown or unset parallelism. This value is
    not a valid parallelism and indicates that the parallelism should remain
    unchanged.
    """
    PARALLELISM_DEFAULT = -1
    PARALLELISM_UNKNOWN = -2

    def __init__(self, j_execution_config):
        if False:
            i = 10
            return i + 15
        self._j_execution_config = j_execution_config

    def enable_closure_cleaner(self) -> 'ExecutionConfig':
        if False:
            print('Hello World!')
        '\n        Enables the ClosureCleaner. This analyzes user code functions and sets fields to null\n        that are not used. This will in most cases make closures or anonymous inner classes\n        serializable that where not serializable due to some Scala or Java implementation artifact.\n        User code must be serializable because it needs to be sent to worker nodes.\n\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.enableClosureCleaner()
        return self

    def disable_closure_cleaner(self) -> 'ExecutionConfig':
        if False:
            print('Hello World!')
        '\n        Disables the ClosureCleaner.\n\n        .. seealso:: :func:`enable_closure_cleaner`\n\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.disableClosureCleaner()
        return self

    def is_closure_cleaner_enabled(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Returns whether the ClosureCleaner is enabled.\n\n        .. seealso:: :func:`enable_closure_cleaner`\n\n        :return: ``True`` means enable and ``False`` means disable.\n        '
        return self._j_execution_config.isClosureCleanerEnabled()

    def set_auto_watermark_interval(self, interval: int) -> 'ExecutionConfig':
        if False:
            print('Hello World!')
        '\n        Sets the interval of the automatic watermark emission. Watermarks are used throughout\n        the streaming system to keep track of the progress of time. They are used, for example,\n        for time based windowing.\n\n        :param interval: The integer value interval between watermarks in milliseconds.\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.setAutoWatermarkInterval(interval)
        return self

    def get_auto_watermark_interval(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the interval of the automatic watermark emission.\n\n        .. seealso:: :func:`set_auto_watermark_interval`\n\n        :return: The integer value interval in milliseconds of the automatic watermark emission.\n        '
        return self._j_execution_config.getAutoWatermarkInterval()

    def set_latency_tracking_interval(self, interval: int) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Interval for sending latency tracking marks from the sources to the sinks.\n\n        Flink will send latency tracking marks from the sources at the specified interval.\n        Setting a tracking interval <= 0 disables the latency tracking.\n\n        :param interval: Integer value interval in milliseconds.\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.setLatencyTrackingInterval(interval)
        return self

    def get_latency_tracking_interval(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the latency tracking interval.\n\n        :return: The latency tracking interval in milliseconds.\n        '
        return self._j_execution_config.getLatencyTrackingInterval()

    def get_parallelism(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        "\n        Gets the parallelism with which operation are executed by default. Operations can\n        individually override this value to use a specific parallelism.\n\n        Other operations may need to run with a different parallelism - for example calling\n        a reduce operation over the entire data set will involve an operation that runs\n        with a parallelism of one (the final reduce to the single result value).\n\n        :return: The parallelism used by operations, unless they override that value. This method\n                 returns :data:`ExecutionConfig.PARALLELISM_DEFAULT` if the environment's default\n                 parallelism should be used.\n        "
        return self._j_execution_config.getParallelism()

    def set_parallelism(self, parallelism: int) -> 'ExecutionConfig':
        if False:
            return 10
        '\n        Sets the parallelism for operations executed through this environment.\n        Setting a parallelism of x here will cause all operators (such as join, map, reduce) to run\n        with x parallel instances.\n\n        This method overrides the default parallelism for this environment.\n        The local execution environment uses by default a value equal to the number of hardware\n        contexts (CPU cores / threads). When executing the program via the command line client\n        from a JAR/Python file, the default parallelism is the one configured for that setup.\n\n        :param parallelism: The parallelism to use.\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.setParallelism(parallelism)
        return self

    def get_max_parallelism(self) -> int:
        if False:
            print('Hello World!')
        '\n        Gets the maximum degree of parallelism defined for the program.\n\n        The maximum degree of parallelism specifies the upper limit for dynamic scaling. It also\n        defines the number of key groups used for partitioned state.\n\n        :return: Maximum degree of parallelism.\n        '
        return self._j_execution_config.getMaxParallelism()

    def set_max_parallelism(self, max_parallelism: int) -> 'ExecutionConfig':
        if False:
            i = 10
            return i + 15
        '\n        Sets the maximum degree of parallelism defined for the program.\n\n        The maximum degree of parallelism specifies the upper limit for dynamic scaling. It also\n        defines the number of key groups used for partitioned state.\n\n        :param max_parallelism: Maximum degree of parallelism to be used for the program.\n        '
        self._j_execution_config.setMaxParallelism(max_parallelism)
        return self

    def get_task_cancellation_interval(self) -> int:
        if False:
            return 10
        '\n        Gets the interval (in milliseconds) between consecutive attempts to cancel a running task.\n\n        :return: The integer value interval in milliseconds.\n        '
        return self._j_execution_config.getTaskCancellationInterval()

    def set_task_cancellation_interval(self, interval: int) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the configuration parameter specifying the interval (in milliseconds)\n        between consecutive attempts to cancel a running task.\n\n        :param interval: The integer value interval in milliseconds.\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.setTaskCancellationInterval(interval)
        return self

    def get_task_cancellation_timeout(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Returns the timeout (in milliseconds) after which an ongoing task\n        cancellation leads to a fatal TaskManager error.\n\n        The value ``0`` means that the timeout is disabled. In\n        this case a stuck cancellation will not lead to a fatal error.\n\n        :return: The timeout in milliseconds.\n        '
        return self._j_execution_config.getTaskCancellationTimeout()

    def set_task_cancellation_timeout(self, timeout: int) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the timeout (in milliseconds) after which an ongoing task cancellation\n        is considered failed, leading to a fatal TaskManager error.\n\n        The cluster default is configured via ``TaskManagerOptions#TASK_CANCELLATION_TIMEOUT``.\n\n        The value ``0`` disables the timeout. In this case a stuck\n        cancellation will not lead to a fatal error.\n\n        :param timeout: The task cancellation timeout (in milliseconds).\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.setTaskCancellationTimeout(timeout)
        return self

    def set_restart_strategy(self, restart_strategy_configuration: RestartStrategyConfiguration) -> 'ExecutionConfig':
        if False:
            print('Hello World!')
        '\n        Sets the restart strategy to be used for recovery.\n        ::\n\n            >>> config = env.get_config()\n            >>> config.set_restart_strategy(RestartStrategies.fixed_delay_restart(10, 1000))\n\n        The restart strategy configurations are all created from :class:`RestartStrategies`.\n\n        :param restart_strategy_configuration: Configuration defining the restart strategy to use.\n        '
        self._j_execution_config.setRestartStrategy(restart_strategy_configuration._j_restart_strategy_configuration)
        return self

    def get_restart_strategy(self) -> RestartStrategyConfiguration:
        if False:
            return 10
        '\n        Returns the restart strategy which has been set for the current job.\n\n        .. seealso:: :func:`set_restart_strategy`\n\n        :return: The specified restart configuration.\n        '
        return RestartStrategies._from_j_restart_strategy(self._j_execution_config.getRestartStrategy())

    def set_execution_mode(self, execution_mode: ExecutionMode) -> 'ExecutionConfig':
        if False:
            i = 10
            return i + 15
        '\n        Sets the execution mode to execute the program. The execution mode defines whether\n        data exchanges are performed in a batch or on a pipelined manner.\n\n        The default execution mode is :data:`ExecutionMode.PIPELINED`.\n\n        Example:\n        ::\n\n            >>> config.set_execution_mode(ExecutionMode.BATCH)\n\n        :param execution_mode: The execution mode to use. The execution mode could be\n                               :data:`ExecutionMode.PIPELINED`,\n                               :data:`ExecutionMode.PIPELINED_FORCED`,\n                               :data:`ExecutionMode.BATCH` or\n                               :data:`ExecutionMode.BATCH_FORCED`.\n        '
        self._j_execution_config.setExecutionMode(execution_mode._to_j_execution_mode())
        return self

    def get_execution_mode(self) -> 'ExecutionMode':
        if False:
            return 10
        '\n        Gets the execution mode used to execute the program. The execution mode defines whether\n        data exchanges are performed in a batch or on a pipelined manner.\n\n        The default execution mode is :data:`ExecutionMode.PIPELINED`.\n\n        .. seealso:: :func:`set_execution_mode`\n\n        :return: The execution mode for the program.\n        '
        j_execution_mode = self._j_execution_config.getExecutionMode()
        return ExecutionMode._from_j_execution_mode(j_execution_mode)

    def set_default_input_dependency_constraint(self, input_dependency_constraint: InputDependencyConstraint) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the default input dependency constraint for vertex scheduling. It indicates when a\n        task should be scheduled considering its inputs status.\n\n        The default constraint is :data:`InputDependencyConstraint.ANY`.\n\n        Example:\n        ::\n\n            >>> config.set_default_input_dependency_constraint(InputDependencyConstraint.ALL)\n\n        :param input_dependency_constraint: The input dependency constraint. The constraints could\n                                            be :data:`InputDependencyConstraint.ANY` or\n                                            :data:`InputDependencyConstraint.ALL`.\n\n        .. note:: Deprecated in 1.13. :class:`InputDependencyConstraint` is not used anymore in the\n                  current scheduler implementations.\n        '
        warnings.warn('Deprecated in 1.13. InputDependencyConstraint is not used anywhere. Therefore, the method call set_default_input_dependency_constraint is obsolete.', DeprecationWarning)
        self._j_execution_config.setDefaultInputDependencyConstraint(input_dependency_constraint._to_j_input_dependency_constraint())
        return self

    def get_default_input_dependency_constraint(self) -> 'InputDependencyConstraint':
        if False:
            print('Hello World!')
        '\n        Gets the default input dependency constraint for vertex scheduling. It indicates when a\n        task should be scheduled considering its inputs status.\n\n        The default constraint is :data:`InputDependencyConstraint.ANY`.\n\n        .. seealso:: :func:`set_default_input_dependency_constraint`\n\n        :return: The input dependency constraint of this job. The possible constraints are\n                 :data:`InputDependencyConstraint.ANY` and :data:`InputDependencyConstraint.ALL`.\n\n        .. note:: Deprecated in 1.13. :class:`InputDependencyConstraint` is not used anymore in the\n                  current scheduler implementations.\n        '
        warnings.warn('Deprecated in 1.13. InputDependencyConstraint is not used anywhere. Therefore, the method call get_default_input_dependency_constraint is obsolete.', DeprecationWarning)
        j_input_dependency_constraint = self._j_execution_config.getDefaultInputDependencyConstraint()
        return InputDependencyConstraint._from_j_input_dependency_constraint(j_input_dependency_constraint)

    def enable_force_kryo(self) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Force TypeExtractor to use Kryo serializer for POJOS even though we could analyze as POJO.\n        In some cases this might be preferable. For example, when using interfaces\n        with subclasses that cannot be analyzed as POJO.\n        '
        self._j_execution_config.enableForceKryo()
        return self

    def disable_force_kryo(self) -> 'ExecutionConfig':
        if False:
            while True:
                i = 10
        '\n        Disable use of Kryo serializer for all POJOs.\n        '
        self._j_execution_config.disableForceKryo()
        return self

    def is_force_kryo_enabled(self) -> bool:
        if False:
            print('Hello World!')
        '\n        :return: Boolean value that represent whether the usage of Kryo serializer for all POJOs\n                 is enabled.\n        '
        return self._j_execution_config.isForceKryoEnabled()

    def enable_generic_types(self) -> 'ExecutionConfig':
        if False:
            print('Hello World!')
        '\n        Enables the use generic types which are serialized via Kryo.\n\n        Generic types are enabled by default.\n\n        .. seealso:: :func:`disable_generic_types`\n        '
        self._j_execution_config.enableGenericTypes()
        return self

    def disable_generic_types(self) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Disables the use of generic types (types that would be serialized via Kryo). If this option\n        is used, Flink will throw an ``UnsupportedOperationException`` whenever it encounters\n        a data type that would go through Kryo for serialization.\n\n        Disabling generic types can be helpful to eagerly find and eliminate the use of types\n        that would go through Kryo serialization during runtime. Rather than checking types\n        individually, using this option will throw exceptions eagerly in the places where generic\n        types are used.\n\n        **Important:** We recommend to use this option only during development and pre-production\n        phases, not during actual production use. The application program and/or the input data may\n        be such that new, previously unseen, types occur at some point. In that case, setting this\n        option would cause the program to fail.\n\n        .. seealso:: :func:`enable_generic_types`\n        '
        self._j_execution_config.disableGenericTypes()
        return self

    def has_generic_types_disabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Checks whether generic types are supported. Generic types are types that go through Kryo\n        during serialization.\n\n        Generic types are enabled by default.\n\n        .. seealso:: :func:`enable_generic_types`\n\n        .. seealso:: :func:`disable_generic_types`\n\n        :return: Boolean value that represent whether the generic types are supported.\n        '
        return self._j_execution_config.hasGenericTypesDisabled()

    def enable_auto_generated_uids(self) -> 'ExecutionConfig':
        if False:
            i = 10
            return i + 15
        "\n        Enables the Flink runtime to auto-generate UID's for operators.\n\n        .. seealso:: :func:`disable_auto_generated_uids`\n        "
        self._j_execution_config.enableAutoGeneratedUIDs()
        return self

    def disable_auto_generated_uids(self) -> 'ExecutionConfig':
        if False:
            while True:
                i = 10
        "\n        Disables auto-generated UIDs. Forces users to manually specify UIDs\n        on DataStream applications.\n\n        It is highly recommended that users specify UIDs before deploying to\n        production since they are used to match state in savepoints to operators\n        in a job. Because auto-generated ID's are likely to change when modifying\n        a job, specifying custom IDs allow an application to evolve overtime\n        without discarding state.\n        "
        self._j_execution_config.disableAutoGeneratedUIDs()
        return self

    def has_auto_generated_uids_enabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Checks whether auto generated UIDs are supported.\n\n        Auto generated UIDs are enabled by default.\n\n        .. seealso:: :func:`enable_auto_generated_uids`\n\n        .. seealso:: :func:`disable_auto_generated_uids`\n\n        :return: Boolean value that represent whether auto generated UIDs are supported.\n        '
        return self._j_execution_config.hasAutoGeneratedUIDsEnabled()

    def enable_force_avro(self) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Forces Flink to use the Apache Avro serializer for POJOs.\n\n        **Important:** Make sure to include the *flink-avro* module.\n        '
        self._j_execution_config.enableForceAvro()
        return self

    def disable_force_avro(self) -> 'ExecutionConfig':
        if False:
            return 10
        '\n        Disables the Apache Avro serializer as the forced serializer for POJOs.\n        '
        self._j_execution_config.disableForceAvro()
        return self

    def is_force_avro_enabled(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns whether the Apache Avro is the default serializer for POJOs.\n\n        :return: Boolean value that represent whether the Apache Avro is the default serializer\n                 for POJOs.\n        '
        return self._j_execution_config.isForceAvroEnabled()

    def enable_object_reuse(self) -> 'ExecutionConfig':
        if False:
            while True:
                i = 10
        '\n        Enables reusing objects that Flink internally uses for deserialization and passing\n        data to user-code functions. Keep in mind that this can lead to bugs when the\n        user-code function of an operation is not aware of this behaviour.\n\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.enableObjectReuse()
        return self

    def disable_object_reuse(self) -> 'ExecutionConfig':
        if False:
            for i in range(10):
                print('nop')
        '\n        Disables reusing objects that Flink internally uses for deserialization and passing\n        data to user-code functions.\n\n        .. seealso:: :func:`enable_object_reuse`\n\n        :return: This object.\n        '
        self._j_execution_config = self._j_execution_config.disableObjectReuse()
        return self

    def is_object_reuse_enabled(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns whether object reuse has been enabled or disabled.\n\n        .. seealso:: :func:`enable_object_reuse`\n\n        :return: Boolean value that represent whether object reuse has been enabled or disabled.\n        '
        return self._j_execution_config.isObjectReuseEnabled()

    def get_global_job_parameters(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        '\n        Gets current configuration dict.\n\n        :return: The configuration dict.\n        '
        return dict(self._j_execution_config.getGlobalJobParameters().toMap())

    def set_global_job_parameters(self, global_job_parameters_dict: Dict) -> 'ExecutionConfig':
        if False:
            i = 10
            return i + 15
        '\n        Register a custom, serializable user configuration dict.\n\n        Example:\n        ::\n\n            >>> config.set_global_job_parameters({"environment.checkpoint_interval": "1000"})\n\n        :param global_job_parameters_dict: Custom user configuration dict.\n        '
        gateway = get_gateway()
        Configuration = gateway.jvm.org.apache.flink.configuration.Configuration
        j_global_job_parameters = Configuration()
        for key in global_job_parameters_dict:
            if not isinstance(global_job_parameters_dict[key], str):
                value = str(global_job_parameters_dict[key])
            else:
                value = global_job_parameters_dict[key]
            j_global_job_parameters.setString(key, value)
        self._j_execution_config.setGlobalJobParameters(j_global_job_parameters)
        return self

    def add_default_kryo_serializer(self, type_class_name: str, serializer_class_name: str) -> 'ExecutionConfig':
        if False:
            while True:
                i = 10
        '\n        Adds a new Kryo default serializer to the Runtime.\n\n        Example:\n        ::\n\n            >>> config.add_default_kryo_serializer("com.aaa.bbb.PojoClass",\n            ...                                    "com.aaa.bbb.Serializer")\n\n        :param type_class_name: The full-qualified java class name of the types serialized with the\n                                given serializer.\n        :param serializer_class_name: The full-qualified java class name of the serializer to use.\n        '
        type_clz = load_java_class(type_class_name)
        j_serializer_clz = load_java_class(serializer_class_name)
        self._j_execution_config.addDefaultKryoSerializer(type_clz, j_serializer_clz)
        return self

    def register_type_with_kryo_serializer(self, type_class_name: str, serializer_class_name: str) -> 'ExecutionConfig':
        if False:
            i = 10
            return i + 15
        '\n        Registers the given Serializer via its class as a serializer for the given type at the\n        KryoSerializer.\n\n        Example:\n        ::\n\n            >>> config.register_type_with_kryo_serializer("com.aaa.bbb.PojoClass",\n            ...                                           "com.aaa.bbb.Serializer")\n\n        :param type_class_name: The full-qualified java class name of the types serialized with\n                                the given serializer.\n        :param serializer_class_name: The full-qualified java class name of the serializer to use.\n        '
        type_clz = load_java_class(type_class_name)
        j_serializer_clz = load_java_class(serializer_class_name)
        self._j_execution_config.registerTypeWithKryoSerializer(type_clz, j_serializer_clz)
        return self

    def register_pojo_type(self, type_class_name: str) -> 'ExecutionConfig':
        if False:
            print('Hello World!')
        '\n        Registers the given type with the serialization stack. If the type is eventually\n        serialized as a POJO, then the type is registered with the POJO serializer. If the\n        type ends up being serialized with Kryo, then it will be registered at Kryo to make\n        sure that only tags are written.\n\n        Example:\n        ::\n\n            >>> config.register_pojo_type("com.aaa.bbb.PojoClass")\n\n        :param type_class_name: The full-qualified java class name of the type to register.\n        '
        type_clz = load_java_class(type_class_name)
        self._j_execution_config.registerPojoType(type_clz)
        return self

    def register_kryo_type(self, type_class_name: str) -> 'ExecutionConfig':
        if False:
            i = 10
            return i + 15
        '\n        Registers the given type with the serialization stack. If the type is eventually\n        serialized as a POJO, then the type is registered with the POJO serializer. If the\n        type ends up being serialized with Kryo, then it will be registered at Kryo to make\n        sure that only tags are written.\n\n        Example:\n        ::\n\n            >>> config.register_kryo_type("com.aaa.bbb.KryoClass")\n\n        :param type_class_name: The full-qualified java class name of the type to register.\n        '
        type_clz = load_java_class(type_class_name)
        self._j_execution_config.registerKryoType(type_clz)
        return self

    def get_registered_types_with_kryo_serializer_classes(self) -> Dict[str, str]:
        if False:
            return 10
        '\n        Returns the registered types with their Kryo Serializer classes.\n\n        :return: The dict which the keys are full-qualified java class names of the registered\n                 types and the values are full-qualified java class names of the Kryo Serializer\n                 classes.\n        '
        j_clz_map = self._j_execution_config.getRegisteredTypesWithKryoSerializerClasses()
        registered_serializers = {}
        for key in j_clz_map:
            registered_serializers[key.getName()] = j_clz_map[key].getName()
        return registered_serializers

    def get_default_kryo_serializer_classes(self) -> Dict[str, str]:
        if False:
            while True:
                i = 10
        '\n        Returns the registered default Kryo Serializer classes.\n\n        :return: The dict which the keys are full-qualified java class names of the registered\n                 types and the values are full-qualified java class names of the Kryo default\n                 Serializer classes.\n        '
        j_clz_map = self._j_execution_config.getDefaultKryoSerializerClasses()
        default_kryo_serializers = {}
        for key in j_clz_map:
            default_kryo_serializers[key.getName()] = j_clz_map[key].getName()
        return default_kryo_serializers

    def get_registered_kryo_types(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Returns the registered Kryo types.\n\n        :return: The list of full-qualified java class names of the registered Kryo types.\n        '
        j_clz_set = self._j_execution_config.getRegisteredKryoTypes()
        return [value.getName() for value in j_clz_set]

    def get_registered_pojo_types(self) -> List[str]:
        if False:
            return 10
        '\n        Returns the registered POJO types.\n\n        :return: The list of full-qualified java class names of the registered POJO types.\n        '
        j_clz_set = self._j_execution_config.getRegisteredPojoTypes()
        return [value.getName() for value in j_clz_set]

    def is_auto_type_registration_disabled(self) -> bool:
        if False:
            return 10
        '\n        Returns whether Flink is automatically registering all types in the user programs with\n        Kryo.\n\n        :return: ``True`` means auto type registration is disabled and ``False`` means enabled.\n        '
        return self._j_execution_config.isAutoTypeRegistrationDisabled()

    def disable_auto_type_registration(self) -> 'ExecutionConfig':
        if False:
            i = 10
            return i + 15
        '\n        Control whether Flink is automatically registering all types in the user programs with\n        Kryo.\n        '
        self._j_execution_config.disableAutoTypeRegistration()
        return self

    def is_use_snapshot_compression(self) -> bool:
        if False:
            return 10
        '\n        Returns whether he compression (snappy) for keyed state in full checkpoints and savepoints\n        is enabled.\n\n        :return: ``True`` means enabled and ``False`` means disabled.\n        '
        return self._j_execution_config.isUseSnapshotCompression()

    def set_use_snapshot_compression(self, use_snapshot_compression: bool) -> 'ExecutionConfig':
        if False:
            print('Hello World!')
        '\n        Control whether the compression (snappy) for keyed state in full checkpoints and savepoints\n        is enabled.\n\n        :param use_snapshot_compression: ``True`` means enabled and ``False`` means disabled.\n        '
        self._j_execution_config.setUseSnapshotCompression(use_snapshot_compression)
        return self

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return isinstance(other, self.__class__) and self._j_execution_config == other._j_execution_config

    def __hash__(self):
        if False:
            return 10
        return self._j_execution_config.hashCode()