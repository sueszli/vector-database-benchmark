import warnings
from pyflink.java_gateway import get_gateway
from pyflink.util.java_utils import create_url_class_loader
from pyflink.common import Configuration
__all__ = ['EnvironmentSettings']

class EnvironmentSettings(object):
    """
    Defines all parameters that initialize a table environment. Those parameters are used only
    during instantiation of a :class:`~pyflink.table.TableEnvironment` and cannot be changed
    afterwards.

    Example:
    ::

        >>> EnvironmentSettings.new_instance() \\
        ...     .in_streaming_mode() \\
        ...     .with_built_in_catalog_name("my_catalog") \\
        ...     .with_built_in_database_name("my_database") \\
        ...     .build()

    :func:`~EnvironmentSettings.in_streaming_mode` or :func:`~EnvironmentSettings.in_batch_mode`
    might be convenient as shortcuts.
    """

    class Builder(object):
        """
        A builder for :class:`~EnvironmentSettings`.
        """

        def __init__(self):
            if False:
                return 10
            gateway = get_gateway()
            self._j_builder = gateway.jvm.EnvironmentSettings.Builder()

        def with_configuration(self, config: Configuration) -> 'EnvironmentSettings.Builder':
            if False:
                print('Hello World!')
            '\n            Creates the EnvironmentSetting with specified Configuration.\n\n            :return: EnvironmentSettings.\n            '
            self._j_builder = self._j_builder.withConfiguration(config._j_configuration)
            return self

        def in_batch_mode(self) -> 'EnvironmentSettings.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Sets that the components should work in a batch mode. Streaming mode by default.\n\n            :return: This object.\n            '
            self._j_builder = self._j_builder.inBatchMode()
            return self

        def in_streaming_mode(self) -> 'EnvironmentSettings.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Sets that the components should work in a streaming mode. Enabled by default.\n\n            :return: This object.\n            '
            self._j_builder = self._j_builder.inStreamingMode()
            return self

        def with_built_in_catalog_name(self, built_in_catalog_name: str) -> 'EnvironmentSettings.Builder':
            if False:
                print('Hello World!')
            '\n            Specifies the name of the initial catalog to be created when instantiating\n            a :class:`~pyflink.table.TableEnvironment`.\n\n            This catalog is an in-memory catalog that will be used to store all temporary objects\n            (e.g. from :func:`~pyflink.table.TableEnvironment.create_temporary_view` or\n            :func:`~pyflink.table.TableEnvironment.create_temporary_system_function`) that cannot\n            be persisted because they have no serializable representation.\n\n            It will also be the initial value for the current catalog which can be altered via\n            :func:`~pyflink.table.TableEnvironment.use_catalog`.\n\n            Default: "default_catalog".\n\n            :param built_in_catalog_name: The specified built-in catalog name.\n            :return: This object.\n            '
            self._j_builder = self._j_builder.withBuiltInCatalogName(built_in_catalog_name)
            return self

        def with_built_in_database_name(self, built_in_database_name: str) -> 'EnvironmentSettings.Builder':
            if False:
                for i in range(10):
                    print('nop')
            '\n            Specifies the name of the default database in the initial catalog to be\n            created when instantiating a :class:`~pyflink.table.TableEnvironment`.\n\n            This database is an in-memory database that will be used to store all temporary\n            objects (e.g. from :func:`~pyflink.table.TableEnvironment.create_temporary_view` or\n            :func:`~pyflink.table.TableEnvironment.create_temporary_system_function`) that cannot\n            be persisted because they have no serializable representation.\n\n            It will also be the initial value for the current catalog which can be altered via\n            :func:`~pyflink.table.TableEnvironment.use_catalog`.\n\n            Default: "default_database".\n\n            :param built_in_database_name: The specified built-in database name.\n            :return: This object.\n            '
            self._j_builder = self._j_builder.withBuiltInDatabaseName(built_in_database_name)
            return self

        def build(self) -> 'EnvironmentSettings':
            if False:
                return 10
            '\n            Returns an immutable instance of EnvironmentSettings.\n\n            :return: an immutable instance of EnvironmentSettings.\n            '
            gateway = get_gateway()
            context_classloader = gateway.jvm.Thread.currentThread().getContextClassLoader()
            new_classloader = create_url_class_loader([], context_classloader)
            gateway.jvm.Thread.currentThread().setContextClassLoader(new_classloader)
            return EnvironmentSettings(self._j_builder.build())

    def __init__(self, j_environment_settings):
        if False:
            for i in range(10):
                print('nop')
        self._j_environment_settings = j_environment_settings

    def get_built_in_catalog_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the specified name of the initial catalog to be created when instantiating a\n        :class:`~pyflink.table.TableEnvironment`.\n\n        :return: The specified name of the initial catalog to be created.\n        '
        return self._j_environment_settings.getBuiltInCatalogName()

    def get_built_in_database_name(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Gets the specified name of the default database in the initial catalog to be created when\n        instantiating a :class:`~pyflink.table.TableEnvironment`.\n\n        :return: The specified name of the default database in the initial catalog to be created.\n        '
        return self._j_environment_settings.getBuiltInDatabaseName()

    def is_streaming_mode(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Tells if the :class:`~pyflink.table.TableEnvironment` should work in a batch or streaming\n        mode.\n\n        :return: True if the TableEnvironment should work in a streaming mode, false otherwise.\n        '
        return self._j_environment_settings.isStreamingMode()

    def to_configuration(self) -> Configuration:
        if False:
            print('Hello World!')
        '\n        Convert to `pyflink.common.Configuration`.\n\n        :return: Configuration with specified value.\n\n        .. note:: Deprecated in 1.15. Please use\n                :func:`EnvironmentSettings.get_configuration` instead.\n        '
        warnings.warn('Deprecated in 1.15.', DeprecationWarning)
        return Configuration(j_configuration=self._j_environment_settings.toConfiguration())

    def get_configuration(self) -> Configuration:
        if False:
            while True:
                i = 10
        '\n        Get the underlying `pyflink.common.Configuration`.\n\n        :return: Configuration with specified value.\n        '
        return Configuration(j_configuration=self._j_environment_settings.getConfiguration())

    @staticmethod
    def new_instance() -> 'EnvironmentSettings.Builder':
        if False:
            while True:
                i = 10
        '\n        Creates a builder for creating an instance of EnvironmentSettings.\n\n        :return: A builder of EnvironmentSettings.\n        '
        return EnvironmentSettings.Builder()

    @staticmethod
    def from_configuration(config: Configuration) -> 'EnvironmentSettings':
        if False:
            i = 10
            return i + 15
        '\n        Creates the EnvironmentSetting with specified Configuration.\n\n        :return: EnvironmentSettings.\n\n        .. note:: Deprecated in 1.15. Please use\n                :func:`EnvironmentSettings.Builder.with_configuration` instead.\n        '
        warnings.warn('Deprecated in 1.15.', DeprecationWarning)
        gateway = get_gateway()
        context_classloader = gateway.jvm.Thread.currentThread().getContextClassLoader()
        new_classloader = create_url_class_loader([], context_classloader)
        gateway.jvm.Thread.currentThread().setContextClassLoader(new_classloader)
        return EnvironmentSettings(get_gateway().jvm.EnvironmentSettings.fromConfiguration(config._j_configuration))

    @staticmethod
    def in_streaming_mode() -> 'EnvironmentSettings':
        if False:
            i = 10
            return i + 15
        '\n        Creates a default instance of EnvironmentSettings in streaming execution mode.\n\n        In this mode, both bounded and unbounded data streams can be processed.\n\n        This method is a shortcut for creating a :class:`~pyflink.table.TableEnvironment` with\n        little code. Use the builder provided in :func:`EnvironmentSettings.new_instance` for\n        advanced settings.\n\n        :return: EnvironmentSettings.\n        '
        return EnvironmentSettings.new_instance().in_streaming_mode().build()

    @staticmethod
    def in_batch_mode() -> 'EnvironmentSettings':
        if False:
            return 10
        '\n        Creates a default instance of EnvironmentSettings in batch execution mode.\n\n        This mode is highly optimized for batch scenarios. Only bounded data streams can be\n        processed in this mode.\n\n        This method is a shortcut for creating a :class:`~pyflink.table.TableEnvironment` with\n        little code. Use the builder provided in :func:`EnvironmentSettings.new_instance` for\n        advanced settings.\n\n        :return: EnvironmentSettings.\n        '
        return EnvironmentSettings.new_instance().in_batch_mode().build()