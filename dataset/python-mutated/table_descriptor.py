from typing import Dict, Union, List, Optional
from pyflink.common.config_options import ConfigOption
from pyflink.java_gateway import get_gateway
from pyflink.table.schema import Schema
from pyflink.util.java_utils import to_jarray
__all__ = ['TableDescriptor', 'FormatDescriptor']

class TableDescriptor(object):
    """
    Describes a CatalogTable representing a source or sink.

    TableDescriptor is a template for creating a CatalogTable instance. It closely resembles the
    "CREATE TABLE" SQL DDL statement, containing schema, connector options, and other
    characteristics. Since tables in Flink are typically backed by external systems, the
    descriptor describes how a connector (and possibly its format) are configured.

    This can be used to register a table in the Table API, see :func:`create_temporary_table` in
    TableEnvironment.
    """

    def __init__(self, j_table_descriptor):
        if False:
            print('Hello World!')
        self._j_table_descriptor = j_table_descriptor

    @staticmethod
    def for_connector(connector: str) -> 'TableDescriptor.Builder':
        if False:
            i = 10
            return i + 15
        '\n        Creates a new :class:`~pyflink.table.TableDescriptor.Builder` for a table using the given\n        connector.\n\n        :param connector: The factory identifier for the connector.\n        '
        gateway = get_gateway()
        j_builder = gateway.jvm.TableDescriptor.forConnector(connector)
        return TableDescriptor.Builder(j_builder)

    def get_schema(self) -> Optional[Schema]:
        if False:
            for i in range(10):
                print('nop')
        j_schema = self._j_table_descriptor.getSchema()
        if j_schema.isPresent():
            return Schema(j_schema.get())
        else:
            return None

    def get_options(self) -> Dict[str, str]:
        if False:
            return 10
        return self._j_table_descriptor.getOptions()

    def get_partition_keys(self) -> List[str]:
        if False:
            while True:
                i = 10
        return self._j_table_descriptor.getPartitionKeys()

    def get_comment(self) -> Optional[str]:
        if False:
            return 10
        j_comment = self._j_table_descriptor.getComment()
        if j_comment.isPresent():
            return j_comment.get()
        else:
            return None

    def __str__(self):
        if False:
            return 10
        return self._j_table_descriptor.toString()

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self.__class__ == other.__class__ and self._j_table_descriptor.equals(other._j_table_descriptor)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self._j_table_descriptor.hashCode()

    class Builder(object):
        """
        Builder for TableDescriptor.
        """

        def __init__(self, j_builder):
            if False:
                while True:
                    i = 10
            self._j_builder = j_builder

        def schema(self, schema: Schema) -> 'TableDescriptor.Builder':
            if False:
                while True:
                    i = 10
            '\n            Define the schema of the TableDescriptor.\n            '
            self._j_builder.schema(schema._j_schema)
            return self

        def option(self, key: Union[str, ConfigOption], value) -> 'TableDescriptor.Builder':
            if False:
                while True:
                    i = 10
            '\n            Sets the given option on the table.\n\n            Option keys must be fully specified. When defining options for a Format, use\n            format(FormatDescriptor) instead.\n\n            Example:\n            ::\n\n                >>> TableDescriptor.for_connector("kafka")                 ...     .option("scan.startup.mode", "latest-offset")                 ...     .build()\n\n            '
            if isinstance(key, str):
                self._j_builder.option(key, value)
            else:
                self._j_builder.option(key._j_config_option, value)
            return self

        def format(self, format: Union[str, 'FormatDescriptor'], format_option: ConfigOption[str]=None) -> 'TableDescriptor.Builder':
            if False:
                for i in range(10):
                    print('nop')
            '\n            Defines the format to be used for this table.\n\n            Note that not every connector requires a format to be specified, while others may use\n            multiple formats.\n\n            Example:\n            ::\n\n                >>> TableDescriptor.for_connector("kafka")                 ...     .format(FormatDescriptor.for_format("json")\n                ...                 .option("ignore-parse-errors", "true")\n                ...                 .build())\n\n                will result in the options:\n\n                    \'format\' = \'json\'\n                    \'json.ignore-parse-errors\' = \'true\'\n\n            '
            if format_option is None:
                if isinstance(format, str):
                    self._j_builder.format(format)
                else:
                    self._j_builder.format(format._j_format_descriptor)
            elif isinstance(format, str):
                self._j_builder.format(format_option._j_config_option, format)
            else:
                self._j_builder.format(format_option._j_config_option, format._j_format_descriptor)
            return self

        def partitioned_by(self, *partition_keys: str) -> 'TableDescriptor.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Define which columns this table is partitioned by.\n            '
            gateway = get_gateway()
            self._j_builder.partitionedBy(to_jarray(gateway.jvm.java.lang.String, partition_keys))
            return self

        def comment(self, comment: str) -> 'TableDescriptor.Builder':
            if False:
                while True:
                    i = 10
            '\n            Define the comment for this table.\n            '
            self._j_builder.comment(comment)
            return self

        def build(self) -> 'TableDescriptor':
            if False:
                return 10
            '\n            Returns an immutable instance of :class:`~pyflink.table.TableDescriptor`.\n            '
            return TableDescriptor(self._j_builder.build())

class FormatDescriptor(object):
    """
    Describes a Format and its options for use with :class:`~pyflink.table.TableDescriptor`.

    Formats are responsible for encoding and decoding data in table connectors. Note that not
    every connector has a format, while others may have multiple formats (e.g. the Kafka connector
    has separate formats for keys and values). Common formats are "json", "csv", "avro", etc.
    """

    def __init__(self, j_format_descriptor):
        if False:
            return 10
        self._j_format_descriptor = j_format_descriptor

    @staticmethod
    def for_format(format: str) -> 'FormatDescriptor.Builder':
        if False:
            return 10
        '\n        Creates a new :class:`~pyflink.table.FormatDescriptor.Builder` describing a format with the\n        given format identifier.\n\n        :param format: The factory identifier for the format.\n        '
        gateway = get_gateway()
        j_builder = gateway.jvm.FormatDescriptor.forFormat(format)
        return FormatDescriptor.Builder(j_builder)

    def get_format(self) -> str:
        if False:
            return 10
        return self._j_format_descriptor.getFormat()

    def get_options(self) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        return self._j_format_descriptor.getOptions()

    def __str__(self):
        if False:
            print('Hello World!')
        return self._j_format_descriptor.toString()

    def __eq__(self, other):
        if False:
            print('Hello World!')
        return self.__class__ == other.__class__ and self._j_format_descriptor.equals(other._j_format_descriptor)

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return self._j_format_descriptor.hashCode()

    class Builder(object):
        """
        Builder for FormatDescriptor.
        """

        def __init__(self, j_builder):
            if False:
                return 10
            self._j_builder = j_builder

        def option(self, key: Union[str, ConfigOption], value) -> 'FormatDescriptor.Builder':
            if False:
                return 10
            '\n            Sets the given option on the format.\n\n            Note that format options must not be prefixed with the format identifier itself here.\n\n            Example:\n            ::\n\n                >>> FormatDescriptor.for_format("json")                 ...     .option("ignore-parse-errors", "true")                 ...     .build()\n\n                will automatically be converted into its prefixed form:\n\n                    \'format\' = \'json\'\n                    \'json.ignore-parse-errors\' = \'true\'\n\n            '
            if isinstance(key, str):
                self._j_builder.option(key, value)
            else:
                self._j_builder.option(key._j_config_option, value)
            return self

        def build(self) -> 'FormatDescriptor':
            if False:
                while True:
                    i = 10
            '\n            Returns an immutable instance of :class:`~pyflink.table.FormatDescriptor`.\n            '
            return FormatDescriptor(self._j_builder.build())