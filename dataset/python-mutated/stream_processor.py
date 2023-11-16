import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from airbyte_cdk.models.airbyte_protocol import DestinationSyncMode, SyncMode
from jinja2 import Template
from normalization.destination_type import DestinationType
from normalization.transform_catalog import dbt_macro
from normalization.transform_catalog.destination_name_transformer import DestinationNameTransformer, transform_json_naming
from normalization.transform_catalog.table_name_registry import TableNameRegistry
from normalization.transform_catalog.utils import is_airbyte_column, is_array, is_big_integer, is_boolean, is_combining_node, is_date, is_datetime, is_datetime_with_timezone, is_datetime_without_timezone, is_long, is_number, is_object, is_simple_property, is_string, is_time, is_time_with_timezone, jinja_call, remove_jinja
MAXIMUM_COLUMNS_TO_USE_EPHEMERAL = 450

class PartitionScheme(Enum):
    """
    When possible, normalization will try to output partitioned/indexed/sorted tables (depending on the destination support)
    This enum specifies which column to use when doing so (which affects how fast the table can be read using that column as predicate)
    """
    ACTIVE_ROW = 'active_row'
    UNIQUE_KEY = 'unique_key'
    NOTHING = 'nothing'
    DEFAULT = ''

class TableMaterializationType(Enum):
    """
    Defines the folders and dbt materialization mode of models (as configured in dbt_project.yml file)
    """
    CTE = 'airbyte_ctes'
    VIEW = 'airbyte_views'
    TABLE = 'airbyte_tables'
    INCREMENTAL = 'airbyte_incremental'

class StreamProcessor(object):
    """
    Takes as input an Airbyte Stream as described in the (configured) Airbyte Catalog's Json Schema.
    Associated input raw data is expected to be stored in a staging area table.

    This processor generates SQL models to transform such a stream into a final table in the destination schema.
    This is done by generating a DBT pipeline of transformations (multiple SQL models queries) that may be materialized
    in the intermediate schema "raw_schema" (changing the dbt_project.yml settings).
    The final output data should be written in "schema".

    The pipeline includes transformations such as:
    - Parsing a JSON blob column and extracting each field property in its own SQL column
    - Casting each SQL column to the proper JSON data type
    - Generating an artificial (primary key) ID column based on the hashing of the row

    If any nested columns are discovered in the stream, a JSON blob SQL column is created in the top level parent stream
    and a new StreamProcessor instance will be spawned for each children substreams. These Sub-Stream Processors are then
    able to generate models to parse and extract recursively from its parent StreamProcessor model into separate SQL tables
    the content of that JSON blob SQL column.
    """

    def __init__(self, stream_name: str, destination_type: DestinationType, raw_schema: str, default_schema: str, schema: str, source_sync_mode: SyncMode, destination_sync_mode: DestinationSyncMode, cursor_field: List[str], primary_key: List[List[str]], json_column_name: str, properties: Dict, tables_registry: TableNameRegistry, from_table: Union[str, dbt_macro.Macro]):
        if False:
            i = 10
            return i + 15
        '\n        See StreamProcessor.create()\n        '
        self.stream_name: str = stream_name
        self.destination_type: DestinationType = destination_type
        self.raw_schema: str = raw_schema
        self.schema: str = schema
        self.source_sync_mode: SyncMode = source_sync_mode
        self.destination_sync_mode: DestinationSyncMode = destination_sync_mode
        self.cursor_field: List[str] = cursor_field
        self.primary_key: List[List[str]] = primary_key
        self.json_column_name: str = json_column_name
        self.properties: Dict = properties
        self.tables_registry: TableNameRegistry = tables_registry
        self.from_table: Union[str, dbt_macro.Macro] = from_table
        self.name_transformer: DestinationNameTransformer = DestinationNameTransformer(destination_type)
        self.json_path: List[str] = [stream_name]
        self.final_table_name: str = ''
        self.sql_outputs: Dict[str, str] = {}
        self.parent: Optional['StreamProcessor'] = None
        self.is_nested_array: bool = False
        self.default_schema: str = default_schema
        self.airbyte_ab_id = '_airbyte_ab_id'
        self.airbyte_emitted_at = '_airbyte_emitted_at'
        self.airbyte_normalized_at = '_airbyte_normalized_at'
        self.airbyte_unique_key = '_airbyte_unique_key'
        self.models_to_source: Dict[str, str] = {}

    @staticmethod
    def create_from_parent(parent, child_name: str, json_column_name: str, properties: Dict, is_nested_array: bool, from_table: str) -> 'StreamProcessor':
        if False:
            return 10
        '\n        @param parent is the Stream Processor that originally created this instance to handle a nested column from that parent table.\n\n        @param json_column_name is the name of the column in the parent data table containing the json column to transform\n        @param properties is the json schema description of this nested stream\n        @param is_nested_array is a boolean flag specifying if the child is a nested array that needs to be extracted\n\n        @param tables_registry is the global context recording all tables created so far\n        @param from_table is the parent table to extract the nested stream from\n\n        The child stream processor will create a separate table to contain the unnested data.\n        '
        if parent.destination_sync_mode.value == DestinationSyncMode.append_dedup.value:
            parent_sync_mode = DestinationSyncMode.append
        else:
            parent_sync_mode = parent.destination_sync_mode
        result = StreamProcessor.create(stream_name=child_name, destination_type=parent.destination_type, raw_schema=parent.raw_schema, default_schema=parent.default_schema, schema=parent.schema, source_sync_mode=parent.source_sync_mode, destination_sync_mode=parent_sync_mode, cursor_field=[], primary_key=[], json_column_name=json_column_name, properties=properties, tables_registry=parent.tables_registry, from_table=from_table)
        result.parent = parent
        result.is_nested_array = is_nested_array
        result.json_path = parent.json_path + [child_name]
        return result

    @staticmethod
    def create(stream_name: str, destination_type: DestinationType, raw_schema: str, default_schema: str, schema: str, source_sync_mode: SyncMode, destination_sync_mode: DestinationSyncMode, cursor_field: List[str], primary_key: List[List[str]], json_column_name: str, properties: Dict, tables_registry: TableNameRegistry, from_table: Union[str, dbt_macro.Macro]) -> 'StreamProcessor':
        if False:
            print('Hello World!')
        '\n        @param stream_name of the stream being processed\n\n        @param destination_type is the destination type of warehouse\n        @param raw_schema is the name of the staging intermediate schema where to create internal tables/views\n        @param schema is the name of the schema where to store the final tables where to store the transformed data\n\n        @param source_sync_mode is describing how source are producing data\n        @param destination_sync_mode is describing how destination should handle the new data batch\n        @param cursor_field is the field to use to determine order of records\n        @param primary_key is a list of fields to use as a (composite) primary key\n\n        @param json_column_name is the name of the column in the raw data table containing the json column to transform\n        @param properties is the json schema description of this stream\n\n        @param tables_registry is the global context recording all tables created so far\n        @param from_table is the table this stream is being extracted from originally\n        '
        return StreamProcessor(stream_name, destination_type, raw_schema, default_schema, schema, source_sync_mode, destination_sync_mode, cursor_field, primary_key, json_column_name, properties, tables_registry, from_table)

    def collect_table_names(self):
        if False:
            for i in range(10):
                print('nop')
        column_names = self.extract_column_names()
        self.tables_registry.register_table(self.get_schema(True), self.get_schema(False), self.stream_name, self.json_path)
        for child in self.find_children_streams(self.from_table, column_names):
            child.collect_table_names()

    def get_stream_source(self):
        if False:
            print('Hello World!')
        if not self.parent:
            return self.from_table.source_name + '.' + self.from_table.table_name
        cur = self.parent
        while cur.parent:
            cur = cur.parent
        return cur.from_table.source_name + '.' + cur.from_table.table_name

    def process(self) -> List['StreamProcessor']:
        if False:
            return 10
        '\n        See description of StreamProcessor class.\n        @return List of StreamProcessor to handle recursively nested columns from this stream\n        '
        if not self.properties:
            print(f"  Ignoring stream '{self.stream_name}' from {self.current_json_path()} because properties list is empty")
            return []
        column_names = self.extract_column_names()
        column_count = len(column_names)
        if column_count == 0:
            print(f"  Ignoring stream '{self.stream_name}' from {self.current_json_path()} because no columns were identified")
            return []
        from_table = str(self.from_table)
        from_table = self.add_to_outputs(self.generate_json_parsing_model(from_table, column_names), self.get_model_materialization_mode(is_intermediate=True), is_intermediate=True, suffix='ab1')
        from_table = self.add_to_outputs(self.generate_column_typing_model(from_table, column_names), self.get_model_materialization_mode(is_intermediate=True, column_count=column_count), is_intermediate=True, suffix='ab2')
        if self.destination_sync_mode != DestinationSyncMode.append_dedup:
            from_table = self.add_to_outputs(self.generate_id_hashing_model(from_table, column_names), self.get_model_materialization_mode(is_intermediate=True, column_count=column_count), is_intermediate=True, suffix='ab3')
            from_table = self.add_to_outputs(self.generate_final_model(from_table, column_names), self.get_model_materialization_mode(is_intermediate=False, column_count=column_count), is_intermediate=False)
        else:
            if self.is_incremental_mode(self.destination_sync_mode):
                if self.destination_type.value == DestinationType.POSTGRES.value:
                    forced_materialization_type = TableMaterializationType.INCREMENTAL
                else:
                    forced_materialization_type = TableMaterializationType.VIEW
            else:
                forced_materialization_type = TableMaterializationType.CTE
            from_table = self.add_to_outputs(self.generate_id_hashing_model(from_table, column_names), forced_materialization_type, is_intermediate=True, suffix='stg')
            from_table = self.add_to_outputs(self.generate_scd_type_2_model(from_table, column_names), self.get_model_materialization_mode(is_intermediate=False, column_count=column_count), is_intermediate=False, suffix='scd', subdir='scd', unique_key=self.name_transformer.normalize_column_name(f'{self.airbyte_unique_key}_scd'), partition_by=PartitionScheme.ACTIVE_ROW)
            where_clause = f"\nand {self.name_transformer.normalize_column_name('_airbyte_active_row')} = 1"
            self.add_to_outputs(self.generate_final_model(from_table, column_names, unique_key=self.get_unique_key()) + where_clause, self.get_model_materialization_mode(is_intermediate=False, column_count=column_count), is_intermediate=False, unique_key=self.get_unique_key(), partition_by=PartitionScheme.UNIQUE_KEY)
        return self.find_children_streams(from_table, column_names)

    def extract_column_names(self) -> Dict[str, Tuple[str, str]]:
        if False:
            while True:
                i = 10
        '\n        Generate a mapping of JSON properties to normalized SQL Column names, handling collisions and avoid duplicate names\n\n        The mapped value to a field property is a tuple where:\n         - the first value is the normalized "raw" column name\n         - the second value is the normalized quoted column name to be used in jinja context\n        '
        fields = []
        for field in self.properties.keys():
            if not is_airbyte_column(field):
                fields.append(field)
        result = {}
        field_names = set()
        for field in fields:
            field_name = self.name_transformer.normalize_column_name(field, in_jinja=False)
            field_name_lookup = self.name_transformer.normalize_column_identifier_case_for_lookup(field_name)
            jinja_name = self.name_transformer.normalize_column_name(field, in_jinja=True)
            if field_name_lookup in field_names:
                for i in range(1, 1000):
                    field_name = self.name_transformer.normalize_column_name(f'{field}_{i}', in_jinja=False)
                    field_name_lookup = self.name_transformer.normalize_column_identifier_case_for_lookup(field_name)
                    jinja_name = self.name_transformer.normalize_column_name(f'{field}_{i}', in_jinja=True)
                    if field_name_lookup not in field_names:
                        break
            field_names.add(field_name_lookup)
            result[field] = (field_name, jinja_name)
        return result

    def find_children_streams(self, from_table: str, column_names: Dict[str, Tuple[str, str]]) -> List['StreamProcessor']:
        if False:
            return 10
        '\n        For each complex type properties, generate a new child StreamProcessor that produce separate child pipelines.\n        The current stream/table is used as the parent from which to extract data from.\n        '
        properties = self.properties
        children: List[StreamProcessor] = []
        for field in properties.keys():
            children_properties = None
            is_nested_array = False
            json_column_name = ''
            if is_airbyte_column(field):
                pass
            elif is_combining_node(properties[field]):
                pass
            elif 'type' not in properties[field] or is_object(properties[field]['type']):
                children_properties = find_properties_object([], field, properties[field])
                is_nested_array = False
                json_column_name = column_names[field][1]
            elif is_array(properties[field]['type']) and 'items' in properties[field]:
                quoted_field = column_names[field][1]
                children_properties = find_properties_object([], field, properties[field]['items'])
                is_nested_array = True
                json_column_name = f'unnested_column_value({quoted_field})'
            if children_properties:
                for child_key in children_properties:
                    stream_processor = StreamProcessor.create_from_parent(parent=self, child_name=field, json_column_name=json_column_name, properties=children_properties[child_key], is_nested_array=is_nested_array, from_table=from_table)
                    children.append(stream_processor)
        return children

    def generate_json_parsing_model(self, from_table: str, column_names: Dict[str, Tuple[str, str]]) -> Any:
        if False:
            i = 10
            return i + 15
        if self.destination_type == DestinationType.ORACLE:
            table_alias = ''
        else:
            table_alias = 'as table_alias'
        template = Template("\n-- SQL model to parse JSON blob stored in a single column and extract into separated field columns as described by the JSON Schema\n-- depends_on: {{ from_table }}\n{{ unnesting_before_query }}\nselect\n{%- if parent_hash_id %}\n    {{ parent_hash_id }},\n{%- endif %}\n{%- for field in fields %}\n    {{ field }},\n{%- endfor %}\n    {{ col_ab_id }},\n    {{ col_emitted_at }},\n    {{ '{{ current_timestamp() }}' }} as {{ col_normalized_at }}\nfrom {{ from_table }} {{ table_alias }}\n{{ sql_table_comment }}\n{{ unnesting_from }}\nwhere 1 = 1\n{{ unnesting_where }}\n")
        sql = template.render(col_ab_id=self.get_ab_id(), col_emitted_at=self.get_emitted_at(), col_normalized_at=self.get_normalized_at(), table_alias=table_alias, unnesting_before_query=self.unnesting_before_query(from_table), parent_hash_id=self.parent_hash_id(), fields=self.extract_json_columns(column_names), from_table=jinja_call(from_table), unnesting_from=self.unnesting_from(), unnesting_where=self.unnesting_where(), sql_table_comment=self.sql_table_comment())
        return sql

    def get_ab_id(self, in_jinja: bool=False):
        if False:
            i = 10
            return i + 15
        return self.name_transformer.normalize_column_name(self.airbyte_ab_id, in_jinja, False)

    def get_emitted_at(self, in_jinja: bool=False):
        if False:
            while True:
                i = 10
        return self.name_transformer.normalize_column_name(self.airbyte_emitted_at, in_jinja, False)

    def get_normalized_at(self, in_jinja: bool=False):
        if False:
            i = 10
            return i + 15
        return self.name_transformer.normalize_column_name(self.airbyte_normalized_at, in_jinja, False)

    def get_unique_key(self, in_jinja: bool=False):
        if False:
            while True:
                i = 10
        return self.name_transformer.normalize_column_name(self.airbyte_unique_key, in_jinja, False)

    def extract_json_columns(self, column_names: Dict[str, Tuple[str, str]]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return [self.extract_json_column(field, self.json_column_name, self.properties[field], column_names[field][0], 'table_alias') for field in column_names]

    @staticmethod
    def extract_json_column(property_name: str, json_column_name: str, definition: Dict, column_name: str, table_alias: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        json_path = [property_name]
        normalized_json_path = [transform_json_naming(property_name)]
        table_alias = f'{table_alias}'
        if 'unnested_column_value' in json_column_name:
            table_alias = ''
        json_extract = jinja_call(f"json_extract('{table_alias}', {json_column_name}, {json_path})")
        if 'type' in definition:
            if is_array(definition['type']):
                json_extract = jinja_call(f'json_extract_array({json_column_name}, {json_path}, {normalized_json_path})')
                if is_simple_property(definition.get('items', {'type': 'object'})):
                    json_extract = jinja_call(f'json_extract_string_array({json_column_name}, {json_path}, {normalized_json_path})')
            elif is_object(definition['type']):
                json_extract = jinja_call(f"json_extract('{table_alias}', {json_column_name}, {json_path}, {normalized_json_path})")
            elif is_simple_property(definition):
                json_extract = jinja_call(f'json_extract_scalar({json_column_name}, {json_path}, {normalized_json_path})')
        return f'{json_extract} as {column_name}'

    def generate_column_typing_model(self, from_table: str, column_names: Dict[str, Tuple[str, str]]) -> Any:
        if False:
            print('Hello World!')
        template = Template("\n-- SQL model to cast each column to its adequate SQL type converted from the JSON schema type\n-- depends_on: {{ from_table }}\nselect\n{%- if parent_hash_id %}\n    {{ parent_hash_id }},\n{%- endif %}\n{%- for field in fields %}\n    {{ field }},\n{%- endfor %}\n    {{ col_ab_id }},\n    {{ col_emitted_at }},\n    {{ '{{ current_timestamp() }}' }} as {{ col_normalized_at }}\nfrom {{ from_table }}\n{{ sql_table_comment }}\nwhere 1 = 1\n    ")
        sql = template.render(col_ab_id=self.get_ab_id(), col_emitted_at=self.get_emitted_at(), col_normalized_at=self.get_normalized_at(), parent_hash_id=self.parent_hash_id(), fields=self.cast_property_types(column_names), from_table=jinja_call(from_table), sql_table_comment=self.sql_table_comment())
        return sql

    def cast_property_types(self, column_names: Dict[str, Tuple[str, str]]) -> List[str]:
        if False:
            while True:
                i = 10
        return [self.cast_property_type(field, column_names[field][0], column_names[field][1]) for field in column_names]

    def cast_property_type(self, property_name: str, column_name: str, jinja_column: str) -> Any:
        if False:
            print('Hello World!')
        definition = self.properties[property_name]
        if 'type' not in definition:
            print(f'WARN: Unknown type for column {property_name} at {self.current_json_path()}')
            return column_name
        elif is_array(definition['type']):
            return column_name
        elif is_object(definition['type']):
            sql_type = jinja_call('type_json()')
        elif is_boolean(definition['type'], definition):
            cast_operation = jinja_call(f'cast_to_boolean({jinja_column})')
            return f'{cast_operation} as {column_name}'
        elif is_big_integer(definition):
            sql_type = jinja_call('type_very_large_integer()')
        elif is_long(definition['type'], definition):
            sql_type = jinja_call('dbt_utils.type_bigint()')
        elif is_number(definition['type']):
            sql_type = jinja_call('dbt_utils.type_float()')
        elif is_datetime(definition):
            if self.destination_type == DestinationType.SNOWFLAKE:
                if is_datetime_without_timezone(definition):
                    return self.generate_snowflake_timestamp_statement(column_name)
                return self.generate_snowflake_timestamp_tz_statement(column_name)
            if self.destination_type == DestinationType.MYSQL and is_datetime_without_timezone(definition):
                return self.generate_mysql_datetime_format_statement(column_name)
            replace_operation = jinja_call(f'empty_string_to_null({jinja_column})')
            if self.destination_type.value == DestinationType.MSSQL.value:
                if is_datetime_with_timezone(definition):
                    sql_type = jinja_call('type_timestamp_with_timezone()')
                else:
                    sql_type = jinja_call('type_timestamp_without_timezone()')
                return f'try_parse({replace_operation} as {sql_type}) as {column_name}'
            if self.destination_type == DestinationType.CLICKHOUSE:
                return f"""parseDateTime64BestEffortOrNull(trim(BOTH '"' from {replace_operation})) as {column_name}"""
            if is_datetime_without_timezone(definition):
                sql_type = jinja_call('type_timestamp_without_timezone()')
            else:
                sql_type = jinja_call('type_timestamp_with_timezone()')
            return f'cast({replace_operation} as {sql_type}) as {column_name}'
        elif is_date(definition):
            if self.destination_type.value == DestinationType.MYSQL.value or self.destination_type.value == DestinationType.TIDB.value or self.destination_type.value == DestinationType.DUCKDB.value:
                return self.generate_mysql_date_format_statement(column_name)
            replace_operation = jinja_call(f'empty_string_to_null({jinja_column})')
            if self.destination_type.value == DestinationType.MSSQL.value:
                sql_type = jinja_call('type_date()')
                return f'try_parse({replace_operation} as {sql_type}) as {column_name}'
            if self.destination_type == DestinationType.CLICKHOUSE:
                return f"""toDate(parseDateTimeBestEffortOrNull(trim(BOTH '"' from {replace_operation}))) as {column_name}"""
            sql_type = jinja_call('type_date()')
            return f'cast({replace_operation} as {sql_type}) as {column_name}'
        elif is_time(definition):
            if is_time_with_timezone(definition):
                sql_type = jinja_call('type_time_with_timezone()')
            else:
                sql_type = jinja_call('type_time_without_timezone()')
            if self.destination_type == DestinationType.CLICKHOUSE:
                trimmed_column_name = f"""trim(BOTH '"' from {column_name})"""
                sql_type = f"'{sql_type}'"
                return f"nullif(accurateCastOrNull({trimmed_column_name}, {sql_type}), 'null') as {column_name}"
            if self.destination_type == DestinationType.MYSQL or self.destination_type == DestinationType.TIDB or self.destination_type == DestinationType.DUCKDB:
                return f'nullif(cast({column_name} as {sql_type}), "") as {column_name}'
            replace_operation = jinja_call(f'empty_string_to_null({jinja_column})')
            return f'cast({replace_operation} as {sql_type}) as {column_name}'
        elif is_string(definition['type']):
            sql_type = jinja_call('dbt_utils.type_string()')
            if self.destination_type == DestinationType.CLICKHOUSE:
                trimmed_column_name = f"""trim(BOTH '"' from {column_name})"""
                sql_type = f"'{sql_type}'"
                return f"nullif(accurateCastOrNull({trimmed_column_name}, {sql_type}), 'null') as {column_name}"
            elif self.destination_type == DestinationType.MYSQL:
                sql_type = f'{sql_type}(1024)'
        else:
            print(f"WARN: Unknown type {definition['type']} for column {property_name} at {self.current_json_path()}")
            return column_name
        if self.destination_type == DestinationType.CLICKHOUSE:
            return f"accurateCastOrNull({column_name}, '{sql_type}') as {column_name}"
        else:
            return f'cast({column_name} as {sql_type}) as {column_name}'

    @staticmethod
    def generate_mysql_date_format_statement(column_name: str) -> Any:
        if False:
            i = 10
            return i + 15
        template = Template("\n        case when {{column_name}} = '' then NULL\n        else cast({{column_name}} as date)\n        end as {{column_name}}\n        ")
        return template.render(column_name=column_name)

    @staticmethod
    def generate_mysql_datetime_format_statement(column_name: str) -> Any:
        if False:
            i = 10
            return i + 15
        regexp = '\\\\d{4}-\\\\d{2}-\\\\d{2}T\\\\d{2}:\\\\d{2}:\\\\d{2}.*'
        template = Template("\n        case when {{column_name}} regexp '{{regexp}}' THEN STR_TO_DATE(SUBSTR({{column_name}}, 1, 19), '%Y-%m-%dT%H:%i:%S')\n        else cast(if({{column_name}} = '', NULL, {{column_name}}) as datetime)\n        end as {{column_name}}\n        ")
        return template.render(column_name=column_name, regexp=regexp)

    @staticmethod
    def generate_snowflake_timestamp_tz_statement(column_name: str) -> Any:
        if False:
            return 10
        '\n        Generates snowflake DB specific timestamp case when statement\n        '
        formats = [{'regex': '\\\\d{4}-\\\\d{2}-\\\\d{2}T(\\\\d{2}:){2}\\\\d{2}(\\\\+|-)\\\\d{4}', 'format': 'YYYY-MM-DDTHH24:MI:SSTZHTZM'}, {'regex': '\\\\d{4}-\\\\d{2}-\\\\d{2}T(\\\\d{2}:){2}\\\\d{2}(\\\\+|-)\\\\d{2}', 'format': 'YYYY-MM-DDTHH24:MI:SSTZH'}, {'regex': '\\\\d{4}-\\\\d{2}-\\\\d{2}T(\\\\d{2}:){2}\\\\d{2}\\\\.\\\\d{1,7}(\\\\+|-)\\\\d{4}', 'format': 'YYYY-MM-DDTHH24:MI:SS.FFTZHTZM'}, {'regex': '\\\\d{4}-\\\\d{2}-\\\\d{2}T(\\\\d{2}:){2}\\\\d{2}\\\\.\\\\d{1,7}(\\\\+|-)\\\\d{2}', 'format': 'YYYY-MM-DDTHH24:MI:SS.FFTZH'}]
        template = Template("\n    case\n{% for format_item in formats %}\n        when {{column_name}} regexp '{{format_item['regex']}}' then to_timestamp_tz({{column_name}}, '{{format_item['format']}}')\n{% endfor %}\n        when {{column_name}} = '' then NULL\n    else to_timestamp_tz({{column_name}})\n    end as {{column_name}}\n    ")
        return template.render(formats=formats, column_name=column_name)

    @staticmethod
    def generate_snowflake_timestamp_statement(column_name: str) -> Any:
        if False:
            while True:
                i = 10
        '\n        Generates snowflake DB specific timestamp case when statement\n        '
        formats = [{'regex': '\\\\d{4}-\\\\d{2}-\\\\d{2}T(\\\\d{2}:){2}\\\\d{2}', 'format': 'YYYY-MM-DDTHH24:MI:SS'}, {'regex': '\\\\d{4}-\\\\d{2}-\\\\d{2}T(\\\\d{2}:){2}\\\\d{2}\\\\.\\\\d{1,7}', 'format': 'YYYY-MM-DDTHH24:MI:SS.FF'}]
        template = Template("\n    case\n{% for format_item in formats %}\n        when {{column_name}} regexp '{{format_item['regex']}}' then to_timestamp({{column_name}}, '{{format_item['format']}}')\n{% endfor %}\n        when {{column_name}} = '' then NULL\n    else to_timestamp({{column_name}})\n    end as {{column_name}}\n    ")
        return template.render(formats=formats, column_name=column_name)

    def generate_id_hashing_model(self, from_table: str, column_names: Dict[str, Tuple[str, str]]) -> Any:
        if False:
            for i in range(10):
                print('nop')
        template = Template("\n-- SQL model to build a hash column based on the values of this record\n-- depends_on: {{ from_table }}\nselect\n    {{ '{{' }} dbt_utils.surrogate_key([\n{%- if parent_hash_id %}\n        {{ parent_hash_id }},\n{%- endif %}\n{%- for field in fields %}\n        {{ field }},\n{%- endfor %}\n    ]) {{ '}}' }} as {{ hash_id }},\n    tmp.*\nfrom {{ from_table }} tmp\n{{ sql_table_comment }}\nwhere 1 = 1\n    ")
        sql = template.render(parent_hash_id=self.parent_hash_id(in_jinja=True), fields=self.safe_cast_to_strings(column_names), hash_id=self.hash_id(), from_table=jinja_call(from_table), sql_table_comment=self.sql_table_comment())
        return sql

    def safe_cast_to_strings(self, column_names: Dict[str, Tuple[str, str]]) -> List[str]:
        if False:
            i = 10
            return i + 15
        return [StreamProcessor.safe_cast_to_string(self.properties[field], column_names[field][1], self.destination_type) for field in column_names]

    @staticmethod
    def safe_cast_to_string(definition: Dict, column_name: str, destination_type: DestinationType) -> str:
        if False:
            print('Hello World!')
        "\n        Note that the result from this static method should always be used within a\n        jinja context (for example, from jinja macro surrogate_key call)\n\n        The jinja_remove function is necessary because of Oracle database, some columns\n        are created with {{ quote('column_name') }} and reused the same fields for this\n        operation. Because the quote is injected inside a jinja macro we need to remove\n        the curly brackets.\n        "
        if 'type' not in definition:
            col = column_name
        elif is_boolean(definition['type'], definition):
            col = f'boolean_to_string({column_name})'
        elif is_array(definition['type']):
            col = f'array_to_string({column_name})'
        elif is_object(definition['type']):
            col = f'object_to_string({column_name})'
        else:
            col = column_name
        if destination_type == DestinationType.ORACLE:
            quote_in_parenthesis = re.compile('quote\\((.*)\\)')
            return remove_jinja(col) if quote_in_parenthesis.findall(col) else col
        return col

    def generate_scd_type_2_model(self, from_table: str, column_names: Dict[str, Tuple[str, str]]) -> Any:
        if False:
            i = 10
            return i + 15
        "\n        This model pulls data from the ID-hashing model and appends it to a log of record updates. When inserting an update to a record, it also\n        checks whether that record had a previously-existing row in the SCD model; if it does, then that previous row's end_at column is set to\n        the new update's start_at.\n\n        See the docs for more details: https://docs.airbyte.com/understanding-airbyte/basic-normalization#normalization-metadata-columns\n        "
        cursor_field = self.get_cursor_field(column_names)
        order_null = f'is null asc,\n            {cursor_field} desc'
        if self.destination_type.value == DestinationType.ORACLE.value:
            order_null = 'desc nulls last'
        if self.destination_type.value == DestinationType.MSSQL.value:
            order_null = 'desc'
        lag_begin = 'lag'
        lag_end = ''
        input_data_table = 'input_data'
        if self.destination_type == DestinationType.CLICKHOUSE:
            lag_begin = 'anyOrNull'
            lag_end = '      ROWS BETWEEN 1 PRECEDING AND 1 PRECEDING'
            input_data_table = 'input_data_with_active_row_num'
        enable_left_join_null = ''
        cast_begin = 'cast('
        cast_as = ' as '
        cast_end = ')'
        if self.destination_type == DestinationType.CLICKHOUSE:
            enable_left_join_null = '--'
            cast_begin = 'accurateCastOrNull('
            cast_as = ", '"
            cast_end = "')"
        cdc_active_row_pattern = ''
        cdc_updated_order_pattern = ''
        cdc_cols = ''
        quoted_cdc_cols = ''
        if '_ab_cdc_deleted_at' in column_names.keys():
            col_cdc_deleted_at = self.name_transformer.normalize_column_name('_ab_cdc_deleted_at')
            col_cdc_updated_at = self.name_transformer.normalize_column_name('_ab_cdc_updated_at')
            quoted_col_cdc_deleted_at = self.name_transformer.normalize_column_name('_ab_cdc_deleted_at', in_jinja=True)
            quoted_col_cdc_updated_at = self.name_transformer.normalize_column_name('_ab_cdc_updated_at', in_jinja=True)
            cdc_active_row_pattern = f' and {col_cdc_deleted_at} is null'
            cdc_updated_order_pattern = f'\n            {col_cdc_updated_at} desc,'
            cdc_cols = f', {cast_begin}{col_cdc_deleted_at}{cast_as}' + '{{ dbt_utils.type_string() }}' + f'{cast_end}' + f', {cast_begin}{col_cdc_updated_at}{cast_as}' + '{{ dbt_utils.type_string() }}' + f'{cast_end}'
            quoted_cdc_cols = f', {quoted_col_cdc_deleted_at}, {quoted_col_cdc_updated_at}'
        if '_ab_cdc_log_pos' in column_names.keys():
            col_cdc_log_pos = self.name_transformer.normalize_column_name('_ab_cdc_log_pos')
            quoted_col_cdc_log_pos = self.name_transformer.normalize_column_name('_ab_cdc_log_pos', in_jinja=True)
            cdc_updated_order_pattern += f'\n            {col_cdc_log_pos} desc,'
            cdc_cols += ''.join([', ', cast_begin, col_cdc_log_pos, cast_as, '{{ dbt_utils.type_string() }}', cast_end])
            quoted_cdc_cols += f', {quoted_col_cdc_log_pos}'
        if '_ab_cdc_lsn' in column_names.keys():
            col_cdc_lsn = self.name_transformer.normalize_column_name('_ab_cdc_lsn')
            quoted_col_cdc_lsn = self.name_transformer.normalize_column_name('_ab_cdc_lsn', in_jinja=True)
            cdc_updated_order_pattern += f'\n            {col_cdc_lsn} desc,'
            cdc_cols += ''.join([', ', cast_begin, col_cdc_lsn, cast_as, '{{ dbt_utils.type_string() }}', cast_end])
            quoted_cdc_cols += f', {quoted_col_cdc_lsn}'
        if self.destination_type == DestinationType.BIGQUERY and self.get_cursor_field_property_name(column_names) != self.airbyte_emitted_at and is_number(self.properties[self.get_cursor_field_property_name(column_names)]['type']):
            airbyte_start_at_string = cast_begin + self.name_transformer.normalize_column_name('_airbyte_start_at') + cast_as + '{{ dbt_utils.type_string() }}' + cast_end
        else:
            airbyte_start_at_string = self.name_transformer.normalize_column_name('_airbyte_start_at')
        jinja_variables = {'active_row': self.name_transformer.normalize_column_name('_airbyte_active_row'), 'airbyte_end_at': self.name_transformer.normalize_column_name('_airbyte_end_at'), 'airbyte_row_num': self.name_transformer.normalize_column_name('_airbyte_row_num'), 'airbyte_start_at': self.name_transformer.normalize_column_name('_airbyte_start_at'), 'airbyte_start_at_string': airbyte_start_at_string, 'airbyte_unique_key_scd': self.name_transformer.normalize_column_name(f'{self.airbyte_unique_key}_scd'), 'cdc_active_row': cdc_active_row_pattern, 'cdc_cols': cdc_cols, 'cdc_updated_at_order': cdc_updated_order_pattern, 'col_ab_id': self.get_ab_id(), 'col_emitted_at': self.get_emitted_at(), 'col_normalized_at': self.get_normalized_at(), 'cursor_field': cursor_field, 'enable_left_join_null': enable_left_join_null, 'fields': self.list_fields(column_names), 'from_table': from_table, 'hash_id': self.hash_id(), 'incremental_clause': self.get_incremental_clause('this'), 'input_data_table': input_data_table, 'lag_begin': lag_begin, 'lag_end': lag_end, 'order_null': order_null, 'parent_hash_id': self.parent_hash_id(), 'primary_key_partition': self.get_primary_key_partition(column_names), 'primary_keys': self.list_primary_keys(column_names), 'quoted_airbyte_row_num': self.name_transformer.normalize_column_name('_airbyte_row_num', in_jinja=True), 'quoted_airbyte_start_at': self.name_transformer.normalize_column_name('_airbyte_start_at', in_jinja=True), 'quoted_cdc_cols': quoted_cdc_cols, 'quoted_col_emitted_at': self.get_emitted_at(in_jinja=True), 'quoted_unique_key': self.get_unique_key(in_jinja=True), 'sql_table_comment': self.sql_table_comment(include_from_table=True), 'unique_key': self.get_unique_key()}
        if self.destination_type == DestinationType.CLICKHOUSE:
            clickhouse_active_row_sql = Template('\ninput_data_with_active_row_num as (\n    select *,\n      row_number() over (\n        partition by {{ primary_key_partition | join(", ") }}\n        order by\n            {{ cursor_field }} {{ order_null }},{{ cdc_updated_at_order }}\n            {{ col_emitted_at }} desc\n      ) as _airbyte_active_row_num\n    from input_data\n),').render(jinja_variables)
            jinja_variables['clickhouse_active_row_sql'] = clickhouse_active_row_sql
            scd_columns_sql = Template('\n      case when _airbyte_active_row_num = 1{{ cdc_active_row }} then 1 else 0 end as {{ active_row }},\n      {{ lag_begin }}({{ cursor_field }}) over (\n        partition by {{ primary_key_partition | join(", ") }}\n        order by\n            {{ cursor_field }} {{ order_null }},{{ cdc_updated_at_order }}\n            {{ col_emitted_at }} desc\n      {{ lag_end }}) as {{ airbyte_end_at }}').render(jinja_variables)
            jinja_variables['scd_columns_sql'] = scd_columns_sql
        else:
            scd_columns_sql = Template('\n      lag({{ cursor_field }}) over (\n        partition by {{ primary_key_partition | join(", ") }}\n        order by\n            {{ cursor_field }} {{ order_null }},{{ cdc_updated_at_order }}\n            {{ col_emitted_at }} desc\n      ) as {{ airbyte_end_at }},\n      case when row_number() over (\n        partition by {{ primary_key_partition | join(", ") }}\n        order by\n            {{ cursor_field }} {{ order_null }},{{ cdc_updated_at_order }}\n            {{ col_emitted_at }} desc\n      ) = 1{{ cdc_active_row }} then 1 else 0 end as {{ active_row }}').render(jinja_variables)
            jinja_variables['scd_columns_sql'] = scd_columns_sql
        sql = Template('\n-- depends_on: {{ from_table }}\nwith\n{{ \'{% if is_incremental() %}\' }}\nnew_data as (\n    -- retrieve incremental "new" data\n    select\n        *\n    from {{\'{{\'}} {{ from_table }}  {{\'}}\'}}\n    {{ sql_table_comment }}\n    where 1 = 1\n    {{ incremental_clause }}\n),\nnew_data_ids as (\n    -- build a subset of {{ unique_key }} from rows that are new\n    select distinct\n        {{ \'{{\' }} dbt_utils.surrogate_key([\n{%- for primary_key in primary_keys %}\n            {{ primary_key }},\n{%- endfor %}\n        ]) {{ \'}}\' }} as {{ unique_key }}\n    from new_data\n),\nempty_new_data as (\n    -- build an empty table to only keep the table\'s column types\n    select * from new_data where 1 = 0\n),\nprevious_active_scd_data as (\n    -- retrieve "incomplete old" data that needs to be updated with an end date because of new changes\n    select\n        {{ \'{{\' }} star_intersect({{ from_table }}, this, from_alias=\'inc_data\', intersect_alias=\'this_data\') {{ \'}}\' }}\n    from {{ \'{{ this }}\' }} as this_data\n    -- make a join with new_data using primary key to filter active data that need to be updated only\n    join new_data_ids on this_data.{{ unique_key }} = new_data_ids.{{ unique_key }}\n    -- force left join to NULL values (we just need to transfer column types only for the star_intersect macro on schema changes)\n    {{ enable_left_join_null }}left join empty_new_data as inc_data on this_data.{{ col_ab_id }} = inc_data.{{ col_ab_id }}\n    where {{ active_row }} = 1\n),\ninput_data as (\n    select {{ \'{{\' }} dbt_utils.star({{ from_table }}) {{ \'}}\' }} from new_data\n    union all\n    select {{ \'{{\' }} dbt_utils.star({{ from_table }}) {{ \'}}\' }} from previous_active_scd_data\n),\n{{ \'{% else %}\' }}\ninput_data as (\n    select *\n    from {{\'{{\'}} {{ from_table }}  {{\'}}\'}}\n    {{ sql_table_comment }}\n),\n{{ \'{% endif %}\' }}\n{{ clickhouse_active_row_sql }}\nscd_data as (\n    -- SQL model to build a Type 2 Slowly Changing Dimension (SCD) table for each record identified by their primary key\n    select\n{%- if parent_hash_id %}\n      {{ parent_hash_id }},\n{%- endif %}\n      {{ \'{{\' }} dbt_utils.surrogate_key([\n{%- for primary_key in primary_keys %}\n      {{ primary_key }},\n{%- endfor %}\n      ]) {{ \'}}\' }} as {{ unique_key }},\n{%- for field in fields %}\n      {{ field }},\n{%- endfor %}\n      {{ cursor_field }} as {{ airbyte_start_at }},\n      {{ scd_columns_sql }},\n      {{ col_ab_id }},\n      {{ col_emitted_at }},\n      {{ hash_id }}\n    from {{ input_data_table }}\n),\ndedup_data as (\n    select\n        -- we need to ensure de-duplicated rows for merge/update queries\n        -- additionally, we generate a unique key for the scd table\n        row_number() over (\n            partition by\n                {{ unique_key }},\n                {{ airbyte_start_at_string }},\n                {{ col_emitted_at }}{{ cdc_cols }}\n            order by {{ active_row }} desc, {{ col_ab_id }}\n        ) as {{ airbyte_row_num }},\n        {{ \'{{\' }} dbt_utils.surrogate_key([\n          {{ quoted_unique_key }},\n          {{ quoted_airbyte_start_at }},\n          {{ quoted_col_emitted_at }}{{ quoted_cdc_cols }}\n        ]) {{ \'}}\' }} as {{ airbyte_unique_key_scd }},\n        scd_data.*\n    from scd_data\n)\nselect\n{%- if parent_hash_id %}\n    {{ parent_hash_id }},\n{%- endif %}\n    {{ unique_key }},\n    {{ airbyte_unique_key_scd }},\n{%- for field in fields %}\n    {{ field }},\n{%- endfor %}\n    {{ airbyte_start_at }},\n    {{ airbyte_end_at }},\n    {{ active_row }},\n    {{ col_ab_id }},\n    {{ col_emitted_at }},\n    {{ \'{{ current_timestamp() }}\' }} as {{ col_normalized_at }},\n    {{ hash_id }}\nfrom dedup_data where {{ airbyte_row_num }} = 1\n').render(jinja_variables)
        return sql

    def get_cursor_field_property_name(self, column_names: Dict[str, Tuple[str, str]]) -> str:
        if False:
            i = 10
            return i + 15
        if not self.cursor_field:
            if '_ab_cdc_updated_at' in column_names.keys():
                return '_ab_cdc_updated_at'
            elif '_ab_cdc_log_pos' in column_names.keys():
                return '_ab_cdc_log_pos'
            elif '_ab_cdc_lsn' in column_names.keys():
                return '_ab_cdc_lsn'
            else:
                return self.airbyte_emitted_at
        elif len(self.cursor_field) == 1:
            return self.cursor_field[0]
        else:
            raise ValueError(f"Unsupported nested cursor field {'.'.join(self.cursor_field)} for stream {self.stream_name}")

    def get_cursor_field(self, column_names: Dict[str, Tuple[str, str]], in_jinja: bool=False) -> str:
        if False:
            print('Hello World!')
        if not self.cursor_field:
            cursor = self.name_transformer.normalize_column_name(self.get_cursor_field_property_name(column_names), in_jinja)
        elif len(self.cursor_field) == 1:
            if not is_airbyte_column(self.cursor_field[0]):
                cursor = column_names[self.cursor_field[0]][0]
            else:
                cursor = self.cursor_field[0]
        else:
            raise ValueError(f"Unsupported nested cursor field {'.'.join(self.cursor_field)} for stream {self.stream_name}")
        return cursor

    def list_primary_keys(self, column_names: Dict[str, Tuple[str, str]]) -> List[str]:
        if False:
            i = 10
            return i + 15
        primary_keys = []
        for key_path in self.primary_key:
            if len(key_path) == 1:
                primary_keys.append(column_names[key_path[0]][1])
            else:
                raise ValueError(f"Unsupported nested path {'.'.join(key_path)} for stream {self.stream_name}")
        return primary_keys

    def get_primary_key_partition(self, column_names: Dict[str, Tuple[str, str]]) -> List[str]:
        if False:
            i = 10
            return i + 15
        if self.primary_key and len(self.primary_key) > 0:
            return [self.get_primary_key_from_path(column_names, path) for path in self.primary_key]
        else:
            raise ValueError(f'No primary key specified for stream {self.stream_name}')

    def get_primary_key_from_path(self, column_names: Dict[str, Tuple[str, str]], path: List[str]) -> str:
        if False:
            return 10
        if path and len(path) == 1:
            field = path[0]
            if not is_airbyte_column(field):
                if 'type' in self.properties[field]:
                    property_type = self.properties[field]['type']
                else:
                    property_type = 'object'
                if is_number(property_type) or is_object(property_type):
                    return f"cast({column_names[field][0]} as {jinja_call('dbt_utils.type_string()')})"
                else:
                    return column_names[field][0]
            else:
                return f"cast({field} as {jinja_call('dbt_utils.type_string()')})"
        elif path:
            raise ValueError(f"Unsupported nested path {'.'.join(path)} for stream {self.stream_name}")
        else:
            raise ValueError(f'No path specified for stream {self.stream_name}')

    def generate_final_model(self, from_table: str, column_names: Dict[str, Tuple[str, str]], unique_key: str='') -> Any:
        if False:
            return 10
        '\n        This is the table that the user actually wants. In addition to the columns that the source outputs, it has some additional metadata columns;\n        see the basic normalization docs for an explanation: https://docs.airbyte.com/understanding-airbyte/basic-normalization#normalization-metadata-columns\n        '
        template = Template("\n-- Final base SQL model\n-- depends_on: {{ from_table }}\nselect\n{%- if parent_hash_id %}\n    {{ parent_hash_id }},\n{%- endif %}\n{%- if unique_key %}\n    {{ unique_key }},\n{%- endif %}\n{%- for field in fields %}\n    {{ field }},\n{%- endfor %}\n    {{ col_ab_id }},\n    {{ col_emitted_at }},\n    {{ '{{ current_timestamp() }}' }} as {{ col_normalized_at }},\n    {{ hash_id }}\nfrom {{ from_table }}\n{{ sql_table_comment }}\nwhere 1 = 1\n    ")
        sql = template.render(col_ab_id=self.get_ab_id(), col_emitted_at=self.get_emitted_at(), col_normalized_at=self.get_normalized_at(), parent_hash_id=self.parent_hash_id(), fields=self.list_fields(column_names), hash_id=self.hash_id(), from_table=jinja_call(from_table), sql_table_comment=self.sql_table_comment(include_from_table=True), unique_key=unique_key)
        return sql

    @staticmethod
    def is_incremental_mode(destination_sync_mode: DestinationSyncMode) -> bool:
        if False:
            while True:
                i = 10
        return destination_sync_mode.value in [DestinationSyncMode.append.value, DestinationSyncMode.append_dedup.value]

    def add_incremental_clause(self, sql_query: str) -> Any:
        if False:
            i = 10
            return i + 15
        template = Template('\n{{ sql_query }}\n{{ incremental_clause }}\n    ')
        sql = template.render(sql_query=sql_query, incremental_clause=self.get_incremental_clause('this'))
        return sql

    def get_incremental_clause(self, tablename: str) -> Any:
        if False:
            i = 10
            return i + 15
        return self.get_incremental_clause_for_column(tablename, self.get_emitted_at(in_jinja=True))

    def get_incremental_clause_for_column(self, tablename: str, column: str) -> Any:
        if False:
            return 10
        return '{{ incremental_clause(' + column + ', ' + tablename + ') }}'

    @staticmethod
    def list_fields(column_names: Dict[str, Tuple[str, str]]) -> List[str]:
        if False:
            i = 10
            return i + 15
        return [column_names[field][0] for field in column_names]

    def add_to_outputs(self, sql: str, materialization_mode: TableMaterializationType, is_intermediate: bool=True, suffix: str='', unique_key: str='', subdir: str='', partition_by: PartitionScheme=PartitionScheme.DEFAULT) -> str:
        if False:
            i = 10
            return i + 15

        def wrap_in_quotes(s: str) -> str:
            if False:
                print('Hello World!')
            return '"' + s + '"'
        schema = self.get_schema(is_intermediate)
        truncate_name = self.destination_type == DestinationType.MYSQL or self.destination_type == DestinationType.TIDB or self.destination_type == DestinationType.DUCKDB
        table_name = self.tables_registry.get_table_name(schema, self.json_path, self.stream_name, suffix, truncate_name)
        file_name = self.tables_registry.get_file_name(schema, self.json_path, self.stream_name, suffix, truncate_name)
        file = f'{file_name}.sql'
        output = os.path.join(materialization_mode.value, subdir, self.schema, file)
        config = self.get_model_partition_config(partition_by, unique_key)
        if file_name != table_name:
            config['alias'] = f'"{table_name}"'
        if self.destination_type == DestinationType.ORACLE:
            config['schema'] = f'"{self.default_schema}"'
        else:
            config['schema'] = f'"{schema}"'
        if self.is_incremental_mode(self.destination_sync_mode):
            stg_schema = self.get_schema(True)
            stg_table = self.tables_registry.get_file_name(schema, self.json_path, self.stream_name, 'stg', truncate_name)
            if self.name_transformer.needs_quotes(stg_table):
                stg_table = jinja_call(self.name_transformer.apply_quote(stg_table))
            if suffix == 'scd':
                hooks = []
                final_table_name = self.tables_registry.get_file_name(schema, self.json_path, self.stream_name, '', truncate_name)
                active_row_column_name = self.name_transformer.normalize_column_name('_airbyte_active_row')
                clickhouse_nullable_join_setting = ''
                if self.destination_type == DestinationType.CLICKHOUSE:
                    delete_statement = 'alter table {{ final_table_relation }} delete'
                    unique_key_reference = self.get_unique_key(in_jinja=False)
                    noop_delete_statement = 'alter table {{ this }} delete where 1=0'
                    clickhouse_nullable_join_setting = 'SETTINGS join_use_nulls=1'
                elif self.destination_type == DestinationType.BIGQUERY:
                    delete_statement = 'delete from {{ final_table_relation }} final_table'
                    unique_key_reference = 'final_table.' + self.get_unique_key(in_jinja=False)
                    noop_delete_statement = 'delete from {{ this }} where 1=0'
                else:
                    delete_statement = 'delete from {{ final_table_relation }}'
                    unique_key_reference = '{{ final_table_relation }}.' + self.get_unique_key(in_jinja=False)
                    noop_delete_statement = 'delete from {{ this }} where 1=0'
                deletion_hook = Template("\n                    {{ '{%' }}\n                    set final_table_relation = adapter.get_relation(\n                            database=this.database,\n                            schema=this.schema,\n                            identifier='{{ final_table_name }}'\n                        )\n                    {{ '%}' }}\n                    {{ '{#' }}\n                    If the final table doesn't exist, then obviously we can't delete anything from it.\n                    Also, after a reset, the final table is created without the _airbyte_unique_key column (this column is created during the first sync)\n                    So skip this deletion if the column doesn't exist. (in this case, the table is guaranteed to be empty anyway)\n                    {{ '#}' }}\n                    {{ '{%' }}\n                    if final_table_relation is not none and {{ quoted_unique_key }} in adapter.get_columns_in_relation(final_table_relation)|map(attribute='name')\n                    {{ '%}' }}\n\n                    -- Delete records which are no longer active:\n                    -- This query is equivalent, but the left join version is more performant:\n                    -- delete from final_table where unique_key in (\n                    --     select unique_key from scd_table where 1 = 1 <incremental_clause(normalized_at, final_table)>\n                    -- ) and unique_key not in (\n                    --     select unique_key from scd_table where active_row = 1 <incremental_clause(normalized_at, final_table)>\n                    -- )\n                    -- We're incremental against normalized_at rather than emitted_at because we need to fetch the SCD\n                    -- entries that were _updated_ recently. This is because a deleted record will have an SCD record\n                    -- which was emitted a long time ago, but recently re-normalized to have active_row = 0.\n                    {{ delete_statement }} where {{ unique_key_reference }} in (\n                        select recent_records.unique_key\n                        from (\n                                select distinct {{ unique_key }} as unique_key\n                                from {{ '{{ this }}' }}\n                                where 1=1 {{ normalized_at_incremental_clause }}\n                            ) recent_records\n                            left join (\n                                select {{ unique_key }} as unique_key, count({{ unique_key }}) as active_count\n                                from {{ '{{ this }}' }}\n                                where {{ active_row_column_name }} = 1 {{ normalized_at_incremental_clause }}\n                                group by {{ unique_key }}\n                            ) active_counts\n                            on recent_records.unique_key = active_counts.unique_key\n                        where active_count is null or active_count = 0\n                    )\n                    {{ '{% else %}' }}\n                    -- We have to have a non-empty query, so just do a noop delete\n                    {{ noop_delete_statement }}\n                    {{ '{% endif %}' }}\n                    ").render(delete_statement=delete_statement, noop_delete_statement=noop_delete_statement, final_table_name=final_table_name, unique_key=self.get_unique_key(in_jinja=False), quoted_unique_key=self.get_unique_key(in_jinja=True), active_row_column_name=active_row_column_name, normalized_at_incremental_clause=self.get_incremental_clause_for_column("{} + '.' + {}".format(self.name_transformer.apply_quote('this.schema', literal=False), self.name_transformer.apply_quote(final_table_name)), self.get_normalized_at(in_jinja=True)), unique_key_reference=unique_key_reference, clickhouse_nullable_join_setting=clickhouse_nullable_join_setting)
                hooks.append(deletion_hook)
                if self.destination_type.value == DestinationType.POSTGRES.value:
                    hooks.append(f'delete from {stg_schema}.{stg_table} where {self.airbyte_emitted_at} != (select max({self.airbyte_emitted_at}) from {stg_schema}.{stg_table})')
                else:
                    hooks.append(f'drop view {stg_schema}.{stg_table}')
                config['post_hook'] = '[' + ','.join(map(wrap_in_quotes, hooks)) + ']'
            else:
                sql = self.add_incremental_clause(sql)
        elif self.destination_sync_mode == DestinationSyncMode.overwrite:
            if suffix == '' and (not is_intermediate):
                scd_table_name = self.tables_registry.get_table_name(schema, self.json_path, self.stream_name, 'scd', truncate_name)
                print(f'  Adding drop table hook for {scd_table_name} to {file_name}')
                hooks = [Template("\n                    {{ '{%' }}\n                        set scd_table_relation = adapter.get_relation(\n                            database=this.database,\n                            schema=this.schema,\n                            identifier='{{ scd_table_name }}'\n                        )\n                    {{ '%}' }}\n                    {{ '{%' }}\n                        if scd_table_relation is not none\n                    {{ '%}' }}\n                    {{ '{%' }}\n                            do adapter.drop_relation(scd_table_relation)\n                    {{ '%}' }}\n                    {{ '{% endif %}' }}\n                        ").render(scd_table_name=scd_table_name)]
                config['post_hook'] = '[' + ','.join(map(wrap_in_quotes, hooks)) + ']'
        template = Template("\n{{ '{{' }} config(\n{%- for key in config %}\n    {{ key }} = {{ config[key] }},\n{%- endfor %}\n    tags = [ {{ tags }} ]\n) {{ '}}' }}\n{{ sql }}\n    ")
        self.sql_outputs[output] = template.render(config=config, sql=sql, tags=self.get_model_tags(is_intermediate))
        json_path = self.current_json_path()
        print(f'  Generating {output} from {json_path}')
        self.models_to_source[file_name] = self.get_stream_source()
        return str(dbt_macro.Ref(file_name))

    def get_model_materialization_mode(self, is_intermediate: bool, column_count: int=0) -> TableMaterializationType:
        if False:
            print('Hello World!')
        if is_intermediate:
            if column_count <= MAXIMUM_COLUMNS_TO_USE_EPHEMERAL:
                return TableMaterializationType.CTE
            else:
                return TableMaterializationType.VIEW
        elif self.is_incremental_mode(self.destination_sync_mode):
            return TableMaterializationType.INCREMENTAL
        else:
            return TableMaterializationType.TABLE

    def get_model_partition_config(self, partition_by: PartitionScheme, unique_key: str) -> Dict:
        if False:
            while True:
                i = 10
        '\n        Defines partition, clustering and unique key parameters for each destination.\n        The goal of these are to make read more performant.\n\n        In general, we need to do lookups on the last emitted_at column to know if a record is freshly produced and need to be\n        incrementally processed or not.\n        But in certain models, such as SCD tables for example, we also need to retrieve older data to update their type 2 SCD end_dates,\n        thus a different partitioning scheme is used to optimize that use case.\n        '
        config = {}
        if self.destination_type == DestinationType.BIGQUERY:
            if partition_by == PartitionScheme.UNIQUE_KEY:
                config['cluster_by'] = f'["{self.airbyte_unique_key}","{self.airbyte_emitted_at}"]'
            elif partition_by == PartitionScheme.ACTIVE_ROW:
                config['cluster_by'] = f'["{self.airbyte_unique_key}_scd","{self.airbyte_emitted_at}"]'
            else:
                config['cluster_by'] = f'"{self.airbyte_emitted_at}"'
            if partition_by == PartitionScheme.ACTIVE_ROW:
                config['partition_by'] = '{"field": "_airbyte_active_row", "data_type": "int64", "range": {"start": 0, "end": 1, "interval": 1}}'
            elif partition_by == PartitionScheme.NOTHING:
                pass
            else:
                config['partition_by'] = '{"field": "' + self.airbyte_emitted_at + '", "data_type": "timestamp", "granularity": "day"}'
        elif self.destination_type == DestinationType.POSTGRES:
            if partition_by == PartitionScheme.ACTIVE_ROW:
                config['indexes'] = "[{'columns':['_airbyte_active_row','" + self.airbyte_unique_key + "_scd','" + self.airbyte_emitted_at + "'],'type': 'btree'}]"
            elif partition_by == PartitionScheme.UNIQUE_KEY:
                config['indexes'] = "[{'columns':['" + self.airbyte_unique_key + "'],'unique':True}]"
            else:
                config['indexes'] = "[{'columns':['" + self.airbyte_emitted_at + "'],'type':'btree'}]"
        elif self.destination_type == DestinationType.REDSHIFT:
            if partition_by == PartitionScheme.ACTIVE_ROW:
                config['sort'] = f'["_airbyte_active_row", "{self.airbyte_unique_key}_scd", "{self.airbyte_emitted_at}"]'
            elif partition_by == PartitionScheme.UNIQUE_KEY:
                config['sort'] = f'["{self.airbyte_unique_key}", "{self.airbyte_emitted_at}"]'
            elif partition_by == PartitionScheme.NOTHING:
                pass
            else:
                config['sort'] = f'"{self.airbyte_emitted_at}"'
        elif self.destination_type == DestinationType.SNOWFLAKE:
            if partition_by == PartitionScheme.ACTIVE_ROW:
                config['cluster_by'] = f'["_AIRBYTE_ACTIVE_ROW", "{self.airbyte_unique_key.upper()}_SCD", "{self.airbyte_emitted_at.upper()}"]'
            elif partition_by == PartitionScheme.UNIQUE_KEY:
                config['cluster_by'] = f'["{self.airbyte_unique_key.upper()}", "{self.airbyte_emitted_at.upper()}"]'
            elif partition_by == PartitionScheme.NOTHING:
                pass
            else:
                config['cluster_by'] = f'["{self.airbyte_emitted_at.upper()}"]'
        if unique_key:
            config['unique_key'] = f'"{unique_key}"'
        elif not self.parent:
            config['unique_key'] = self.get_ab_id(in_jinja=True)
        return config

    def get_model_tags(self, is_intermediate: bool) -> str:
        if False:
            while True:
                i = 10
        tags = ''
        if self.parent:
            tags += 'nested'
        else:
            tags += 'top-level'
        if is_intermediate:
            tags += '-intermediate'
        return f'"{tags}"'

    def get_schema(self, is_intermediate: bool) -> str:
        if False:
            for i in range(10):
                print('nop')
        if is_intermediate:
            return self.raw_schema
        else:
            return self.schema

    def current_json_path(self) -> str:
        if False:
            while True:
                i = 10
        return '/'.join(self.json_path)

    def normalized_stream_name(self) -> str:
        if False:
            return 10
        '\n        This is the normalized name of this stream to be used as a table (different as referring it as a column).\n        Note that it might not be the actual table name in case of collisions with other streams (see actual_table_name)...\n        '
        return self.name_transformer.normalize_table_name(self.stream_name)

    def sql_table_comment(self, include_from_table: bool=False) -> str:
        if False:
            while True:
                i = 10
        result = f'-- {self.normalized_stream_name()}'
        if len(self.json_path) > 1:
            result += f' at {self.current_json_path()}'
        if include_from_table:
            from_table = jinja_call(self.from_table)
            result += f' from {from_table}'
        return result

    def hash_id(self, in_jinja: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        hash_id_col = f'_airbyte_{self.normalized_stream_name()}_hashid'
        if self.parent:
            if self.normalized_stream_name().lower() == self.parent.stream_name.lower():
                level = len(self.json_path)
                hash_id_col = f'_airbyte_{self.normalized_stream_name()}_{level}_hashid'
        return self.name_transformer.normalize_column_name(hash_id_col, in_jinja)

    def parent_hash_id(self, in_jinja: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        if self.parent:
            return self.parent.hash_id(in_jinja)
        return ''

    def unnesting_before_query(self, from_table: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.parent and self.is_nested_array:
            parent_stream_name = f"'{self.parent.normalized_stream_name()}'"
            quoted_field = self.name_transformer.normalize_column_name(self.stream_name, in_jinja=True)
            return jinja_call(f'unnest_cte({from_table}, {parent_stream_name}, {quoted_field})')
        return ''

    def unnesting_from(self) -> str:
        if False:
            while True:
                i = 10
        if self.parent:
            if self.is_nested_array:
                parent_stream_name = f"'{self.parent.normalized_stream_name()}'"
                quoted_field = self.name_transformer.normalize_column_name(self.stream_name, in_jinja=True)
                return jinja_call(f'cross_join_unnest({parent_stream_name}, {quoted_field})')
        return ''

    def unnesting_where(self) -> str:
        if False:
            while True:
                i = 10
        if self.parent:
            column_name = self.name_transformer.normalize_column_name(self.stream_name)
            return f'and {column_name} is not null'
        return ''

def find_properties_object(path: List[str], field: str, properties) -> Dict[str, Dict]:
    if False:
        for i in range(10):
            print('nop')
    '\n    This function is trying to look for a nested "properties" node under the current JSON node to\n    identify all nested objects.\n\n    @param path JSON path traversed so far to arrive to this node\n    @param field is the current field being considered in the Json Tree\n    @param properties is the child tree of properties of the current field being searched\n    '
    result = {}
    current_path = path + [field]
    current = '_'.join(current_path)
    if isinstance(properties, str) or isinstance(properties, int):
        return {}
    elif 'items' in properties:
        return find_properties_object(path, field, properties['items'])
    elif 'properties' in properties:
        return {current: properties['properties']}
    elif 'type' in properties and is_simple_property(properties):
        return {current: {}}
    elif isinstance(properties, dict):
        for key in properties.keys():
            child = find_properties_object(path=current_path, field=key, properties=properties[key])
            if child:
                result.update(child)
    elif isinstance(properties, list):
        for item in properties:
            child = find_properties_object(path=current_path, field=field, properties=item)
            if child:
                result.update(child)
    return result