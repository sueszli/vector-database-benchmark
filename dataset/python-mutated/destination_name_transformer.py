import unicodedata as ud
from re import match, sub
from normalization.destination_type import DestinationType
from normalization.transform_catalog.reserved_keywords import is_reserved_keyword
from normalization.transform_catalog.utils import jinja_call
DESTINATION_SIZE_LIMITS = {DestinationType.BIGQUERY.value: 1024, DestinationType.SNOWFLAKE.value: 255, DestinationType.REDSHIFT.value: 127, DestinationType.POSTGRES.value: 63, DestinationType.MYSQL.value: 64, DestinationType.ORACLE.value: 128, DestinationType.MSSQL.value: 64, DestinationType.CLICKHOUSE.value: 63, DestinationType.TIDB.value: 64, DestinationType.DUCKDB.value: 64}
TRUNCATE_DBT_RESERVED_SIZE = 12
TRUNCATE_RESERVED_SIZE = 8

class DestinationNameTransformer:
    """
    Handles naming conventions in destinations for all kind of sql identifiers:
    - schema
    - table
    - column
    """

    def __init__(self, destination_type: DestinationType):
        if False:
            print('Hello World!')
        '\n        @param destination_type is the destination type of warehouse\n        '
        self.destination_type: DestinationType = destination_type

    def needs_quotes(self, input_name: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        @param input_name to test if it needs to manipulated with quotes or not\n        '
        if is_reserved_keyword(input_name, self.destination_type):
            return True
        if self.destination_type.value == DestinationType.BIGQUERY.value:
            return False
        if self.destination_type.value == DestinationType.ORACLE.value and input_name.startswith('_'):
            return True
        doesnt_start_with_alphaunderscore = match('[^A-Za-z_]', input_name[0]) is not None
        contains_non_alphanumeric = match('.*[^A-Za-z0-9_].*', input_name) is not None
        return doesnt_start_with_alphaunderscore or contains_non_alphanumeric

    def normalize_schema_name(self, schema_name: str, in_jinja: bool=False, truncate: bool=True) -> str:
        if False:
            i = 10
            return i + 15
        "\n        @param schema_name is the schema to normalize\n        @param in_jinja is a boolean to specify if the returned normalized will be used inside a jinja macro or not\n        @param truncate force ignoring truncate operation on resulting normalized name. For example, if we don't\n        control how the name would be normalized\n        "
        if self.destination_type == DestinationType.ORACLE and schema_name.startswith('_'):
            schema_name = schema_name[1:]
        return self.__normalize_non_column_identifier_name(input_name=schema_name, in_jinja=in_jinja, truncate=truncate)

    def normalize_table_name(self, table_name: str, in_jinja: bool=False, truncate: bool=True, conflict: bool=False, conflict_level: int=0) -> str:
        if False:
            for i in range(10):
                print('nop')
        "\n        @param table_name is the table to normalize\n        @param in_jinja is a boolean to specify if the returned normalized will be used inside a jinja macro or not\n        @param truncate force ignoring truncate operation on resulting normalized name. For example, if we don't\n        control how the name would be normalized\n        @param conflict if there is a conflict between stream name and fields\n        @param conflict_level is the json_path level conflict happened\n        "
        if self.destination_type == DestinationType.ORACLE and table_name.startswith('_'):
            table_name = table_name[1:]
        return self.__normalize_non_column_identifier_name(input_name=table_name, in_jinja=in_jinja, truncate=truncate, conflict=conflict, conflict_level=conflict_level)

    def normalize_column_name(self, column_name: str, in_jinja: bool=False, truncate: bool=True, conflict: bool=False, conflict_level: int=0) -> str:
        if False:
            while True:
                i = 10
        "\n        @param column_name is the column to normalize\n        @param in_jinja is a boolean to specify if the returned normalized will be used inside a jinja macro or not\n        @param truncate force ignoring truncate operation on resulting normalized name. For example, if we don't\n        control how the name would be normalized\n        @param conflict if there is a conflict between stream name and fields\n        @param conflict_level is the json_path level conflict happened\n        "
        return self.__normalize_identifier_name(column_name=column_name, in_jinja=in_jinja, truncate=truncate, conflict=conflict, conflict_level=conflict_level)

    def truncate_identifier_name(self, input_name: str, custom_limit: int=-1, conflict: bool=False, conflict_level: int=0) -> str:
        if False:
            return 10
        '\n        @param input_name is the identifier name to middle truncate\n        @param custom_limit uses a custom length as the max instead of the destination max length\n        @param conflict if there is a conflict between stream name and fields\n        @param conflict_level is the json_path level conflict happened\n        '
        limit = custom_limit - 1 if custom_limit > 0 else self.get_name_max_length()
        if limit < len(input_name):
            middle = round(limit / 2)
            prefix = input_name[:limit - middle - 1]
            suffix = input_name[1 - middle:]
            print(f'Truncating {input_name} (#{len(input_name)}) to {prefix}_{suffix} (#{2 + len(prefix) + len(suffix)})')
            mid = '__'
            if conflict:
                mid = f'_{conflict_level}'
            input_name = f'{prefix}{mid}{suffix}'
        return input_name

    def get_name_max_length(self):
        if False:
            i = 10
            return i + 15
        if self.destination_type.value in DESTINATION_SIZE_LIMITS:
            destination_limit = DESTINATION_SIZE_LIMITS[self.destination_type.value]
            return destination_limit - TRUNCATE_DBT_RESERVED_SIZE - TRUNCATE_RESERVED_SIZE
        else:
            raise KeyError(f'Unknown destination type {self.destination_type}')

    def __normalize_non_column_identifier_name(self, input_name: str, in_jinja: bool=False, truncate: bool=True, conflict: bool=False, conflict_level: int=0) -> str:
        if False:
            i = 10
            return i + 15
        result = transform_standard_naming(input_name)
        result = self.__normalize_naming_conventions(result, is_column=False)
        if truncate:
            result = self.truncate_identifier_name(input_name=result, conflict=conflict, conflict_level=conflict_level)
        result = self.__normalize_identifier_case(result, is_quoted=False)
        if result[0].isdigit():
            if self.destination_type == DestinationType.MSSQL:
                result = '_' + result
            elif self.destination_type == DestinationType.ORACLE:
                result = 'ab_' + result
        return result

    def __normalize_identifier_name(self, column_name: str, in_jinja: bool=False, truncate: bool=True, conflict: bool=False, conflict_level: int=0) -> str:
        if False:
            while True:
                i = 10
        result = self.__normalize_naming_conventions(column_name, is_column=True)
        if truncate:
            result = self.truncate_identifier_name(input_name=result, conflict=conflict, conflict_level=conflict_level)
        if self.needs_quotes(result):
            if self.destination_type.value == DestinationType.CLICKHOUSE.value:
                result = result.replace('"', '_')
                result = result.replace('`', '_')
                result = result.replace("'", '_')
            elif self.destination_type.value != DestinationType.MYSQL.value and self.destination_type.value != DestinationType.TIDB.value and (self.destination_type.value != DestinationType.DUCKDB.value):
                result = result.replace('"', '""')
            else:
                result = result.replace('`', '_')
            result = result.replace("'", "\\'")
            result = self.__normalize_identifier_case(result, is_quoted=True)
            result = self.apply_quote(result)
            if not in_jinja:
                result = jinja_call(result)
            return result
        else:
            result = self.__normalize_identifier_case(result, is_quoted=False)
        if in_jinja:
            return f"'{result}'"
        return result

    def apply_quote(self, input: str, literal=True) -> str:
        if False:
            return 10
        if literal:
            input = f"'{input}'"
        if self.destination_type == DestinationType.ORACLE:
            return f'quote({input})'
        elif self.destination_type == DestinationType.CLICKHOUSE:
            return f'quote({input})'
        return f'adapter.quote({input})'

    def __normalize_naming_conventions(self, input_name: str, is_column: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        result = input_name
        if self.destination_type.value == DestinationType.ORACLE.value:
            return transform_standard_naming(result)
        elif self.destination_type.value == DestinationType.BIGQUERY.value:
            result = transform_standard_naming(result)
            doesnt_start_with_alphaunderscore = match('[^A-Za-z_]', result[0]) is not None
            if is_column and doesnt_start_with_alphaunderscore:
                result = f'_{result}'
        return result

    def __normalize_identifier_case(self, input_name: str, is_quoted: bool=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        result = input_name
        if self.destination_type.value == DestinationType.BIGQUERY.value:
            pass
        elif self.destination_type.value == DestinationType.REDSHIFT.value:
            result = input_name.lower()
        elif self.destination_type.value == DestinationType.POSTGRES.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
        elif self.destination_type.value == DestinationType.SNOWFLAKE.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.upper()
        elif self.destination_type.value == DestinationType.MYSQL.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
        elif self.destination_type.value == DestinationType.MSSQL.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
        elif self.destination_type.value == DestinationType.ORACLE.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
            else:
                result = input_name.upper()
        elif self.destination_type.value == DestinationType.CLICKHOUSE.value:
            pass
        elif self.destination_type.value == DestinationType.TIDB.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
        elif self.destination_type.value == DestinationType.DUCKDB.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
        else:
            raise KeyError(f'Unknown destination type {self.destination_type}')
        return result

    def normalize_column_identifier_case_for_lookup(self, input_name: str, is_quoted: bool=False) -> str:
        if False:
            i = 10
            return i + 15
        '\n        This function adds an additional normalization regarding the column name casing to determine if multiple columns\n        are in collisions. On certain destinations/settings, case sensitivity matters, in others it does not.\n        We separate this from standard identifier normalization "__normalize_identifier_case",\n        so the generated SQL queries are keeping the original casing from the catalog.\n        But we still need to determine if casing matters or not, thus by using this function.\n        '
        result = input_name
        if self.destination_type.value == DestinationType.BIGQUERY.value:
            result = input_name.lower()
        elif self.destination_type.value == DestinationType.REDSHIFT.value:
            result = input_name.lower()
        elif self.destination_type.value == DestinationType.POSTGRES.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
        elif self.destination_type.value == DestinationType.SNOWFLAKE.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.upper()
        elif self.destination_type.value == DestinationType.MYSQL.value:
            result = input_name.lower()
        elif self.destination_type.value == DestinationType.MSSQL.value:
            result = input_name.lower()
        elif self.destination_type.value == DestinationType.ORACLE.value:
            if not is_quoted and (not self.needs_quotes(input_name)):
                result = input_name.lower()
            else:
                result = input_name.upper()
        elif self.destination_type.value == DestinationType.CLICKHOUSE.value:
            pass
        elif self.destination_type.value == DestinationType.TIDB.value:
            result = input_name.lower()
        elif self.destination_type.value == DestinationType.DUCKDB.value:
            result = input_name.lower()
        else:
            raise KeyError(f'Unknown destination type {self.destination_type}')
        return result

def transform_standard_naming(input_name: str) -> str:
    if False:
        return 10
    result = input_name.strip()
    result = strip_accents(result)
    result = sub('\\s+', '_', result)
    result = sub('[^a-zA-Z0-9_]', '_', result)
    return result

def transform_json_naming(input_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    result = sub('[\'\\"`]', '_', input_name)
    return result

def strip_accents(input_name: str) -> str:
    if False:
        while True:
            i = 10
    return ''.join((c for c in ud.normalize('NFD', input_name) if ud.category(c) != 'Mn'))