import json
import re
from typing import Dict, List
from mage_ai.data_preparation.models.constants import DATAFRAME_ANALYSIS_MAX_COLUMNS, DATAFRAME_SAMPLE_COUNT_PREVIEW, BlockType
from mage_ai.server.kernels import KernelName
from mage_ai.shared.code import is_pyspark_code
REGEX_PATTERN = '^[ ]{2,}[\\w]+'

def remove_comments(code_lines: List[str]) -> List[str]:
    if False:
        while True:
            i = 10
    return list(filter(lambda x: not re.search('^\\#', str(x).strip()), code_lines))

def remove_empty_last_lines(code_lines: List[str]) -> List[str]:
    if False:
        i = 10
        return i + 15
    idx = len(code_lines) - 1
    last_line = code_lines[idx]
    while idx >= 0 and len(str(last_line).strip()) == 0:
        idx -= 1
        last_line = code_lines[idx]
    return code_lines[:idx + 1]

def find_index_of_last_expression_lines(code_lines: List[str]) -> int:
    if False:
        return 10
    starting_index = len(code_lines) - 1
    brackets_close = code_lines[starting_index].count('}')
    brackets_open = code_lines[starting_index].count('{')
    paranthesis_close = code_lines[starting_index].count(')')
    paranthesis_open = code_lines[starting_index].count('(')
    square_brackets_close = code_lines[starting_index].count(']')
    square_brackets_open = code_lines[starting_index].count('[')
    while starting_index >= 0 and (brackets_close > brackets_open or paranthesis_close > paranthesis_open or square_brackets_close > square_brackets_open):
        starting_index -= 1
        brackets_close += code_lines[starting_index].count('}')
        brackets_open += code_lines[starting_index].count('{')
        paranthesis_close += code_lines[starting_index].count(')')
        paranthesis_open += code_lines[starting_index].count('(')
        square_brackets_close += code_lines[starting_index].count(']')
        square_brackets_open += code_lines[starting_index].count('[')
    return starting_index

def get_content_inside_triple_quotes(parts):
    if False:
        i = 10
        return i + 15
    parts_length = len(parts) - 1
    start_index = None
    for i in range(parts_length):
        idx = parts_length - (i + 1)
        part = parts[idx]
        if re.search('"""', part):
            start_index = idx
        if start_index is not None:
            break
    if start_index is not None:
        first_line = parts[start_index]
        variable = None
        if re.search('[\\w]+[ ]*=[ ]*[f]*"""', first_line):
            variable = first_line.split('=')[0].strip()
        return ('\n'.join(parts[start_index + 1:-1]).replace('"', '\\"'), variable)
    return (None, None)

def add_internal_output_info(code: str) -> str:
    if False:
        i = 10
        return i + 15
    if code.startswith('%%sql') or code.startswith('%%bash') or len(code) == 0:
        return code
    code_lines = remove_comments(code.split('\n'))
    code_lines = remove_empty_last_lines(code_lines)
    starting_index = find_index_of_last_expression_lines(code_lines)
    if starting_index < len(code_lines) - 1:
        last_line = ' '.join(code_lines[starting_index:])
        code_lines = code_lines[:starting_index] + [last_line]
    else:
        last_line = code_lines[len(code_lines) - 1]
    matches = re.search('^[ ]*([^{^(^\\[^=^ ]+)[ ]*=[ ]*', last_line)
    if matches:
        last_line = matches.group(1)
    last_line = last_line.strip()
    is_print_statement = False
    if re.findall('print\\(', last_line):
        is_print_statement = True
    last_line_in_block = False
    if len(code_lines) >= 2:
        if re.search(REGEX_PATTERN, code_lines[-2]) or re.search(REGEX_PATTERN, code_lines[-1]):
            last_line_in_block = True
    elif re.search('^import[ ]{1,}|^from[ ]{1,}', code_lines[-1].strip()):
        last_line_in_block = True
    if re.search('"""$', last_line):
        (triple_quotes_content, variable) = get_content_inside_triple_quotes(code_lines)
        if variable:
            return f'{code}\nprint({variable})'
        elif triple_quotes_content:
            return f'{code}\nprint("""\n{triple_quotes_content}\n""")'
    if not last_line or last_line_in_block or re.match('^from|^import|^\\%\\%', last_line.strip()):
        return code
    else:
        if matches:
            end_index = len(code_lines)
        else:
            end_index = -1
        code_without_last_line = '\n'.join(code_lines[:end_index])
        internal_output = f"\n# Post processing code below (source: output_display.py)\n\n\ndef __custom_output():\n    from datetime import datetime\n    from mage_ai.shared.parsers import encode_complex, sample_output\n    import json\n    import pandas as pd\n    import polars as pl\n    import simplejson\n    import warnings\n\n    if pd.__version__ < '1.5.0':\n        from pandas.core.common import SettingWithCopyWarning\n    else:\n        from pandas.errors import SettingWithCopyWarning\n\n    warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)\n\n    _internal_output_return = {last_line}\n\n    if isinstance(_internal_output_return, pd.DataFrame) and (\n        type(_internal_output_return).__module__ != 'geopandas.geodataframe'\n    ):\n        _sample = _internal_output_return.iloc[:{DATAFRAME_SAMPLE_COUNT_PREVIEW}]\n        _columns = _sample.columns.tolist()[:{DATAFRAME_ANALYSIS_MAX_COLUMNS}]\n        _rows = json.loads(_sample[_columns].to_json(default_handler=str, orient='split'))['data']\n        _shape = _internal_output_return.shape\n        _index = _sample.index.tolist()\n\n        _json_string = simplejson.dumps(\n            dict(\n                data=dict(\n                    columns=_columns,\n                    index=_index,\n                    rows=_rows,\n                    shape=_shape,\n                ),\n                type='table',\n            ),\n            default=encode_complex,\n            ignore_nan=True,\n        )\n        return print(f'[__internal_output__]{{_json_string}}')\n    elif isinstance(_internal_output_return, pl.DataFrame):\n        return print(_internal_output_return)\n    elif type(_internal_output_return).__module__ == 'pyspark.sql.dataframe':\n        _sample = _internal_output_return.limit({DATAFRAME_SAMPLE_COUNT_PREVIEW}).toPandas()\n        _columns = _sample.columns.tolist()[:40]\n        _rows = _sample.to_numpy().tolist()\n        _shape = [_internal_output_return.count(), len(_sample.columns.tolist())]\n        _index = _sample.index.tolist()\n\n        _json_string = simplejson.dumps(\n            dict(\n                data=dict(\n                    columns=_columns,\n                    index=_index,\n                    rows=_rows,\n                    shape=_shape,\n                ),\n                type='table',\n            ),\n            default=encode_complex,\n            ignore_nan=True,\n        )\n        return print(f'[__internal_output__]{{_json_string}}')\n    elif not {is_print_statement}:\n        output, sampled = sample_output(encode_complex(_internal_output_return))\n        if sampled:\n            print('Sampled output is provided here for preview.')\n        return output\n\n    return\n\n__custom_output()\n"
        custom_code = f'{code_without_last_line}\n{internal_output}\n'
        return custom_code

def add_execution_code(pipeline_uuid: str, block_uuid: str, code: str, global_vars, block_type: BlockType=None, extension_uuid: str=None, kernel_name: str=None, output_messages_to_logs: bool=False, pipeline_config: Dict=None, repo_config: Dict=None, run_incomplete_upstream: bool=False, run_settings: Dict=None, run_tests: bool=False, run_upstream: bool=False, update_status: bool=True, upstream_blocks: List[str]=None, variables: Dict=None, widget: bool=False) -> str:
    if False:
        print('Hello World!')
    escaped_code = code.replace("'''", '"""')
    if extension_uuid:
        extension_uuid = f"'{extension_uuid}'"
    if upstream_blocks:
        upstream_blocks = ', '.join([f"'{u}'" for u in upstream_blocks])
        upstream_blocks = f'[{upstream_blocks}]'
    run_settings_json = json.dumps(run_settings or {})
    magic_header = ''
    spark_session_init = ''
    if kernel_name == KernelName.PYSPARK:
        if block_type == BlockType.CHART or (block_type == BlockType.SENSOR and (not is_pyspark_code(code))):
            magic_header = '%%local'
            run_incomplete_upstream = False
            run_upstream = False
        elif block_type in [BlockType.DATA_LOADER, BlockType.TRANSFORMER]:
            magic_header = '%%spark -o df --maxrows 10000'
    elif pipeline_config['type'] == 'databricks':
        spark_session_init = '\nfrom pyspark.sql import SparkSession\nspark = SparkSession.builder.getOrCreate()\n'
    return f"{magic_header}\nfrom mage_ai.data_preparation.models.pipeline import Pipeline\nfrom mage_ai.settings.repo import get_repo_path\nfrom mage_ai.orchestration.db import db_connection\nfrom mage_ai.shared.array import find\nfrom mage_ai.shared.hash import merge_dict\nimport datetime\nimport json\nimport logging\nimport pandas as pd\n\n\ndb_connection.start_session()\n{spark_session_init}\n\nif 'context' not in globals():\n    context = dict()\n\ndef execute_custom_code():\n    block_uuid='{block_uuid}'\n    run_incomplete_upstream={str(run_incomplete_upstream)}\n    run_upstream={str(run_upstream)}\n    pipeline = Pipeline(\n        uuid='{pipeline_uuid}',\n        config={pipeline_config},\n        repo_config={repo_config},\n    )\n    block = pipeline.get_block(block_uuid, extension_uuid={extension_uuid}, widget={widget})\n\n    upstream_blocks = {upstream_blocks}\n    if upstream_blocks and len(upstream_blocks) >= 1:\n        blocks = pipeline.get_blocks({upstream_blocks})\n        block.upstream_blocks = blocks\n\n    code = r'''\n{escaped_code}\n    '''\n\n    global_vars = merge_dict({global_vars} or dict(), pipeline.variables or dict())\n\n    if {variables}:\n        global_vars = merge_dict(global_vars, {variables})\n\n    if pipeline.run_pipeline_in_one_process:\n        # Use shared context for blocks\n        global_vars['context'] = context\n\n    try:\n        global_vars['spark'] = spark\n    except Exception:\n        pass\n\n    if run_incomplete_upstream or run_upstream:\n        block.run_upstream_blocks(\n            from_notebook=True,\n            global_vars=global_vars,\n            incomplete_only=run_incomplete_upstream,\n        )\n\n    logger = logging.getLogger('{block_uuid}_test')\n    logger.setLevel('INFO')\n    if 'logger' not in global_vars:\n        global_vars['logger'] = logger\n    block_output = block.execute_with_callback(\n        custom_code=code,\n        from_notebook=True,\n        global_vars=global_vars,\n        logger=logger,\n        output_messages_to_logs={output_messages_to_logs},\n        run_settings=json.loads('{run_settings_json}'),\n        update_status={update_status},\n    )\n    if {run_tests}:\n        block.run_tests(\n            custom_code=code,\n            from_notebook=True,\n            logger=logger,\n            global_vars=global_vars,\n            update_tests=False,\n        )\n    output = block_output['output'] or []\n\n    if {widget}:\n        return output\n    else:\n        return find(lambda val: val is not None, output)\n\ndf = execute_custom_code()\n    "

def get_block_output_process_code(pipeline_uuid: str, block_uuid: str, block_type: BlockType=None, kernel_name: str=None):
    if False:
        return 10
    if kernel_name != KernelName.PYSPARK or block_type not in [BlockType.DATA_LOADER, BlockType.TRANSFORMER]:
        return None
    return f"%%local\nfrom mage_ai.data_preparation.models.constants import BlockStatus\nfrom mage_ai.data_preparation.models.pipeline import Pipeline\n\nimport pandas\n\nblock_uuid='{block_uuid}'\npipeline = Pipeline(\n    uuid='{pipeline_uuid}',\n)\nblock = pipeline.get_block(block_uuid)\nvariable_mapping = dict(df=df)\nblock.store_variables(variable_mapping)\nblock.analyze_outputs(variable_mapping)\nblock.update_status(BlockStatus.EXECUTED)\n    "

def get_pipeline_execution_code(pipeline_uuid: str, global_vars: Dict=None, kernel_name: str=None, pipeline_config: Dict=None, repo_config: Dict=None, update_status: bool=True) -> str:
    if False:
        return 10
    spark_session_init = ''
    if pipeline_config['type'] == 'databricks':
        spark_session_init = "\nfrom pyspark.sql import SparkSession\nimport os\nspark = SparkSession.builder.master(os.getenv('SPARK_MASTER_HOST', 'local')).getOrCreate()\n"
    return f"\nfrom mage_ai.data_preparation.models.pipeline import Pipeline\nimport asyncio\n\n{spark_session_init}\n\ndef execute_pipeline():\n    pipeline = Pipeline(\n        uuid='{pipeline_uuid}',\n        config={pipeline_config},\n        repo_config={repo_config},\n    )\n\n    global_vars = {global_vars} or dict()\n\n    try:\n        global_vars['spark'] = spark\n    except Exception:\n        pass\n\n    asyncio.run(pipeline.execute(\n        analyze_outputs=False,\n        global_vars=global_vars,\n        update_status={update_status},\n    ))\nexecute_pipeline()\n    "