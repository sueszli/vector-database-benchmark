import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Union
import rapidjson

def get_strategy_run_id(strategy) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate unique identification hash for a backtest run. Identical config and strategy file will\n    always return an identical hash.\n    :param strategy: strategy object.\n    :return: hex string id.\n    '
    digest = hashlib.sha1()
    config = deepcopy(strategy.config)
    not_important_keys = ('strategy_list', 'original_config', 'telegram', 'api_server')
    for k in not_important_keys:
        if k in config:
            del config[k]
    digest.update(rapidjson.dumps(config, default=str, number_mode=rapidjson.NM_NAN).encode('utf-8'))
    digest.update(rapidjson.dumps(strategy._ft_params_from_file, default=str, number_mode=rapidjson.NM_NAN).encode('utf-8'))
    with Path(strategy.__file__).open('rb') as fp:
        digest.update(fp.read())
    return digest.hexdigest().lower()

def get_backtest_metadata_filename(filename: Union[Path, str]) -> Path:
    if False:
        return 10
    'Return metadata filename for specified backtest results file.'
    filename = Path(filename)
    return filename.parent / Path(f'{filename.stem}.meta{filename.suffix}')