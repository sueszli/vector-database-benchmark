import json
import os
from typing import Optional
from esphome.config import load_config, _format_vol_invalid, Config
from esphome.core import CORE, DocumentRange
import esphome.config_validation as cv

def _get_invalid_range(res: Config, invalid: cv.Invalid) -> Optional[DocumentRange]:
    if False:
        for i in range(10):
            print('nop')
    return res.get_deepest_document_range_for_path(invalid.path, invalid.error_message == 'extra keys not allowed')

def _dump_range(range: Optional[DocumentRange]) -> Optional[dict]:
    if False:
        i = 10
        return i + 15
    if range is None:
        return None
    return {'document': range.start_mark.document, 'start_line': range.start_mark.line, 'start_col': range.start_mark.column, 'end_line': range.end_mark.line, 'end_col': range.end_mark.column}

class VSCodeResult:

    def __init__(self):
        if False:
            return 10
        self.yaml_errors = []
        self.validation_errors = []

    def dump(self):
        if False:
            while True:
                i = 10
        return json.dumps({'type': 'result', 'yaml_errors': self.yaml_errors, 'validation_errors': self.validation_errors})

    def add_yaml_error(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.yaml_errors.append({'message': message})

    def add_validation_error(self, range_, message):
        if False:
            for i in range(10):
                print('nop')
        self.validation_errors.append({'range': _dump_range(range_), 'message': message})

def read_config(args):
    if False:
        print('Hello World!')
    while True:
        CORE.reset()
        data = json.loads(input())
        assert data['type'] == 'validate'
        CORE.vscode = True
        CORE.ace = args.ace
        f = data['file']
        if CORE.ace:
            CORE.config_path = os.path.join(args.configuration, f)
        else:
            CORE.config_path = data['file']
        vs = VSCodeResult()
        try:
            res = load_config(dict(args.substitution) if args.substitution else {})
        except Exception as err:
            vs.add_yaml_error(str(err))
        else:
            for err in res.errors:
                try:
                    range_ = _get_invalid_range(res, err)
                    vs.add_validation_error(range_, _format_vol_invalid(err, res))
                except Exception:
                    continue
        print(vs.dump())