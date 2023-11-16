from __future__ import annotations
import json
import os
from typing import Any
from mypy.plugin import Plugin, ReportConfigContext

class ConfigDataPlugin(Plugin):

    def report_config_data(self, ctx: ReportConfigContext) -> Any:
        if False:
            print('Hello World!')
        path = os.path.join('tmp/test.json')
        with open(path) as f:
            data = json.load(f)
        return data.get(ctx.id)

def plugin(version: str) -> type[ConfigDataPlugin]:
    if False:
        for i in range(10):
            print('nop')
    return ConfigDataPlugin