from dagster import Definitions

def _make_defs():
    if False:
        i = 10
        return i + 15
    return Definitions()
defs = _make_defs()
double_defs = _make_defs()