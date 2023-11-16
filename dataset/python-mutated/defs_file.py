from dagster import Definitions, asset

def _make_defs():
    if False:
        return 10

    @asset
    def an_asset():
        if False:
            i = 10
            return i + 15
        pass
    return Definitions(assets=[an_asset])
defs = _make_defs()