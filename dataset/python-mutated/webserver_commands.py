from typing import Any, Dict
from freqtrade.enums import RunMode

def start_webserver(args: Dict[str, Any]) -> None:
    if False:
        return 10
    '\n    Main entry point for webserver mode\n    '
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.rpc.api_server import ApiServer
    config = setup_utils_configuration(args, RunMode.WEBSERVER)
    ApiServer(config, standalone=True)