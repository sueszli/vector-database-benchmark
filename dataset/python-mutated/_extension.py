from typing import Any

def load_ipython_extension(ip: Any) -> None:
    if False:
        print('Hello World!')
    from rich.pretty import install
    from rich.traceback import install as tr_install
    install()
    tr_install()