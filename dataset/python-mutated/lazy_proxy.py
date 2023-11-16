from typing import Any
try:
    from babel.support import LazyProxy
except ImportError:

    class LazyProxy:

        def __init__(self, func: Any, *args: Any, **kwargs: Any) -> None:
            if False:
                print('Hello World!')
            raise RuntimeError('LazyProxy can be used only when Babel installed\nJust install Babel (`pip install Babel`) or aiogram with i18n support (`pip install aiogram[i18n]`)')