import os
from typing import Optional
from electrum.i18n import languages
jLocale = None
if 'ANDROID_DATA' in os.environ:
    from jnius import autoclass, cast
    jLocale = autoclass('java.util.Locale')

def get_default_language(*, gui_name: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    if gui_name == 'qt':
        from PyQt5.QtCore import QLocale
        name = QLocale.system().name()
        return name if name in languages else 'en_UK'
    elif gui_name == 'qml':
        from PyQt6.QtCore import QLocale
        try:
            name = str(jLocale.getDefault().toString())
        except Exception:
            name = QLocale.system().name()
        return name if name in languages else 'en_GB'
    return ''