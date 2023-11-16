import sys
from PyInstaller.utils.hooks import collect_submodules

def pycryptodome_module():
    if False:
        while True:
            i = 10
    try:
        import Cryptodome
    except ImportError:
        try:
            import Crypto
            print('WARNING: Using Crypto since Cryptodome is not available. Install with: pip install pycryptodomex', file=sys.stderr)
            return 'Crypto'
        except ImportError:
            pass
    return 'Cryptodome'

def get_hidden_imports():
    if False:
        return 10
    yield from ('yt_dlp.compat._legacy', 'yt_dlp.compat._deprecated')
    yield from ('yt_dlp.utils._legacy', 'yt_dlp.utils._deprecated')
    yield pycryptodome_module()
    for module in ('websockets', 'requests', 'urllib3'):
        yield from collect_submodules(module)
    yield from ('mutagen', 'brotli', 'certifi', 'secretstorage')
hiddenimports = list(get_hidden_imports())
print(f'Adding imports: {hiddenimports}')
excludedimports = ['youtube_dl', 'youtube_dlc', 'test', 'ytdlp_plugins', 'devscripts']