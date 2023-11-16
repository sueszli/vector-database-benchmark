"""LargeBlobSenderConsts module"""
USE_DISK = 1
ChunkSize = 100
FilePattern = 'largeBlob.%s'

def getLargeBlobPath():
    if False:
        print('Hello World!')
    from panda3d.core import ConfigVariableString, ConfigFlags
    return ConfigVariableString('large-blob-path', 'i:\\toontown_in_game_editor_temp', 'DConfig', ConfigFlags.F_dconfig).value