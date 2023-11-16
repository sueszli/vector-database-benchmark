"""
Associate files or filepaths with hash values to determine the exact version of a game.
"""

class GameFileVersion:
    """
    Associates a file hash with a specific version number.
    This can be used to pinpoint the exact version of a game.
    """
    hash_algo = 'SHA3-256'

    def __init__(self, filepaths: list[str], hashes: dict[str, str]):
        if False:
            return 10
        '\n        Create a new file hash to version association.\n\n        :param filepaths: Paths to the specified file. Only one of the paths\n                          needs to exist. The other paths are interpreted as\n                          alternatives, e.g. if the game is released on different\n                          platforms with different names for the same file.\n        :type filepaths: list\n        :param hashes: Maps hashes to a version number string.\n        :type hashes: dict\n        '
        self.paths = filepaths
        if len(self.paths) < 1:
            raise ValueError(f'{self}: List of paths cannot be empty.')
        self.hashes = hashes

    def get_paths(self) -> list[str]:
        if False:
            print('Hello World!')
        '\n        Return all known paths to the file.\n        '
        return self.paths

    def get_hashes(self) -> dict[str, str]:
        if False:
            return 10
        '\n        Return the hash-version association for the file paths.\n        '
        return self.hashes