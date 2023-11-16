"""
Stores information about base game editions and expansions.
"""
from __future__ import annotations
from dataclasses import dataclass
import enum
from ..read.media_types import MediaType
from .game_file_version import GameFileVersion

@enum.unique
class Support(enum.Enum):
    """
    Support state of a game version
    """
    NOPE = 'not supported'
    YES = 'supported'
    BREAKS = 'presence breaks conversion'

class GameBase:
    """
    Common base class for GameEdition and GameExpansion
    """

    def __init__(self, game_id: str, support: Support, game_version_info: list[tuple[list[str], dict[str, str]]], media_paths: list[tuple[str, list[str]]], modpacks: list[str], **flags):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param game_id: Unique id for the given game.\n        :type game_id: str\n        :param support: Whether the converter can read/convert\n                               the game to openage formats.\n        :type support: str\n        :param modpacks: List of modpacks.\n        :type modpacks: list\n        :param game_version_info: Versioning information about the game.\n        :type game_version_info: list[tuple]\n        :param media_paths: Media types and their paths\n        :type media_paths: list[tuple]\n        :param flags: Anything else specific to this version which is useful\n                      for the converter.\n        '
        self.game_id = game_id
        self.flags = flags
        self.target_modpacks = modpacks
        self.support = Support[support.upper()]
        self.game_file_versions = []
        self.media_paths = {}
        self.media_cache = None
        if 'media_cache' in flags:
            self.media_cache = flags['media_cache']
        for (path, hash_map) in game_version_info:
            self.add_game_file_versions(path, hash_map)
        for (media_type, paths) in media_paths:
            self.add_media_paths(media_type, paths)

    def add_game_file_versions(self, filepaths: list[str], hashes: dict[str, str]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add a GameFileVersion object for files which are unique\n        to this version of the game.\n\n        :param filepaths: Paths to the specified file. Only one of the paths\n                          needs to exist. The other paths are interpreted as\n                          alternatives, e.g. if the game is released on different\n                          platforms with different names for the same file.\n        :type filepaths: list\n        :param hashes: Hash value mapped to a file version.\n        :type hashes: dict\n        '
        self.game_file_versions.append(GameFileVersion(filepaths, hashes))

    def add_media_paths(self, media_type: str, paths: list[str]) -> None:
        if False:
            while True:
                i = 10
        '\n        Add a media type with the associated files.\n\n        :param media_type: The type of media file.\n        :type media_type: MediaType\n        :param paths: Paths to those media files.\n        :type paths: list\n        '
        self.media_paths[MediaType[media_type.upper()]] = paths

    def __eq__(self, other: GameBase) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compare equality by comparing IDs.\n        '
        return self.game_id == other.game_id

    def __hash__(self) -> int:
        if False:
            return 10
        '\n        Reimplement hash to only consider the game ID.\n        '
        return hash(self.game_id)

class GameExpansion(GameBase):
    """
    An optional expansion to a GameEdition.
    """

    def __init__(self, name: str, game_id: str, support: Support, game_version_info: list[tuple[list[str], dict[str, str]]], media_paths: list[tuple[str, list[str]]], modpacks: list[str], **flags):
        if False:
            i = 10
            return i + 15
        '\n        Create a new GameExpansion instance.\n\n        :param name: Name of the game.\n        :type name: str\n        '
        super().__init__(game_id, support, game_version_info, media_paths, modpacks, **flags)
        self.expansion_name = name

    def __str__(self):
        if False:
            print('Hello World!')
        return self.expansion_name

class GameEdition(GameBase):
    """
    Standalone/base version of a game. Multiple standalone versions
    may exist, e.g. AoC, HD, DE2 for AoE2.

    Note that we treat AoE1+Rise of Rome and AoE2+The Conquerors as
    standalone versions. AoE1 without Rise of Rome or AoK without
    The Conquerors are considered "downgrade" expansions.
    """

    def __init__(self, name: str, game_id: str, support: Support, game_version_info: list[tuple[list[str], dict[str, str]]], media_paths: list[tuple[str, list[str]]], install_paths: dict[str, list[str]], modpacks: list[str], expansions: list[str], **flags):
        if False:
            return 10
        '\n        Create a new GameEdition instance.\n\n        :param name: Name of the game.\n        :type name: str\n        :param expansions: A list of expansions.\n        :type expansion: list\n        '
        super().__init__(game_id, support, game_version_info, media_paths, modpacks, **flags)
        self.install_paths = install_paths
        self.edition_name = name
        self.expansions = tuple(expansions)

    def __str__(self):
        if False:
            print('Hello World!')
        return self.edition_name

@dataclass(frozen=True)
class GameVersion:
    """
    Combination of edition and expansions that defines the exact version
    of a detected game in a folder.
    """
    edition: GameEdition
    expansions: tuple[GameExpansion, ...] = tuple()