from __future__ import annotations
from dataclasses import dataclass, field
import huggingface_hub
import semantic_version
import semantic_version as semver

@dataclass
class ThemeAsset:
    filename: str
    version: semver.Version = field(init=False)

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.version = semver.Version(self.filename.split('@')[1].replace('.json', ''))

def get_theme_assets(space_info: huggingface_hub.hf_api.SpaceInfo) -> list[ThemeAsset]:
    if False:
        for i in range(10):
            print('nop')
    return [ThemeAsset(filename.rfilename) for filename in space_info.siblings if filename.rfilename.startswith('themes/')]

def get_matching_version(assets: list[ThemeAsset], expression: str | None) -> ThemeAsset | None:
    if False:
        i = 10
        return i + 15
    expression = expression or '*'
    matching_version = semantic_version.SimpleSpec(expression).select([a.version for a in assets])
    return next((a for a in assets if a.version == matching_version), None)