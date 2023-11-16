from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Union, cast
import dagster._check as check
import orjson
DbtManifestParam = Union[Mapping[str, Any], str, Path]

@lru_cache(maxsize=None)
def read_manifest_path(manifest_path: Path) -> Mapping[str, Any]:
    if False:
        i = 10
        return i + 15
    "Reads a dbt manifest path and returns the parsed JSON as a dict.\n\n    This function is cached to ensure that we don't read the same path multiple times, which\n    creates multiple copies of the parsed manifest in memory.\n\n    If we fix the fact that the manifest is held in memory instead of garbage collected, we\n    can delete this cache.\n    "
    return cast(Mapping[str, Any], orjson.loads(manifest_path.read_bytes()))

def validate_manifest(manifest: DbtManifestParam) -> Mapping[str, Any]:
    if False:
        i = 10
        return i + 15
    check.inst_param(manifest, 'manifest', (Path, str, dict))
    if isinstance(manifest, str):
        manifest = Path(manifest)
    if isinstance(manifest, Path):
        manifest = read_manifest_path(manifest.resolve())
    return manifest