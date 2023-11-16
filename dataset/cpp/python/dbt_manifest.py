from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Union, cast

import dagster._check as check
import orjson

DbtManifestParam = Union[Mapping[str, Any], str, Path]


@lru_cache(maxsize=None)
def read_manifest_path(manifest_path: Path) -> Mapping[str, Any]:
    """Reads a dbt manifest path and returns the parsed JSON as a dict.

    This function is cached to ensure that we don't read the same path multiple times, which
    creates multiple copies of the parsed manifest in memory.

    If we fix the fact that the manifest is held in memory instead of garbage collected, we
    can delete this cache.
    """
    return cast(Mapping[str, Any], orjson.loads(manifest_path.read_bytes()))


def validate_manifest(manifest: DbtManifestParam) -> Mapping[str, Any]:
    check.inst_param(manifest, "manifest", (Path, str, dict))

    if isinstance(manifest, str):
        manifest = Path(manifest)

    if isinstance(manifest, Path):
        # Resolve the path to ensure a consistent key for the cache
        manifest = read_manifest_path(manifest.resolve())

    return manifest
