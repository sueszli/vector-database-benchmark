from __future__ import annotations
from typing import TYPE_CHECKING
from airflow_breeze.utils.cache import touch_cache_file
from airflow_breeze.utils.md5_build_check import calculate_md5_checksum_for_files
from airflow_breeze.utils.path_utils import BUILD_CACHE_DIR
if TYPE_CHECKING:
    from airflow_breeze.params.build_ci_params import BuildCiParams

def mark_image_as_refreshed(ci_image_params: BuildCiParams):
    if False:
        for i in range(10):
            print('nop')
    ci_image_cache_dir = BUILD_CACHE_DIR / ci_image_params.airflow_branch
    ci_image_cache_dir.mkdir(parents=True, exist_ok=True)
    touch_cache_file(f'built_{ci_image_params.python}', root_dir=ci_image_cache_dir)
    calculate_md5_checksum_for_files(ci_image_params.md5sum_cache_dir, update=True)