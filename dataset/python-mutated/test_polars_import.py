from __future__ import annotations
import compileall
import subprocess
import sys
from pathlib import Path
import pytest
import polars as pl
from polars import selectors as cs
MAX_ALLOWED_IMPORT_TIME = 200000

def _import_time_from_frame(tm: pl.DataFrame) -> int:
    if False:
        i = 10
        return i + 15
    return int(tm.filter(pl.col('import').str.strip_chars() == 'polars').select('cumulative_time').item())

def _import_timings() -> bytes:
    if False:
        print('Hello World!')
    cmd = f'{sys.executable} -X importtime -c "import polars"'
    output = subprocess.run(cmd, shell=True, capture_output=True).stderr.replace(b'import time:', b'').strip()
    return output

def _import_timings_as_frame(n_tries: int) -> tuple[pl.DataFrame, int]:
    if False:
        return 10
    import_timings = []
    for _ in range(n_tries):
        df_import = pl.read_csv(source=_import_timings(), separator='|', has_header=True, new_columns=['own_time', 'cumulative_time', 'import']).with_columns(cs.ends_with('_time').str.strip_chars().cast(pl.UInt32)).select('import', 'own_time', 'cumulative_time').reverse()
        polars_import_time = _import_time_from_frame(df_import)
        if polars_import_time < MAX_ALLOWED_IMPORT_TIME:
            return (df_import, polars_import_time)
        import_timings.append(df_import)
    df_fastest_import = sorted(import_timings, key=_import_time_from_frame)[0]
    return (df_fastest_import, _import_time_from_frame(df_fastest_import))

@pytest.mark.skipif(sys.platform == 'win32', reason='Unreliable on Windows')
@pytest.mark.slow()
def test_polars_import() -> None:
    if False:
        i = 10
        return i + 15
    polars_path = Path(pl.__file__).parent
    compileall.compile_dir(polars_path, quiet=1)
    (df_import, polars_import_time) = _import_timings_as_frame(n_tries=5)
    with pl.Config(tbl_rows=250, fmt_str_lengths=100, tbl_hide_dataframe_shape=True):
        lazy_modules = [dep for dep in pl.dependencies.__all__ if not dep.startswith('_')]
        for mod in lazy_modules:
            not_imported = not df_import['import'].str.starts_with(mod).any()
            if_err = f'lazy-loading regression: found {mod!r} at import time'
            assert not_imported, f'{if_err}\n{df_import}'
        if polars_import_time > MAX_ALLOWED_IMPORT_TIME:
            import_time_ms = polars_import_time // 1000
            raise AssertionError(f'Possible import speed regression; took {import_time_ms}ms\n{df_import}')