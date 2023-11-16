from pathlib import Path
from textwrap import dedent
from pyodide_lock.spec import PackageSpec

def create_pypa_index(packages: dict[str, PackageSpec], target_dir: Path, dist_url: str) -> None:
    if False:
        i = 10
        return i + 15
    'Create a pip-compatible Python package (pypa) index to be used with a Pyodide virtual\n    environment.\n\n    To use, pass as an `--index-url` or `--extra-index-url` parameter to pip.\n    The argument should be a `file:` url pointing to the `pypa_index` folder (or\n    if you serve `pypa_index` it can be a normal url). It is also used\n    automatically by Pyodide virtual environments created from a release version\n    of Pyodide.\n\n    Parameters\n    ----------\n    packages:\n        A dictionary of packages that we want to index. This should be the\n        "packages" field from pyodide-lock.json.\n\n    target_dir:\n        Where to put the  index. It will be placed in a subfolder of\n        target_dir called `pypa_index`. `target_dir` should exist but\n        `target_dir/pypa_index` should not exist.\n\n    dist_url:\n        The CDN url to download packages from. This will be hard coded into the\n        generated index. If you wish to install from local files, then prefix\n        with `file:` e.g., `f"file:{pyodide_root}/dist"`.\n    '
    packages = {pkgname: pkginfo for (pkgname, pkginfo) in packages.items() if pkginfo.file_name.endswith('.whl')}
    if not target_dir.exists():
        raise RuntimeError(f'target_dir={target_dir} does not exist')
    index_dir = target_dir / 'pypa_index'
    if index_dir.exists():
        raise RuntimeError(f'{index_dir} already exists')
    index_dir.mkdir()
    packages_str = '\n'.join((f'<a href="{x}/">{x}</a>' for x in packages.keys()))
    index_html = dedent(f'\n        <!DOCTYPE html>\n        <html>\n        <head>\n        <meta name="pypi:repository-version" content="1.0">\n        <title>Simple index</title>\n        </head>\n        <body>\n        {packages_str}\n        </body>\n        </html>\n        ').strip()
    (index_dir / 'index.html').write_text(index_html)
    files_template = dedent('\n        <!DOCTYPE html>\n        <html>\n        <head>\n        <meta name="pypi:repository-version" content="1.0">\n        <title>Links for {pkgname}</title>\n        </head>\n        <body>\n        <h1>Links for {pkgname}</h1>\n        {links}\n        </body>\n        </html>\n        ').strip()
    for (pkgname, pkginfo) in packages.items():
        pkgdir = index_dir / pkgname
        filename = pkginfo.file_name
        shasum = pkginfo.sha256
        href = f'{dist_url}{filename}#sha256={shasum}'
        links_str = f'<a href="{href}">{pkgname}</a>\n'
        files_html = files_template.format(pkgname=pkgname, links=links_str)
        pkgdir.mkdir()
        (pkgdir / 'index.html').write_text(files_html)