"""To publish HTML docs at GitHub Pages, create .nojekyll file."""
from __future__ import annotations
import contextlib
import os
import urllib.parse
from typing import TYPE_CHECKING, Any
import sphinx
if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.environment import BuildEnvironment

def _get_domain_from_url(url: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get the domain from a URL.'
    return url and urllib.parse.urlparse(url).hostname or ''

def create_nojekyll_and_cname(app: Sphinx, env: BuildEnvironment) -> None:
    if False:
        while True:
            i = 10
    "Manage the ``.nojekyll`` and ``CNAME`` files for GitHub Pages.\n\n    For HTML-format builders (e.g. 'html', 'dirhtml') we unconditionally create\n    the ``.nojekyll`` file to signal that GitHub Pages should not run Jekyll\n    processing.\n\n    If the :confval:`html_baseurl` option is set, we also create a CNAME file\n    with the domain from ``html_baseurl``, so long as it is not a ``github.io``\n    domain.\n\n    If this extension is loaded and the domain in ``html_baseurl`` no longer\n    requires a CNAME file, we remove any existing ``CNAME`` files from the\n    output directory.\n    "
    if app.builder.format != 'html':
        return
    app.builder.outdir.joinpath('.nojekyll').touch()
    cname_path = os.path.join(app.builder.outdir, 'CNAME')
    domain = _get_domain_from_url(app.config.html_baseurl)
    if domain and (not domain.endswith('.github.io')):
        with open(cname_path, 'w', encoding='utf-8') as f:
            f.write(domain)
    else:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(cname_path)

def setup(app: Sphinx) -> dict[str, Any]:
    if False:
        return 10
    app.connect('env-updated', create_nojekyll_and_cname)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}