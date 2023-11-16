from __future__ import annotations
from pathlib import Path

def fix_provider_references(app, exception):
    if False:
        while True:
            i = 10
    'Sphinx "build-finished" event handler.'
    from sphinx.builders import html as builders
    if exception or not isinstance(app.builder, builders.StandaloneHTMLBuilder):
        return
    for path in Path(app.outdir).rglob('*.html'):
        if path.exists():
            lines = path.read_text().splitlines(True)
            with path.open('w') as output_file:
                for line in lines:
                    output_file.write(line.replace('|version|', app.config.version))

def setup(app):
    if False:
        return 10
    'Setup plugin'
    app.connect('build-finished', fix_provider_references)
    return {'parallel_write_safe': True, 'parallel_read_safe': True}