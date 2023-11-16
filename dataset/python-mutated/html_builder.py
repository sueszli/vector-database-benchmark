from __future__ import annotations
import json
import os
import textwrap
from pathlib import Path
from typing import Any
from sphinx.addnodes import document
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment import BuildEnvironment

class MypyHTMLBuilder(StandaloneHTMLBuilder):

    def __init__(self, app: Sphinx, env: BuildEnvironment) -> None:
        if False:
            return 10
        super().__init__(app, env)
        self._ref_to_doc = {}

    def write_doc(self, docname: str, doctree: document) -> None:
        if False:
            while True:
                i = 10
        super().write_doc(docname, doctree)
        self._ref_to_doc.update({_id: docname for _id in doctree.ids})

    def _verify_error_codes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        from mypy.errorcodes import error_codes
        missing_error_codes = {c for c in error_codes if f'code-{c}' not in self._ref_to_doc}
        if missing_error_codes:
            raise ValueError(f"Some error codes are not documented: {', '.join(sorted(missing_error_codes))}")

    def _write_ref_redirector(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if os.getenv('VERIFY_MYPY_ERROR_CODES'):
            self._verify_error_codes()
        p = Path(self.outdir) / '_refs.html'
        data = f"\n        <html>\n        <body>\n        <script>\n        const ref_to_doc = {json.dumps(self._ref_to_doc)};\n        const hash = window.location.hash.substring(1);\n        const doc = ref_to_doc[hash];\n        if (doc) {{\n            window.location.href = doc + '.html' + '#' + hash;\n        }} else {{\n            window.document.innerText = 'Unknown reference: ' + hash;\n        }}\n        </script>\n        </body>\n        </html>\n        "
        p.write_text(textwrap.dedent(data))

    def finish(self) -> None:
        if False:
            print('Hello World!')
        super().finish()
        self._write_ref_redirector()

def setup(app: Sphinx) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    app.add_builder(MypyHTMLBuilder, override=True)
    return {'version': '0.1', 'parallel_read_safe': True, 'parallel_write_safe': True}