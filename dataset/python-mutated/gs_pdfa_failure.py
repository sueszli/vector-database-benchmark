from __future__ import annotations
from unittest.mock import patch
from ocrmypdf import hookimpl
from ocrmypdf.builtin_plugins import ghostscript
from ocrmypdf.subprocess import run_polling_stderr

def run_rig_args(args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    new_args = [arg for arg in args if not arg.startswith('-dPDFA') and (not arg.endswith('.ps'))]
    proc = run_polling_stderr(new_args, **kwargs)
    return proc

@hookimpl
def generate_pdfa(pdf_pages, pdfmark, output_file, context, pdf_version, pdfa_part):
    if False:
        print('Hello World!')
    with patch('ocrmypdf._exec.ghostscript.run_polling_stderr') as mock:
        mock.side_effect = run_rig_args
        ghostscript.generate_pdfa(pdf_pages=pdf_pages, pdfmark=pdfmark, output_file=output_file, context=context, pdf_version=pdf_version, pdfa_part=pdfa_part, progressbar_class=None, stop_on_soft_error=True)
        mock.assert_called()
        return output_file