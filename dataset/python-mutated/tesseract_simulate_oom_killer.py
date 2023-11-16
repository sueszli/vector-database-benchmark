"""Tesseract no-op plugin that simulates the OOM killer on page 4.

OCRmyPDF can use a lot of memory, even that it might trigger the
OOM killer on Linux or similar features on other platforms. We want to
ensure we fail with an error rather than deadlock in such cases.

Page 4 was chosen because of this number's association with bad luck
in many East Asian cultures.
"""
from __future__ import annotations
import os
import signal
from pathlib import Path
from ocrmypdf import hookimpl
parent_file = Path(__file__).with_name('tesseract_noop.py')
parent = compile(parent_file.read_text(), parent_file, mode='exec')
exec(parent)
NoopOcrEngine = locals()['NoopOcrEngine']

class Page4Engine(NoopOcrEngine):

    def __str__(self):
        if False:
            return 10
        return f'NO-OP Page 4 {NoopOcrEngine.version()}'

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr, output_text, options):
        if False:
            while True:
                i = 10
        if input_file.stem.startswith('000004'):
            os.kill(os.getpid(), signal.SIGKILL)
        else:
            return NoopOcrEngine.generate_hocr(input_file, output_hocr, output_text, options)

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options):
        if False:
            for i in range(10):
                print('nop')
        if input_file.stem.startswith('000004'):
            os.kill(os.getpid(), signal.SIGKILL)
        else:
            return NoopOcrEngine.generate_pdf(input_file, output_pdf, output_text, options)

@hookimpl
def check_options(options):
    if False:
        print('Hello World!')
    if options.use_threads:
        raise ValueError("I'm not compatible with use_threads")

@hookimpl
def get_ocr_engine():
    if False:
        while True:
            i = 10
    return Page4Engine()