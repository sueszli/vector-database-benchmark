"""Tesseract bad utf8.

In some cases, some versions of Tesseract can output binary gibberish or data
that is not UTF-8 compatible, so we are forced to check that we can convert it
and present it to the user.
"""
from __future__ import annotations
from contextlib import contextmanager
from subprocess import CalledProcessError
from unittest.mock import patch
from ocrmypdf import hookimpl
from ocrmypdf.builtin_plugins.tesseract_ocr import TesseractOcrEngine

def bad_utf8(*args, **kwargs):
    if False:
        return 10
    raise CalledProcessError(1, 'tesseract', output=b'\x96\xb3\x8c\xf8\x82\xc8UTF-8\n', stderr=b'')

@contextmanager
def patch_tesseract_run():
    if False:
        return 10
    with patch('ocrmypdf._exec.tesseract.run') as mock:
        mock.side_effect = bad_utf8
        yield
        mock.assert_called()

class BadUtf8OcrEngine(TesseractOcrEngine):

    @staticmethod
    def generate_hocr(input_file, output_hocr, output_text, options):
        if False:
            return 10
        with patch_tesseract_run():
            TesseractOcrEngine.generate_hocr(input_file, output_hocr, output_text, options)

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options):
        if False:
            while True:
                i = 10
        with patch_tesseract_run():
            TesseractOcrEngine.generate_pdf(input_file, output_pdf, output_text, options)

@hookimpl
def get_ocr_engine():
    if False:
        i = 10
        return i + 15
    return BadUtf8OcrEngine()