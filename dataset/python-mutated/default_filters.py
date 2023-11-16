"""OCRmyPDF automatically installs these filters as plugins."""
from __future__ import annotations
from ocrmypdf import hookimpl

@hookimpl
def filter_pdf_page(page, image_filename, output_pdf):
    if False:
        print('Hello World!')
    return output_pdf