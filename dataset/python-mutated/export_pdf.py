import asyncio
import concurrent.futures
import logging
import os
import tempfile
from subprocess import PIPE, Popen
pdf_export_available = True
try:
    from pyppeteer import launch
except Exception:
    pdf_export_available = False

async def html_to_pdf(html_file, pdf_file, pyppeteer_args=None):
    """Convert a HTML file to a PDF"""
    browser = await launch(handleSIGINT=False, handleSIGTERM=False, handleSIGHUP=False, headless=True, args=['--no-sandbox'])
    page = await browser.newPage()
    await page.setViewport(dict(width=994, height=768))
    await page.emulateMedia('screen')
    await page.goto(f'file://{html_file}', {'waitUntil': ['networkidle2']})
    page_margins = {'left': '20px', 'right': '20px', 'top': '30px', 'bottom': '30px'}
    dimensions = await page.evaluate('() => {\n        return {\n            width: document.body.scrollWidth,\n            height: document.body.scrollHeight,\n            offsetWidth: document.body.offsetWidth,\n            offsetHeight: document.body.offsetHeight,\n            deviceScaleFactor: window.devicePixelRatio,\n        }\n    }')
    width = dimensions['width']
    height = dimensions['height']
    await page.evaluate('\n    function getOffset( el ) {\n        var _x = 0;\n        var _y = 0;\n        while( el && !isNaN( el.offsetLeft ) && !isNaN( el.offsetTop ) ) {\n            _x += el.offsetLeft - el.scrollLeft;\n            _y += el.offsetTop - el.scrollTop;\n            el = el.offsetParent;\n        }\n        return { top: _y, left: _x };\n        }\n    ', force_expr=True)
    await page.addStyleTag({'content': '\n                #notebook-container {\n                    box-shadow: none;\n                    padding: unset\n                }\n                div.cell {\n                    page-break-inside: avoid;\n                    break-inside: avoid;\n                }\n                div.output_wrapper {\n                    page-break-inside: avoid;\n                    break-inside: avoid;\n                }\n                div.output {\n                    page-break-inside: avoid;\n                    break-inside: avoid;\n                }\n                /* Jupyterlab based HTML uses these classes */\n                .jp-Cell-inputWrapper {\n                    page-break-inside: avoid;\n                    break-inside: avoid;\n                }\n                .jp-Cell-outputWrapper {\n                    page-break-inside: avoid;\n                    break-inside: avoid;\n                }\n                .jp-Notebook {\n                    margin: 0px;\n                }\n                /* Hide the message box used by MathJax */\n                #MathJax_Message {\n                    display: none;\n                }\n         '})
    await page.pdf({'path': pdf_file, 'format': 'A4', 'printBackground': True, 'margin': page_margins})
    await browser.close()

def install_chromium():
    if False:
        while True:
            i = 10
    command = ['pyppeteer-install']
    with Popen(command, stdout=PIPE, stderr=PIPE) as proc:
        print(proc.stdout.read())
        print(proc.stderr.read())

def to_pdf(html_input_file, pdf_output_file):
    if False:
        print('Hello World!')
    if not pdf_export_available:
        print('PDF export not available')
        return
    prev_log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)
    pool = concurrent.futures.ThreadPoolExecutor()
    pool.submit(asyncio.run, html_to_pdf(html_input_file, pdf_output_file)).result()
    logging.getLogger().setLevel(prev_log_level)