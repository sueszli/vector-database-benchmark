import asyncio
import concurrent.futures
import logging
import os
import tempfile
from subprocess import PIPE, Popen

from pyppeteer import launch


async def html_to_png(html_file, png_file, pyppeteer_args=None):
    """Convert a HTML file to a PDF"""
    browser = await launch(
        handleSIGINT=False,
        handleSIGTERM=False,
        handleSIGHUP=False,
        headless=True,
        args=["--no-sandbox"],
    )

    page = await browser.newPage()
    await page.setViewport(dict(width=600, height=400))
    await page.emulateMedia("screen")

    await page.goto(f"file://{html_file}", {"waitUntil": ["networkidle2"]})

    page_margins = {
        "left": "20px",
        "right": "20px",
        "top": "30px",
        "bottom": "30px",
    }

    dimensions = await page.evaluate(
        """() => {
        return {
            width: document.body.scrollWidth,
            height: document.body.scrollHeight,
            offsetWidth: document.body.offsetWidth,
            offsetHeight: document.body.offsetHeight,
            deviceScaleFactor: window.devicePixelRatio,
        }
    }"""
    )
    width = dimensions["width"]
    height = dimensions["height"]

    await page.evaluate(
        """
    function getOffset( el ) {
        var _x = 0;
        var _y = 0;
        while( el && !isNaN( el.offsetLeft ) && !isNaN( el.offsetTop ) ) {
            _x += el.offsetLeft - el.scrollLeft;
            _y += el.offsetTop - el.scrollTop;
            el = el.offsetParent;
        }
        return { top: _y, left: _x };
        }
    """,
        force_expr=True,
    )

    await page.addStyleTag(
        {
            "content": """
                #notebook-container {
                    box-shadow: none;
                    padding: unset
                }
                div.cell {
                    page-break-inside: avoid;
                    break-inside: avoid;
                }
                div.output_wrapper {
                    page-break-inside: avoid;
                    break-inside: avoid;
                }
                div.output {
                    page-break-inside: avoid;
                    break-inside: avoid;
                }
                /* Jupyterlab based HTML uses these classes */
                .jp-Cell-inputWrapper {
                    page-break-inside: avoid;
                    break-inside: avoid;
                }
                .jp-Cell-outputWrapper {
                    page-break-inside: avoid;
                    break-inside: avoid;
                }
                .jp-Notebook {
                    margin: 0px;
                }
                /* Hide the message box used by MathJax */
                #MathJax_Message {
                    display: none;
                }
         """
        }
    )

    await page.screenshot(
        {
            "path": png_file,
            # "clip": {
            #     "x": 0,
            #     "y": 0,
            #     "width": 500,
            #     "height": 500
            # }
        }
    )

    await browser.close()


def install_chromium():
    command = ["pyppeteer-install"]
    with Popen(command, stdout=PIPE, stderr=PIPE) as proc:
        print(proc.stdout.read())
        print(proc.stderr.read())


def to_png(html_input_file, png_output_file):
    # make sure chromium is installed
    # install_chromium()

    # dont want to see DEBUG logs for chromium ...
    prev_log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.ERROR)

    # convert notebook to PDF
    pool = concurrent.futures.ThreadPoolExecutor()
    pool.submit(
        asyncio.run,
        html_to_png(html_input_file, png_output_file),
    ).result()

    # set previous log level
    logging.getLogger().setLevel(prev_log_level)
