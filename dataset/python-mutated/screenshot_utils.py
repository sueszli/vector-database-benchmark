import os
import logging
from pathlib import Path
import asyncio
try:
    from pyppeteer import launch
except ImportError:
    from typing_extensions import Coroutine
    import typing
    typing.Coroutine = Coroutine
    from pyppeteer import launch
from PIL import Image
CELL_DROPDOWN_SELECTOR = '#menus > div > div > ul > li:nth-child(5) > a'
RUN_ALL_SELECTOR = '#run_all_cells > a'
SECONDS_BEFORE_REEXECUTION = 0.5
SECONDS_BEFORE_SCREENSHOT = 10.0

async def get_notebook_page_height(page):
    return await page.evaluate("document.querySelector('#notebook').scrollHeight")

async def go_to_url(page, url):
    num_attempts = 0
    while True:
        try:
            num_attempts += 1
            logging.info('Attempting to read page %s' % url)
            await page.goto(url)
            break
        except Exception:
            await asyncio.sleep(SECONDS_BEFORE_REEXECUTION)
            if num_attempts > 5:
                raise TimeoutError('Jupyter notebook failed to open')

def rename_png(file_name):
    if False:
        print('Hello World!')
    return str(Path(file_name).with_suffix('.png'))

def is_ipynb(file_name):
    if False:
        print('Hello World!')
    return str(file_name).endswith('.ipynb')

async def go_to_page_and_screenshot(url, file_name, output_dir='.', sleep_seconds=SECONDS_BEFORE_SCREENSHOT):
    browser = None
    try:
        browser = await launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
        page = await browser.newPage()
        await go_to_url(page, url)
        if is_ipynb(file_name):
            await page.waitForSelector(CELL_DROPDOWN_SELECTOR, {'timeout': 10000})
            await page.click(CELL_DROPDOWN_SELECTOR)
            await page.click(RUN_ALL_SELECTOR)
            await asyncio.sleep(sleep_seconds)
            page_height = await get_notebook_page_height(page)
            await page.setViewport({'width': 768, 'height': page_height})
        else:
            await asyncio.sleep(sleep_seconds)
        str_path = str(output_dir)
        screenshot_path = os.path.join(str_path, rename_png(file_name))
        logging.info('Writing screenshot to %s' % screenshot_path)
        await page._screenshotTask('png', {'path': screenshot_path, 'fullPage': True})
        await browser.close()
    except Exception as e:
        browser.process.kill()
        raise e