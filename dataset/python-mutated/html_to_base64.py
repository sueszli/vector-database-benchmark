import base64
import os
import random
import string
from html2image import Html2Image

def html_to_base64(code):
    if False:
        i = 10
        return i + 15
    hti = Html2Image()
    temp_filename = ''.join(random.choices(string.digits, k=10)) + '.png'
    hti.screenshot(html_str=code, save_as=temp_filename, size=(1280, 720))
    with open(temp_filename, 'rb') as image_file:
        screenshot_base64 = base64.b64encode(image_file.read()).decode()
    os.remove(temp_filename)
    return screenshot_base64