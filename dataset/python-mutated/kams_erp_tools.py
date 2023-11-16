import re

def clean_html(raw_html):
    if False:
        i = 10
        return i + 15
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext