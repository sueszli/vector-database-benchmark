import re

def nfo_geturl(data):
    if False:
        i = 10
        return i + 15
    result = re.search('https://musicbrainz.org/(ws/2/)?artist/([0-9a-z\\-]*)', data)
    if result:
        return result.group(2)