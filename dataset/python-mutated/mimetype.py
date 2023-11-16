import re
import json
from defusedxml import ElementTree
from lib.core.settings import QUERY_STRING_REGEX

class MimeTypeUtils:

    @staticmethod
    def is_json(content):
        if False:
            i = 10
            return i + 15
        try:
            json.loads(content)
            return True
        except json.decoder.JSONDecodeError:
            return False

    @staticmethod
    def is_xml(content):
        if False:
            i = 10
            return i + 15
        try:
            ElementTree.fromstring(content)
            return True
        except ElementTree.ParseError:
            return False
        except Exception:
            return True

    @staticmethod
    def is_query_string(content):
        if False:
            i = 10
            return i + 15
        if re.match(QUERY_STRING_REGEX, content):
            return True
        return False

def guess_mimetype(content):
    if False:
        return 10
    if MimeTypeUtils.is_json(content):
        return 'application/json'
    elif MimeTypeUtils.is_xml(content):
        return 'application/xml'
    elif MimeTypeUtils.is_query_string(content):
        return 'application/x-www-form-urlencoded'
    else:
        return 'text/plain'