import re
from urllib.parse import urlunparse, urlparse
from searx import settings
from searx.plugins import logger
from flask_babel import gettext
name = gettext('Hostname replace')
description = gettext('Rewrite result hostnames or remove results based on the hostname')
default_on = False
preference_section = 'general'
plugin_id = 'hostname_replace'
replacements = {re.compile(p): r for (p, r) in settings[plugin_id].items()} if plugin_id in settings else {}
logger = logger.getChild(plugin_id)
parsed = 'parsed_url'
_url_fields = ['iframe_src', 'audio_src']

def on_result(request, search, result):
    if False:
        for i in range(10):
            print('nop')
    for (pattern, replacement) in replacements.items():
        if parsed in result:
            if pattern.search(result[parsed].netloc):
                if not replacement:
                    return False
                result[parsed] = result[parsed]._replace(netloc=pattern.sub(replacement, result[parsed].netloc))
                result['url'] = urlunparse(result[parsed])
        for url_field in _url_fields:
            if result.get(url_field):
                url_src = urlparse(result[url_field])
                if pattern.search(url_src.netloc):
                    if not replacement:
                        del result[url_field]
                    else:
                        url_src = url_src._replace(netloc=pattern.sub(replacement, url_src.netloc))
                        result[url_field] = urlunparse(url_src)
    return True