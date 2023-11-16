import re
import urllib.parse as urlparse
REPLACE_STR = '$encrypted$'

class UriCleaner(object):
    REPLACE_STR = REPLACE_STR
    SENSITIVE_URI_PATTERN = re.compile('(\\w{1,20}:(\\/?\\/?)[^\\s]+)', re.MULTILINE)

    @staticmethod
    def remove_sensitive(cleartext):
        if False:
            i = 10
            return i + 15
        redactedtext = cleartext
        text_index = 0
        while True:
            match = UriCleaner.SENSITIVE_URI_PATTERN.search(redactedtext, text_index)
            if not match:
                break
            uri_str = match.group(1)
            try:
                o = urlparse.urlsplit(uri_str)
                if not o.username and (not o.password):
                    if o.netloc and ':' in o.netloc:
                        (username, password) = o.netloc.split(':')
                    else:
                        text_index += len(match.group(1))
                        continue
                else:
                    username = o.username
                    password = o.password
                uri_str = redactedtext[match.start():match.end()]
                if username:
                    uri_str = uri_str.replace(username, UriCleaner.REPLACE_STR, 1)
                if password:
                    uri_str = uri_str.replace(password, UriCleaner.REPLACE_STR, 2)
                t = redactedtext[:match.start()] + uri_str
                text_index = len(t)
                if match.end() < len(redactedtext):
                    t += redactedtext[match.end():]
                redactedtext = t
                if text_index >= len(redactedtext):
                    text_index = len(redactedtext) - 1
            except ValueError:
                redactedtext = redactedtext[:match.start()] + UriCleaner.REPLACE_STR + redactedtext[match.end():]
                text_index = match.start() + len(UriCleaner.REPLACE_STR)
        return redactedtext

class PlainTextCleaner(object):
    REPLACE_STR = REPLACE_STR

    @staticmethod
    def remove_sensitive(cleartext, sensitive):
        if False:
            return 10
        if sensitive == '':
            return cleartext
        return re.sub('%s' % re.escape(sensitive), '$encrypted$', cleartext)