"""Provide a bot to tests"""
import base64
import json
import os
import random
FALLBACKS = 'W3sidG9rZW4iOiAiNTc5Njk0NzE0OkFBRnBLOHc2emtrVXJENHhTZVl3RjNNTzhlLTRHcm1jeTdjIiwgInBheW1lbnRfcHJvdmlkZXJfdG9rZW4iOiAiMjg0Njg1MDYzOlRFU1Q6TmpRME5qWmxOekk1WWpKaSIsICJjaGF0X2 lkIjogIjY3NTY2NjIyNCIsICJzdXBlcl9ncm91cF9pZCI6ICItMTAwMTMxMDkxMTEzNSIsICJmb3J1bV9ncm91cF9pZCI6ICItMTAwMTgzODAwNDU3NyIsICJjaGFubmVsX2lkIjogIkBweXRob250ZWxlZ3JhbWJvdHRlc3RzIi wgIm5hbWUiOiAiUFRCIHRlc3RzIGZhbGxiYWNrIDEiLCAidXNlcm5hbWUiOiAiQHB0Yl9mYWxsYmFja18xX2JvdCJ9LCB7InRva2VuIjogIjU1ODE5NDA2NjpBQUZ3RFBJRmx6R1VsQ2FXSHRUT0VYNFJGclg4dTlETXFmbyIsIC JwYXltZW50X3Byb3ZpZGVyX3Rva2VuIjogIjI4NDY4NTA2MzpURVNUOllqRXdPRFF3TVRGbU5EY3kiLCAiY2hhdF9pZCI6ICI2NzU2NjYyMjQiLCAic3VwZXJfZ3JvdXBfaWQiOiAiLTEwMDEyMjEyMTY4MzAiLCAiZm9ydW1fZ3 JvdXBfaWQiOiAiLTEwMDE4NTc4NDgzMTQiLCAiY2hhbm5lbF9pZCI6ICJAcHl0aG9udGVsZWdyYW1ib3R0ZXN0cyIsICJuYW1lIjogIlBUQiB0ZXN0cyBmYWxsYmFjayAyIiwgInVzZXJuYW1lIjogIkBwdGJfZmFsbGJhY2tfMl9ib3QifV0='
GITHUB_ACTION = os.getenv('GITHUB_ACTION', None)
BOTS = os.getenv('BOTS', None)
JOB_INDEX = os.getenv('JOB_INDEX', None)
if GITHUB_ACTION is not None and BOTS is not None and (JOB_INDEX is not None):
    BOTS = json.loads(base64.b64decode(BOTS).decode('utf-8'))
    JOB_INDEX = int(JOB_INDEX)
FALLBACKS = json.loads(base64.b64decode(FALLBACKS).decode('utf-8'))

class BotInfoProvider:

    def __init__(self):
        if False:
            return 10
        self._cached = {}

    @staticmethod
    def _get_value(key, fallback):
        if False:
            while True:
                i = 10
        if GITHUB_ACTION is not None and BOTS is not None and (JOB_INDEX is not None):
            try:
                return BOTS[JOB_INDEX][key]
            except (IndexError, KeyError):
                pass
        return fallback

    def get_info(self):
        if False:
            while True:
                i = 10
        if self._cached:
            return self._cached
        self._cached = {k: self._get_value(k, v) for (k, v) in random.choice(FALLBACKS).items()}
        return self._cached
BOT_INFO_PROVIDER = BotInfoProvider()