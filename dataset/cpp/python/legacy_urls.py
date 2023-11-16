from typing import List, Union

from django.urls import URLPattern, URLResolver, path

import zerver.views
import zerver.views.auth
import zerver.views.report
import zerver.views.streams
import zerver.views.tutorial

# Future endpoints should add to urls.py, which includes these legacy URLs

legacy_urls: List[Union[URLPattern, URLResolver]] = [
    # These are json format views used by the web client.  They require a logged in browser.
    # We should remove this endpoint and all code related to it.
    # It returns a 404 if the stream doesn't exist, which is confusing
    # for devs, and I don't think we need to go to the server
    # any more to find out about subscriptions, since they are already
    # pushed to us via the event system.
    path("json/subscriptions/exists", zerver.views.streams.json_stream_exists),
]
