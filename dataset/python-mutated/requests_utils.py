from typing import Optional, List
import logging
from tenacity import retry, wait_exponential, retry_if_exception_type, stop_after_attempt, before_log, after_log
import requests
logger = logging.getLogger(__file__)

def request_with_retry(attempts: int=3, status_codes_to_retry: Optional[List[int]]=None, **kwargs) -> requests.Response:
    if False:
        while True:
            i = 10
    '\n    request_with_retry is a simple wrapper function that executes an HTTP request\n    with a configurable exponential backoff retry on failures.\n\n    All kwargs will be passed to ``requests.request``, so it accepts the same arguments.\n\n    Example Usage:\n    --------------\n\n    # Sending an HTTP request with default retry configs\n    res = request_with_retry(method="GET", url="https://example.com")\n\n    # Sending an HTTP request with custom number of attempts\n    res = request_with_retry(method="GET", url="https://example.com", attempts=10)\n\n    # Sending an HTTP request with custom HTTP codes to retry\n    res = request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=[408, 503])\n\n    # Sending an HTTP request with custom timeout in seconds\n    res = request_with_retry(method="GET", url="https://example.com", timeout=5)\n\n    # Sending an HTTP request with custom authorization handling\n    class CustomAuth(requests.auth.AuthBase):\n        def __call__(self, r):\n            r.headers["authorization"] = "Basic <my_token_here>"\n            return r\n\n    res = request_with_retry(method="GET", url="https://example.com", auth=CustomAuth())\n\n    # All of the above combined\n    res = request_with_retry(\n        method="GET",\n        url="https://example.com",\n        auth=CustomAuth(),\n        attempts=10,\n        status_codes_to_retry=[408, 503],\n        timeout=5\n    )\n\n    # Sending a POST request\n    res = request_with_retry(method="POST", url="https://example.com", data={"key": "value"}, attempts=10)\n\n    # Retry all 5xx status codes\n    res = request_with_retry(method="GET", url="https://example.com", status_codes_to_retry=list(range(500, 600)))\n\n    :param attempts: Maximum number of attempts to retry the request, defaults to 3\n    :param status_codes_to_retry: List of HTTP status codes that will trigger a retry, defaults to [408, 418, 429, 503]:\n        - `408: Request Timeout`\n        - `418`\n        - `429: Too Many Requests`\n        - `503: Service Unavailable`\n    :param **kwargs: Optional arguments that ``request`` takes.\n    :return: :class:`Response <Response>` object\n    '
    if status_codes_to_retry is None:
        status_codes_to_retry = [408, 418, 429, 503]

    @retry(reraise=True, wait=wait_exponential(), retry=retry_if_exception_type((requests.HTTPError, TimeoutError)), stop=stop_after_attempt(attempts), before=before_log(logger, logging.DEBUG), after=after_log(logger, logging.DEBUG))
    def run():
        if False:
            print('Hello World!')
        timeout = kwargs.pop('timeout', 10)
        res = requests.request(**kwargs, timeout=timeout)
        if res.status_code in status_codes_to_retry:
            res.raise_for_status()
        return res
    res = run()
    res.raise_for_status()
    return res