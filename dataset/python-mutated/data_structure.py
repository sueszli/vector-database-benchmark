from dataclasses import dataclass

@dataclass
class SessionOptions:
    rpc_version: str = '2.0'
    pubsub_version: str = '1.0'

class Request:
    """fuo 协议请求对象"""

    def __init__(self, cmd, cmd_args=None, cmd_options=None, options=None, has_heredoc=False, heredoc_word=None):
        if False:
            print('Hello World!')
        self.cmd = cmd
        self.cmd_args = cmd_args or []
        self.cmd_options = cmd_options or {}
        self.options = options or {}
        self.has_heredoc = has_heredoc
        self.heredoc_word = heredoc_word

    def set_heredoc_body(self, body):
        if False:
            for i in range(10):
                print('nop')
        assert self.has_heredoc is True and self.heredoc_word
        self.cmd_args = [body]

class Response:

    def __init__(self, ok=True, text='', req=None, options=None):
        if False:
            while True:
                i = 10
        self.code = 'OK' if ok else 'Oops'
        self.text = text
        self.options = options or {}
        self.req = req