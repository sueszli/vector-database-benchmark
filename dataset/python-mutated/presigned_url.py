from localstack.http import Response
from localstack.services.s3.presigned_url import S3PreSignedURLRequestHandler
from ..api import RequestContext
from ..chain import Handler, HandlerChain

class ParsePreSignedUrlRequest(Handler):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.pre_signed_handlers: dict[str, Handler] = {'s3': S3PreSignedURLRequestHandler()}

    def __call__(self, chain: HandlerChain, context: RequestContext, response: Response):
        if False:
            for i in range(10):
                print('nop')
        if not context.service:
            return
        if (handler := self.pre_signed_handlers.get(context.service.service_name)):
            handler(chain, context, response)