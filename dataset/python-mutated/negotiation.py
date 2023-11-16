"""
Content negotiation deals with selecting an appropriate renderer given the
incoming request.  Typically this will be based on the request's Accept header.
"""
from django.http import Http404
from rest_framework import exceptions
from rest_framework.settings import api_settings
from rest_framework.utils.mediatypes import _MediaType, media_type_matches, order_by_precedence

class BaseContentNegotiation:

    def select_parser(self, request, parsers):
        if False:
            print('Hello World!')
        raise NotImplementedError('.select_parser() must be implemented')

    def select_renderer(self, request, renderers, format_suffix=None):
        if False:
            print('Hello World!')
        raise NotImplementedError('.select_renderer() must be implemented')

class DefaultContentNegotiation(BaseContentNegotiation):
    settings = api_settings

    def select_parser(self, request, parsers):
        if False:
            return 10
        '\n        Given a list of parsers and a media type, return the appropriate\n        parser to handle the incoming request.\n        '
        for parser in parsers:
            if media_type_matches(parser.media_type, request.content_type):
                return parser
        return None

    def select_renderer(self, request, renderers, format_suffix=None):
        if False:
            while True:
                i = 10
        '\n        Given a request and a list of renderers, return a two-tuple of:\n        (renderer, media type).\n        '
        format_query_param = self.settings.URL_FORMAT_OVERRIDE
        format = format_suffix or request.query_params.get(format_query_param)
        if format:
            renderers = self.filter_renderers(renderers, format)
        accepts = self.get_accept_list(request)
        for media_type_set in order_by_precedence(accepts):
            for renderer in renderers:
                for media_type in media_type_set:
                    if media_type_matches(renderer.media_type, media_type):
                        media_type_wrapper = _MediaType(media_type)
                        if _MediaType(renderer.media_type).precedence > media_type_wrapper.precedence:
                            full_media_type = ';'.join((renderer.media_type,) + tuple(('{}={}'.format(key, value) for (key, value) in media_type_wrapper.params.items())))
                            return (renderer, full_media_type)
                        else:
                            return (renderer, media_type)
        raise exceptions.NotAcceptable(available_renderers=renderers)

    def filter_renderers(self, renderers, format):
        if False:
            for i in range(10):
                print('nop')
        "\n        If there is a '.json' style format suffix, filter the renderers\n        so that we only negotiation against those that accept that format.\n        "
        renderers = [renderer for renderer in renderers if renderer.format == format]
        if not renderers:
            raise Http404
        return renderers

    def get_accept_list(self, request):
        if False:
            for i in range(10):
                print('nop')
        '\n        Given the incoming request, return a tokenized list of media\n        type strings.\n        '
        header = request.META.get('HTTP_ACCEPT', '*/*')
        return [token.strip() for token in header.split(',')]