import re
import sys
from flask import got_request_exception, current_app
from flask_restful import Api, http_status_message
from werkzeug.datastructures import Headers
from werkzeug.exceptions import HTTPException

class ExternalApi(Api):

    def handle_error(self, e):
        if False:
            while True:
                i = 10
        'Error handler for the API transforms a raised exception into a Flask\n        response, with the appropriate HTTP status code and body.\n\n        :param e: the raised Exception object\n        :type e: Exception\n\n        '
        got_request_exception.send(current_app, exception=e)
        headers = Headers()
        if isinstance(e, HTTPException):
            if e.response is not None:
                resp = e.get_response()
                return resp
            status_code = e.code
            default_data = {'code': re.sub('(?<!^)(?=[A-Z])', '_', type(e).__name__).lower(), 'message': getattr(e, 'description', http_status_message(status_code)), 'status': status_code}
            headers = e.get_response().headers
        elif isinstance(e, ValueError):
            status_code = 400
            default_data = {'code': 'invalid_param', 'message': str(e), 'status': status_code}
        else:
            status_code = 500
            default_data = {'message': http_status_message(status_code)}
        remove_headers = ('Content-Length',)
        for header in remove_headers:
            headers.pop(header, None)
        data = getattr(e, 'data', default_data)
        error_cls_name = type(e).__name__
        if error_cls_name in self.errors:
            custom_data = self.errors.get(error_cls_name, {})
            custom_data = custom_data.copy()
            status_code = custom_data.get('status', 500)
            if 'message' in custom_data:
                custom_data['message'] = custom_data['message'].format(message=str(e.description if hasattr(e, 'description') else e))
            data.update(custom_data)
        if status_code and status_code >= 500:
            exc_info = sys.exc_info()
            if exc_info[1] is None:
                exc_info = None
            current_app.log_exception(exc_info)
        if status_code == 406 and self.default_mediatype is None:
            supported_mediatypes = list(self.representations.keys())
            fallback_mediatype = supported_mediatypes[0] if supported_mediatypes else 'text/plain'
            data = {'code': 'not_acceptable', 'message': data.get('message')}
            resp = self.make_response(data, status_code, headers, fallback_mediatype=fallback_mediatype)
        elif status_code == 400:
            if isinstance(data.get('message'), dict):
                (param_key, param_value) = list(data.get('message').items())[0]
                data = {'code': 'invalid_param', 'message': param_value, 'params': param_key}
            elif 'code' not in data:
                data['code'] = 'unknown'
            resp = self.make_response(data, status_code, headers)
        else:
            if 'code' not in data:
                data['code'] = 'unknown'
            resp = self.make_response(data, status_code, headers)
        if status_code == 401:
            resp = self.unauthorized(resp)
        return resp