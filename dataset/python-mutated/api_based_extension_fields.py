from flask_restful import fields
from libs.helper import TimestampField

class HiddenAPIKey(fields.Raw):

    def output(self, key, obj):
        if False:
            for i in range(10):
                print('nop')
        api_key = obj.api_key
        if len(api_key) <= 8:
            return api_key[0] + '******' + api_key[-1]
        else:
            return api_key[:3] + '******' + api_key[-3:]
api_based_extension_fields = {'id': fields.String, 'name': fields.String, 'api_endpoint': fields.String, 'api_key': HiddenAPIKey, 'created_at': TimestampField}