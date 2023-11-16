import simplejson as json
import datetime
from rqalpha import const

def convert_dict_to_json(dict_obj):
    if False:
        return 10
    dict_obj = json.dumps(dict_obj, default=custom_encode)
    return dict_obj

def convert_json_to_dict(json_str):
    if False:
        return 10
    dict_obj = json.loads(json_str, object_hook=custom_decode)
    return dict_obj

def custom_encode(obj):
    if False:
        print('Hello World!')
    if isinstance(obj, datetime.datetime):
        obj = {'__datetime__': True, 'as_str': obj.strftime('%Y%m%dT%H:%M:%S.%f')}
    elif isinstance(obj, datetime.date):
        obj = {'__date__': True, 'as_str': obj.strftime('%Y%m%d')}
    elif isinstance(obj, const.CustomEnum):
        obj = {'__enum__': True, 'as_str': str(obj)}
    else:
        raise TypeError('Unserializable object {} of type {}'.format(obj, type(obj)))
    return obj

def custom_decode(obj):
    if False:
        return 10
    if '__datetime__' in obj:
        obj = datetime.datetime.strptime(obj['as_str'], '%Y%m%dT%H:%M:%S.%f')
    elif '__date__' in obj:
        obj = datetime.datetime.strptime(obj['as_str'], '%Y%m%d').date()
    elif '__enum__' in obj:
        [e, v] = obj['as_str'].split('.')
        obj = getattr(getattr(const, e), v)
    return obj