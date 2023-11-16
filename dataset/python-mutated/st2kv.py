import logging
import six
from orquesta import exceptions as exc
from st2common.exceptions import db as db_exc
from st2common.persistence import auth as auth_db_access
from st2common.util import keyvalue as kvp_util
LOG = logging.getLogger(__name__)

def st2kv_(context, key, **kwargs):
    if False:
        while True:
            i = 10
    if not isinstance(key, six.string_types):
        raise TypeError('Given key is not typeof string.')
    decrypt = kwargs.get('decrypt', False)
    if not isinstance(decrypt, bool):
        raise TypeError('Decrypt parameter is not typeof bool.')
    try:
        username = context['__vars']['st2']['user']
    except KeyError:
        raise KeyError('Could not get user from context.')
    try:
        user_db = auth_db_access.User.get(username)
    except Exception as e:
        raise Exception('Failed to retrieve User object for user "%s", "%s"' % (username, six.text_type(e)))
    has_default = 'default' in kwargs
    default_value = kwargs.get('default')
    try:
        return kvp_util.get_key(key=key, user_db=user_db, decrypt=decrypt)
    except db_exc.StackStormDBObjectNotFoundError as e:
        if not has_default:
            raise exc.ExpressionEvaluationException(str(e))
        else:
            return default_value
    except Exception as e:
        raise exc.ExpressionEvaluationException(str(e))