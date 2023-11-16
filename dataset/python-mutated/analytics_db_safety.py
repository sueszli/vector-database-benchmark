import re
from flask_babel import lazy_gettext as _
from sqlalchemy.engine.url import URL
from sqlalchemy.exc import NoSuchModuleError
from superset import feature_flag_manager
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import SupersetSecurityException
BLOCKLIST = {re.compile('sqlite(?:\\+[^\\s]*)?$'), re.compile('shillelagh$'), re.compile('shillelagh\\+apsw$')}

def check_sqlalchemy_uri(uri: URL) -> None:
    if False:
        print('Hello World!')
    if not feature_flag_manager.is_feature_enabled('ENABLE_SUPERSET_META_DB'):
        BLOCKLIST.add(re.compile('superset$'))
    for blocklist_regex in BLOCKLIST:
        if not re.match(blocklist_regex, uri.drivername):
            continue
        try:
            dialect = uri.get_dialect().__name__
        except (NoSuchModuleError, ValueError):
            dialect = uri.drivername
        raise SupersetSecurityException(SupersetError(error_type=SupersetErrorType.DATABASE_SECURITY_ACCESS_ERROR, message=_('%(dialect)s cannot be used as a data source for security reasons.', dialect=dialect), level=ErrorLevel.ERROR))