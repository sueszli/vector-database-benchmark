import hashlib
from django.conf import settings
from django.contrib.staticfiles.storage import HashedFilesMixin
from django.templatetags.static import static
from wagtail import __version__
try:
    use_version_strings = settings.WAGTAILADMIN_STATIC_FILE_VERSION_STRINGS
except AttributeError:
    if settings.DEBUG:
        use_version_strings = True
    else:
        try:
            from django.conf import STATICFILES_STORAGE_ALIAS
            from django.core.files.storage import storages
            storage = storages[STATICFILES_STORAGE_ALIAS].__class__
        except ImportError:
            from django.core.files.storage import get_storage_class
            storage = get_storage_class(settings.STATICFILES_STORAGE)
        use_version_strings = not issubclass(storage, HashedFilesMixin)
if use_version_strings:
    VERSION_HASH = hashlib.sha1((__version__ + settings.SECRET_KEY).encode('utf-8')).hexdigest()[:8]
else:
    VERSION_HASH = None

def versioned_static(path):
    if False:
        i = 10
        return i + 15
    "\n    Wrapper for Django's static file finder to append a cache-busting query parameter\n    that updates on each Wagtail version\n    "
    if path.startswith(('http://', 'https://', '/')):
        return path
    base_url = static(path)
    if VERSION_HASH is None or '?' in base_url:
        return base_url
    else:
        return base_url + '?v=' + VERSION_HASH