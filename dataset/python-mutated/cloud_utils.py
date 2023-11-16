from typing import TYPE_CHECKING, Any, Optional
from django.conf import settings
from django.db.utils import ProgrammingError
from sentry_sdk import capture_exception
if TYPE_CHECKING:
    from ee.models.license import License
is_cloud_cached: Optional[bool] = None
is_instance_licensed_cached: Optional[bool] = None
instance_license_cached: Optional['License'] = None

def is_cloud():
    if False:
        return 10
    global is_cloud_cached
    if not settings.EE_AVAILABLE:
        return False
    if isinstance(is_cloud_cached, bool):
        return is_cloud_cached
    if not is_cloud_cached:
        try:
            from ee.models.license import License
            license = License.objects.first_valid()
            is_cloud_cached = license.plan == 'cloud' if license else False
        except ProgrammingError:
            pass
        except Exception as e:
            print('ERROR: Unable to check license', e)
            capture_exception(e)
    return is_cloud_cached

def TEST_clear_cloud_cache(value: Optional[bool]=None):
    if False:
        while True:
            i = 10
    global is_cloud_cached
    is_cloud_cached = value

def get_cached_instance_license() -> Optional['License']:
    if False:
        for i in range(10):
            print('nop')
    'Returns the first valid license and caches the value for the lifetime of the instance, as it is not expected to change.\n    If there is no valid license, it returns None.\n    '
    global instance_license_cached
    global is_instance_licensed_cached
    try:
        from ee.models.license import License
    except ProgrammingError:
        pass
    except Exception as e:
        capture_exception(e)
        return None
    if isinstance(instance_license_cached, License):
        return instance_license_cached
    if is_instance_licensed_cached is False:
        return None
    license = License.objects.first_valid()
    if license:
        instance_license_cached = license
        is_instance_licensed_cached = True
    else:
        is_instance_licensed_cached = False
    return instance_license_cached

def TEST_clear_instance_license_cache(is_instance_licensed: Optional[bool]=None, instance_license: Optional[Any]=None):
    if False:
        for i in range(10):
            print('nop')
    global instance_license_cached
    instance_license_cached = instance_license
    global is_instance_licensed_cached
    is_instance_licensed_cached = is_instance_licensed