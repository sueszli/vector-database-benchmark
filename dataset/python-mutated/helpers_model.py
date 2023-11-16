"""Provides helper functions used throughout the InvenTree project that access the database."""
import io
import logging
from decimal import Decimal
from urllib.parse import urljoin
from django.conf import settings
from django.core.validators import URLValidator
from django.db.utils import OperationalError, ProgrammingError
from django.utils.translation import gettext_lazy as _
import requests
from djmoney.contrib.exchange.models import convert_money
from djmoney.money import Money
from PIL import Image
import common.models
import InvenTree
import InvenTree.helpers_model
import InvenTree.version
from common.notifications import InvenTreeNotificationBodies, NotificationBody, trigger_notification
from InvenTree.format import format_money
logger = logging.getLogger('inventree')

def getSetting(key, backup_value=None):
    if False:
        i = 10
        return i + 15
    'Shortcut for reading a setting value from the database.'
    return common.models.InvenTreeSetting.get_setting(key, backup_value=backup_value)

def construct_absolute_url(*arg, **kwargs):
    if False:
        i = 10
        return i + 15
    'Construct (or attempt to construct) an absolute URL from a relative URL.\n\n    This is useful when (for example) sending an email to a user with a link\n    to something in the InvenTree web framework.\n    A URL is constructed in the following order:\n    1. If settings.SITE_URL is set (e.g. in the Django settings), use that\n    2. If the InvenTree setting INVENTREE_BASE_URL is set, use that\n    3. Otherwise, use the current request URL (if available)\n    '
    relative_url = '/'.join(arg)
    site_url = getattr(settings, 'SITE_URL', None)
    if not site_url:
        try:
            site_url = common.models.InvenTreeSetting.get_setting('INVENTREE_BASE_URL', create=False, cache=False)
        except (ProgrammingError, OperationalError):
            pass
    if not site_url:
        request = kwargs.get('request', None)
        if request:
            site_url = request.build_absolute_uri('/')
    if not site_url:
        return relative_url
    return urljoin(site_url, relative_url)

def get_base_url(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Return the base URL for the InvenTree server'
    return construct_absolute_url('', **kwargs)

def download_image_from_url(remote_url, timeout=2.5):
    if False:
        return 10
    "Download an image file from a remote URL.\n\n    This is a potentially dangerous operation, so we must perform some checks:\n    - The remote URL is available\n    - The Content-Length is provided, and is not too large\n    - The file is a valid image file\n\n    Arguments:\n        remote_url: The remote URL to retrieve image\n        max_size: Maximum allowed image size (default = 1MB)\n        timeout: Connection timeout in seconds (default = 5)\n\n    Returns:\n        An in-memory PIL image file, if the download was successful\n\n    Raises:\n        requests.exceptions.ConnectionError: Connection could not be established\n        requests.exceptions.Timeout: Connection timed out\n        requests.exceptions.HTTPError: Server responded with invalid response code\n        ValueError: Server responded with invalid 'Content-Length' value\n        TypeError: Response is not a valid image\n    "
    validator = URLValidator()
    validator(remote_url)
    max_size = int(common.models.InvenTreeSetting.get_setting('INVENTREE_DOWNLOAD_IMAGE_MAX_SIZE')) * 1024 * 1024
    user_agent = common.models.InvenTreeSetting.get_setting('INVENTREE_DOWNLOAD_FROM_URL_USER_AGENT')
    if user_agent:
        headers = {'User-Agent': user_agent}
    else:
        headers = None
    try:
        response = requests.get(remote_url, timeout=timeout, allow_redirects=True, stream=True, headers=headers)
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise Exception(_('Connection error') + f': {str(exc)}')
    except requests.exceptions.Timeout as exc:
        raise exc
    except requests.exceptions.HTTPError:
        raise requests.exceptions.HTTPError(_('Server responded with invalid status code') + f': {response.status_code}')
    except Exception as exc:
        raise Exception(_('Exception occurred') + f': {str(exc)}')
    if response.status_code != 200:
        raise Exception(_('Server responded with invalid status code') + f': {response.status_code}')
    try:
        content_length = int(response.headers.get('Content-Length', 0))
    except ValueError:
        raise ValueError(_('Server responded with invalid Content-Length value'))
    if content_length > max_size:
        raise ValueError(_('Image size is too large'))
    file = io.BytesIO()
    dl_size = 0
    chunk_size = 64 * 1024
    for chunk in response.iter_content(chunk_size=chunk_size):
        dl_size += len(chunk)
        if dl_size > max_size:
            raise ValueError(_('Image download exceeded maximum size'))
        file.write(chunk)
    if dl_size == 0:
        raise ValueError(_('Remote server returned empty response'))
    try:
        img = Image.open(file).convert()
        img.verify()
    except Exception:
        raise TypeError(_('Supplied URL is not a valid image file'))
    return img

def render_currency(money, decimal_places=None, currency=None, min_decimal_places=None, max_decimal_places=None):
    if False:
        while True:
            i = 10
    'Render a currency / Money object to a formatted string (e.g. for reports)\n\n    Arguments:\n        money: The Money instance to be rendered\n        decimal_places: The number of decimal places to render to. If unspecified, uses the PRICING_DECIMAL_PLACES setting.\n        currency: Optionally convert to the specified currency\n        min_decimal_places: The minimum number of decimal places to render to. If unspecified, uses the PRICING_DECIMAL_PLACES_MIN setting.\n        max_decimal_places: The maximum number of decimal places to render to. If unspecified, uses the PRICING_DECIMAL_PLACES setting.\n    '
    if money in [None, '']:
        return '-'
    if type(money) is not Money:
        return '-'
    if currency is not None:
        try:
            money = convert_money(money, currency)
        except Exception:
            pass
    if decimal_places is None:
        decimal_places = common.models.InvenTreeSetting.get_setting('PRICING_DECIMAL_PLACES', 6)
    if min_decimal_places is None:
        min_decimal_places = common.models.InvenTreeSetting.get_setting('PRICING_DECIMAL_PLACES_MIN', 0)
    if max_decimal_places is None:
        max_decimal_places = common.models.InvenTreeSetting.get_setting('PRICING_DECIMAL_PLACES', 6)
    value = Decimal(str(money.amount)).normalize()
    value = str(value)
    if '.' in value:
        decimals = len(value.split('.')[-1])
        decimals = max(decimals, min_decimal_places)
        decimals = min(decimals, decimal_places)
        decimal_places = decimals
    else:
        decimal_places = max(decimal_places, 2)
    decimal_places = max(decimal_places, max_decimal_places)
    return format_money(money, decimal_places=decimal_places)

def getModelsWithMixin(mixin_class) -> list:
    if False:
        for i in range(10):
            print('nop')
    'Return a list of models that inherit from the given mixin class.\n\n    Args:\n        mixin_class: The mixin class to search for\n    Returns:\n        List of models that inherit from the given mixin class\n    '
    from django.contrib.contenttypes.models import ContentType
    db_models = [x.model_class() for x in ContentType.objects.all() if x is not None]
    return [x for x in db_models if x is not None and issubclass(x, mixin_class)]

def notify_responsible(instance, sender, content: NotificationBody=InvenTreeNotificationBodies.NewOrder, exclude=None):
    if False:
        return 10
    'Notify all responsible parties of a change in an instance.\n\n    Parses the supplied content with the provided instance and sender and sends a notification to all responsible users,\n    excluding the optional excluded list.\n\n    Args:\n        instance: The newly created instance\n        sender: Sender model reference\n        content (NotificationBody, optional): _description_. Defaults to InvenTreeNotificationBodies.NewOrder.\n        exclude (User, optional): User instance that should be excluded. Defaults to None.\n    '
    notify_users([instance.responsible], instance, sender, content=content, exclude=exclude)

def notify_users(users, instance, sender, content: NotificationBody=InvenTreeNotificationBodies.NewOrder, exclude=None):
    if False:
        for i in range(10):
            print('nop')
    'Notify all passed users or groups.\n\n    Parses the supplied content with the provided instance and sender and sends a notification to all users,\n    excluding the optional excluded list.\n\n    Args:\n        users: List of users or groups to notify\n        instance: The newly created instance\n        sender: Sender model reference\n        content (NotificationBody, optional): _description_. Defaults to InvenTreeNotificationBodies.NewOrder.\n        exclude (User, optional): User instance that should be excluded. Defaults to None.\n    '
    content_context = {'instance': str(instance), 'verbose_name': sender._meta.verbose_name, 'app_label': sender._meta.app_label, 'model_name': sender._meta.model_name}
    context = {'instance': instance, 'name': content.name.format(**content_context), 'message': content.message.format(**content_context), 'link': InvenTree.helpers_model.construct_absolute_url(instance.get_absolute_url()), 'template': {'subject': content.name.format(**content_context)}}
    if content.template:
        context['template']['html'] = content.template.format(**content_context)
    trigger_notification(instance, content.slug.format(**content_context), targets=users, target_exclude=[exclude], context=context)