from . import attachment
from . import URLBase
from .AppriseAsset import AppriseAsset
from .logger import logger
from .common import ContentLocation
from .common import CONTENT_LOCATIONS
from .common import ATTACHMENT_SCHEMA_MAP
from .utils import GET_SCHEMA_RE

class AppriseAttachment:
    """
    Our Apprise Attachment File Manager

    """

    def __init__(self, paths=None, asset=None, cache=True, location=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Loads all of the paths/urls specified (if any).\n\n        The path can either be a single string identifying one explicit\n        location, otherwise you can pass in a series of locations to scan\n        via a list.\n\n        By default we cache our responses so that subsiquent calls does not\n        cause the content to be retrieved again.  For local file references\n        this makes no difference at all.  But for remote content, this does\n        mean more then one call can be made to retrieve the (same) data.  This\n        method can be somewhat inefficient if disabled.  Only disable caching\n        if you understand the consequences.\n\n        You can alternatively set the cache value to an int identifying the\n        number of seconds the previously retrieved can exist for before it\n        should be considered expired.\n\n        It's also worth nothing that the cache value is only set to elements\n        that are not already of subclass AttachBase()\n\n        Optionally set your current ContentLocation in the location argument.\n        This is used to further handle attachments. The rules are as follows:\n          - INACCESSIBLE: You simply have disabled use of the object; no\n                          attachments will be retrieved/handled.\n          - HOSTED:       You are hosting an attachment service for others.\n                          In these circumstances all attachments that are LOCAL\n                          based (such as file://) will not be allowed.\n          - LOCAL:        The least restrictive mode as local files can be\n                          referenced in addition to hosted.\n\n        In all both HOSTED and LOCAL modes, INACCESSIBLE attachment types will\n        continue to be inaccessible.  However if you set this field (location)\n        to None (it's default value) the attachment location category will not\n        be tested in any way (all attachment types will be allowed).\n\n        The location field is also a global option that can be set when\n        initializing the Apprise object.\n\n        "
        self.attachments = list()
        self.cache = cache
        self.asset = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        if location is not None and location not in CONTENT_LOCATIONS:
            msg = 'An invalid Attachment location ({}) was specified.'.format(location)
            logger.warning(msg)
            raise TypeError(msg)
        self.location = location
        if paths is not None:
            if not self.add(paths):
                raise TypeError('One or more attachments could not be added.')

    def add(self, attachments, asset=None, cache=None):
        if False:
            while True:
                i = 10
        "\n        Adds one or more attachments into our list.\n\n        By default we cache our responses so that subsiquent calls does not\n        cause the content to be retrieved again.  For local file references\n        this makes no difference at all.  But for remote content, this does\n        mean more then one call can be made to retrieve the (same) data.  This\n        method can be somewhat inefficient if disabled.  Only disable caching\n        if you understand the consequences.\n\n        You can alternatively set the cache value to an int identifying the\n        number of seconds the previously retrieved can exist for before it\n        should be considered expired.\n\n        It's also worth nothing that the cache value is only set to elements\n        that are not already of subclass AttachBase()\n        "
        return_status = True
        cache = cache if cache is not None else self.cache
        if asset is None:
            asset = self.asset
        if isinstance(attachments, attachment.AttachBase):
            self.attachments.append(attachments)
            return True
        elif isinstance(attachments, str):
            attachments = (attachments,)
        elif not isinstance(attachments, (tuple, set, list)):
            logger.error('An invalid attachment url (type={}) was specified.'.format(type(attachments)))
            return False
        for _attachment in attachments:
            if self.location == ContentLocation.INACCESSIBLE:
                logger.warning('Attachments are disabled; ignoring {}'.format(_attachment))
                return_status = False
                continue
            if isinstance(_attachment, str):
                logger.debug('Loading attachment: {}'.format(_attachment))
                instance = AppriseAttachment.instantiate(_attachment, asset=asset, cache=cache)
                if not isinstance(instance, attachment.AttachBase):
                    return_status = False
                    continue
            elif isinstance(_attachment, AppriseAttachment):
                instance = _attachment.attachments
            elif not isinstance(_attachment, attachment.AttachBase):
                logger.warning('An invalid attachment (type={}) was specified.'.format(type(_attachment)))
                return_status = False
                continue
            else:
                instance = _attachment
            if self.location and (self.location == ContentLocation.HOSTED and instance.location != ContentLocation.HOSTED or instance.location == ContentLocation.INACCESSIBLE):
                logger.warning('Attachment was disallowed due to accessibility restrictions ({}->{}): {}'.format(self.location, instance.location, instance.url(privacy=True)))
                return_status = False
                continue
            if isinstance(instance, list):
                self.attachments.extend(instance)
            else:
                self.attachments.append(instance)
        return return_status

    @staticmethod
    def instantiate(url, asset=None, cache=None, suppress_exceptions=True):
        if False:
            while True:
                i = 10
        '\n        Returns the instance of a instantiated attachment plugin based on\n        the provided Attachment URL.  If the url fails to be parsed, then None\n        is returned.\n\n        A specified cache value will over-ride anything set\n\n        '
        schema = GET_SCHEMA_RE.match(url)
        if schema is None:
            schema = attachment.AttachFile.protocol
            url = '{}://{}'.format(schema, URLBase.quote(url))
        else:
            schema = schema.group('schema').lower()
            if schema not in ATTACHMENT_SCHEMA_MAP:
                logger.warning('Unsupported schema {}.'.format(schema))
                return None
        results = ATTACHMENT_SCHEMA_MAP[schema].parse_url(url)
        if not results:
            logger.warning('Unparseable URL {}.'.format(url))
            return None
        results['asset'] = asset if isinstance(asset, AppriseAsset) else AppriseAsset()
        if cache is not None:
            results['cache'] = cache
        if suppress_exceptions:
            try:
                attach_plugin = ATTACHMENT_SCHEMA_MAP[results['schema']](**results)
            except Exception:
                logger.warning('Could not load URL: %s' % url)
                return None
        else:
            attach_plugin = ATTACHMENT_SCHEMA_MAP[results['schema']](**results)
        return attach_plugin

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Empties our attachment list\n\n        '
        self.attachments[:] = []

    def size(self):
        if False:
            print('Hello World!')
        '\n        Returns the total size of accumulated attachments\n        '
        return sum([len(a) for a in self.attachments if len(a) > 0])

    def pop(self, index=-1):
        if False:
            i = 10
            return i + 15
        '\n        Removes an indexed Apprise Attachment from the stack and returns it.\n\n        by default the last element is poped from the list\n        '
        return self.attachments.pop(index)

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        '\n        Returns the indexed entry of a loaded apprise attachments\n        '
        return self.attachments[index]

    def __bool__(self):
        if False:
            while True:
                i = 10
        "\n        Allows the Apprise object to be wrapped in an 'if statement'.\n        True is returned if at least one service has been loaded.\n        "
        return True if self.attachments else False

    def __iter__(self):
        if False:
            print('Hello World!')
        '\n        Returns an iterator to our attachment list\n        '
        return iter(self.attachments)

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the number of attachment entries loaded\n        '
        return len(self.attachments)