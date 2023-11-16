from metaflow.exception import MetaflowTaggingError
from metaflow.util import unicode_type, bytes_type

def is_utf8_encodable(x):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns true if the object can be encoded with UTF-8\n    '
    try:
        x.encode('utf-8')
        return True
    except UnicodeError:
        return False

def is_utf8_decodable(x):
    if False:
        i = 10
        return i + 15
    '\n    Returns true if the object can be decoded with UTF-8\n    '
    try:
        x.decode('utf-8')
        return True
    except UnicodeError:
        return False
MAX_USER_TAG_SET_SIZE = 50
MAX_TAG_SIZE = 500

def validate_tags(tags, existing_tags=None):
    if False:
        print('Hello World!')
    "\n    Raises MetaflowTaggingError if invalid based on these rules:\n\n    Tag set size is too large. But it's OK if tag set is not larger\n    than an existing tag set (if provided).\n\n    Then, we validate each tag.  See validate_tag()\n    "
    tag_set = frozenset(tags)
    if len(tag_set) > MAX_USER_TAG_SET_SIZE:
        if existing_tags is None or len(frozenset(existing_tags)) < len(tag_set):
            raise MetaflowTaggingError(msg='Cannot increase size of tag set beyond %d' % (MAX_USER_TAG_SET_SIZE,))
    for tag in tag_set:
        validate_tag(tag)

def validate_tag(tag):
    if False:
        print('Hello World!')
    '\n    - Tag must be either of bytes-type or unicode-type.\n    - If tag is of bytes-type, it must be UTF-8 decodable\n    - If tag is of unicode-type, it must be UTF-8 encodable\n    - Tag may not be empty string.\n    - Tag cannot be too long (500 chars)\n    '
    if isinstance(tag, bytes_type):
        if not is_utf8_decodable(tag):
            raise MetaflowTaggingError('Tags must be UTF-8 decodable')
    elif isinstance(tag, unicode_type):
        if not is_utf8_encodable(tag):
            raise MetaflowTaggingError('Tags must be UTF-8 encodable')
    else:
        raise MetaflowTaggingError('Tags must be some kind of string (bytes or unicode), got %s', str(type(tag)))
    if not len(tag):
        raise MetaflowTaggingError('Tags must not be empty string')
    if len(tag) > MAX_TAG_SIZE:
        raise MetaflowTaggingError('Tag is too long %d > %d' % (len(tag), MAX_TAG_SIZE))