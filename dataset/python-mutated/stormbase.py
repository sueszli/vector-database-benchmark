from __future__ import absolute_import
import abc
import datetime
import bson
import six
import mongoengine as me
from oslo_config import cfg
from st2common.util import mongoescape
from st2common.models.base import DictSerializableClassMixin
from st2common.models.system.common import ResourceReference
from st2common.constants.types import ResourceType
from st2common.util.jsonify import json_decode
__all__ = ['StormFoundationDB', 'StormBaseDB', 'EscapedDictField', 'EscapedDynamicField', 'TagField', 'RefFieldMixin', 'UIDFieldMixin', 'TagsMixin', 'ContentPackResourceMixin']
JSON_UNFRIENDLY_TYPES = (datetime.datetime, bson.ObjectId)
DICT_FIELD_NOT_SET_MARKER = 'dict-field-not-set'

class StormFoundationDB(me.Document, DictSerializableClassMixin):
    """
    Base abstraction for a model entity. This foundation class should only be directly
    inherited from the application domain models.
    """
    RESOURCE_TYPE = ResourceType.UNKNOWN
    objects = me.queryset.QuerySetManager()
    meta = {'abstract': True}

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        attrs = list()
        for k in sorted(self._fields.keys()):
            v = getattr(self, k)
            v = '"%s"' % str(v) if type(v) in [str, six.text_type, datetime.datetime] else str(v)
            attrs.append('%s=%s' % (k, v))
        return '%s(%s)' % (self.__class__.__name__, ', '.join(attrs))

    def get_resource_type(self):
        if False:
            for i in range(10):
                print('nop')
        return self.RESOURCE_TYPE

    def mask_secrets(self, value):
        if False:
            print('Hello World!')
        '\n        Process the model dictionary and mask secret values.\n\n        :type value: ``dict``\n        :param value: Document dictionary.\n\n        :rtype: ``dict``\n        '
        return value

    def to_serializable_dict(self, mask_secrets=False):
        if False:
            print('Hello World!')
        '\n        Serialize database model to a dictionary.\n\n        :param mask_secrets: True to mask secrets in the resulting dict.\n        :type mask_secrets: ``boolean``\n\n        :rtype: ``dict``\n        '
        serializable_dict = {}
        for k in sorted(six.iterkeys(self._fields)):
            v = getattr(self, k)
            if isinstance(v, JSON_UNFRIENDLY_TYPES):
                v = str(v)
            elif isinstance(v, me.EmbeddedDocument):
                v = json_decode(v.to_json())
            serializable_dict[k] = v
        if mask_secrets and cfg.CONF.log.mask_secrets:
            serializable_dict = self.mask_secrets(value=serializable_dict)
        return serializable_dict

class StormBaseDB(StormFoundationDB):
    """Abstraction for a user content model."""
    name = me.StringField(required=True, unique=True)
    description = me.StringField()
    meta = {'abstract': True}

class EscapedDictField(me.DictField):

    def to_mongo(self, value, use_db_field=True, fields=None):
        if False:
            print('Hello World!')
        value = mongoescape.escape_chars(value)
        return super(EscapedDictField, self).to_mongo(value=value, use_db_field=use_db_field, fields=fields)

    def to_python(self, value):
        if False:
            i = 10
            return i + 15
        value = super(EscapedDictField, self).to_python(value)
        return mongoescape.unescape_chars(value)

    def validate(self, value):
        if False:
            print('Hello World!')
        if not isinstance(value, dict):
            self.error('Only dictionaries may be used in a DictField')
        if me.fields.key_not_string(value):
            self.error('Invalid dictionary key - documents must have only string keys')
        me.base.ComplexBaseField.validate(self, value)

class EscapedDynamicField(me.DynamicField):

    def to_mongo(self, value, use_db_field=True, fields=None):
        if False:
            while True:
                i = 10
        value = mongoescape.escape_chars(value)
        return super(EscapedDynamicField, self).to_mongo(value=value, use_db_field=use_db_field, fields=fields)

    def to_python(self, value):
        if False:
            i = 10
            return i + 15
        value = super(EscapedDynamicField, self).to_python(value)
        return mongoescape.unescape_chars(value)

class TagField(me.EmbeddedDocument):
    """
    To be attached to a db model object for the purpose of providing supplemental
    information.
    """
    name = me.StringField(max_length=1024)
    value = me.StringField(max_length=1024)

class TagsMixin(object):
    """
    Mixin to include tags on an object.
    """
    tags = me.ListField(field=me.EmbeddedDocumentField(TagField))

    @classmethod
    def get_indexes(cls):
        if False:
            return 10
        return ['tags.name', 'tags.value']

class RefFieldMixin(object):
    """
    Mixin class which adds "ref" field to the class inheriting from it.
    """
    ref = me.StringField(required=True, unique=True)

class UIDFieldMixin(object):
    """
    Mixin class which adds "uid" field to the class inheriting from it.

    UID field is a unique identifier which we can be used to unambiguously reference a resource in
    the system.
    """
    UID_SEPARATOR = ':'
    RESOURCE_TYPE = abc.abstractproperty
    UID_FIELDS = abc.abstractproperty
    uid = me.StringField(required=True)

    @classmethod
    def get_indexes(cls):
        if False:
            return 10
        indexes = [{'fields': ['uid'], 'unique': True, 'sparse': True}]
        return indexes

    def get_uid(self):
        if False:
            i = 10
            return i + 15
        '\n        Return an object UID constructed from the object properties / fields.\n\n        :rtype: ``str``\n        '
        parts = []
        parts.append(self.RESOURCE_TYPE)
        for field in self.UID_FIELDS:
            value = getattr(self, field, None) or ''
            parts.append(value)
        uid = self.UID_SEPARATOR.join(parts)
        return uid

    def get_uid_parts(self):
        if False:
            return 10
        '\n        Return values for fields which make up the UID.\n\n        :rtype: ``list``\n        '
        parts = self.uid.split(self.UID_SEPARATOR)
        parts = [part for part in parts if part.strip()]
        return parts

    def has_valid_uid(self):
        if False:
            i = 10
            return i + 15
        '\n        Return True if object contains a valid id (aka all parts contain a valid value).\n\n        :rtype: ``bool``\n        '
        parts = self.get_uid_parts()
        return len(parts) == len(self.UID_FIELDS) + 1

class ContentPackResourceMixin(object):
    """
    Mixin class provides utility methods for models which belong to a pack.
    """
    metadata_file = me.StringField(required=False, help_text='Path to the metadata file (file on disk which contains resource definition) relative to the pack directory.')

    def get_pack_uid(self):
        if False:
            print('Hello World!')
        '\n        Return an UID of a pack this resource belongs to.\n\n        :rtype ``str``\n        '
        parts = [ResourceType.PACK, self.pack]
        uid = UIDFieldMixin.UID_SEPARATOR.join(parts)
        return uid

    def get_reference(self):
        if False:
            return 10
        '\n        Retrieve referene object for this model.\n\n        :rtype: :class:`ResourceReference`\n        '
        if getattr(self, 'ref', None):
            ref = ResourceReference.from_string_reference(ref=self.ref)
        else:
            ref = ResourceReference(pack=self.pack, name=self.name)
        return ref

    @classmethod
    def get_indexes(cls):
        if False:
            for i in range(10):
                print('nop')
        return [{'fields': ['metadata_file']}]

class ChangeRevisionFieldMixin(object):
    rev = me.IntField(required=True, default=1)

    @classmethod
    def get_indexes(cls):
        if False:
            while True:
                i = 10
        return [{'fields': ['id', 'rev'], 'unique': True}]