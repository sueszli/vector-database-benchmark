# coding: utf-8

"""
    flyteidl/service/admin.proto

    No description provided (generated by Swagger Codegen https://github.com/swagger-api/swagger-codegen)  # noqa: E501

    OpenAPI spec version: version not set
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from flyteadmin.models.admin_description import AdminDescription  # noqa: F401,E501
from flyteadmin.models.admin_source_code import AdminSourceCode  # noqa: F401,E501
from flyteadmin.models.core_identifier import CoreIdentifier  # noqa: F401,E501


class AdminDescriptionEntity(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'id': 'CoreIdentifier',
        'short_description': 'str',
        'long_description': 'AdminDescription',
        'source_code': 'AdminSourceCode',
        'tags': 'list[str]'
    }

    attribute_map = {
        'id': 'id',
        'short_description': 'short_description',
        'long_description': 'long_description',
        'source_code': 'source_code',
        'tags': 'tags'
    }

    def __init__(self, id=None, short_description=None, long_description=None, source_code=None, tags=None):  # noqa: E501
        """AdminDescriptionEntity - a model defined in Swagger"""  # noqa: E501

        self._id = None
        self._short_description = None
        self._long_description = None
        self._source_code = None
        self._tags = None
        self.discriminator = None

        if id is not None:
            self.id = id
        if short_description is not None:
            self.short_description = short_description
        if long_description is not None:
            self.long_description = long_description
        if source_code is not None:
            self.source_code = source_code
        if tags is not None:
            self.tags = tags

    @property
    def id(self):
        """Gets the id of this AdminDescriptionEntity.  # noqa: E501

        id represents the unique identifier of the description entity.  # noqa: E501

        :return: The id of this AdminDescriptionEntity.  # noqa: E501
        :rtype: CoreIdentifier
        """
        return self._id

    @id.setter
    def id(self, id):
        """Sets the id of this AdminDescriptionEntity.

        id represents the unique identifier of the description entity.  # noqa: E501

        :param id: The id of this AdminDescriptionEntity.  # noqa: E501
        :type: CoreIdentifier
        """

        self._id = id

    @property
    def short_description(self):
        """Gets the short_description of this AdminDescriptionEntity.  # noqa: E501

        One-liner overview of the entity.  # noqa: E501

        :return: The short_description of this AdminDescriptionEntity.  # noqa: E501
        :rtype: str
        """
        return self._short_description

    @short_description.setter
    def short_description(self, short_description):
        """Sets the short_description of this AdminDescriptionEntity.

        One-liner overview of the entity.  # noqa: E501

        :param short_description: The short_description of this AdminDescriptionEntity.  # noqa: E501
        :type: str
        """

        self._short_description = short_description

    @property
    def long_description(self):
        """Gets the long_description of this AdminDescriptionEntity.  # noqa: E501

        Full user description with formatting preserved.  # noqa: E501

        :return: The long_description of this AdminDescriptionEntity.  # noqa: E501
        :rtype: AdminDescription
        """
        return self._long_description

    @long_description.setter
    def long_description(self, long_description):
        """Sets the long_description of this AdminDescriptionEntity.

        Full user description with formatting preserved.  # noqa: E501

        :param long_description: The long_description of this AdminDescriptionEntity.  # noqa: E501
        :type: AdminDescription
        """

        self._long_description = long_description

    @property
    def source_code(self):
        """Gets the source_code of this AdminDescriptionEntity.  # noqa: E501

        Optional link to source code used to define this entity.  # noqa: E501

        :return: The source_code of this AdminDescriptionEntity.  # noqa: E501
        :rtype: AdminSourceCode
        """
        return self._source_code

    @source_code.setter
    def source_code(self, source_code):
        """Sets the source_code of this AdminDescriptionEntity.

        Optional link to source code used to define this entity.  # noqa: E501

        :param source_code: The source_code of this AdminDescriptionEntity.  # noqa: E501
        :type: AdminSourceCode
        """

        self._source_code = source_code

    @property
    def tags(self):
        """Gets the tags of this AdminDescriptionEntity.  # noqa: E501

        User-specified tags. These are arbitrary and can be used for searching filtering and discovering tasks.  # noqa: E501

        :return: The tags of this AdminDescriptionEntity.  # noqa: E501
        :rtype: list[str]
        """
        return self._tags

    @tags.setter
    def tags(self, tags):
        """Sets the tags of this AdminDescriptionEntity.

        User-specified tags. These are arbitrary and can be used for searching filtering and discovering tasks.  # noqa: E501

        :param tags: The tags of this AdminDescriptionEntity.  # noqa: E501
        :type: list[str]
        """

        self._tags = tags

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(AdminDescriptionEntity, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, AdminDescriptionEntity):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
