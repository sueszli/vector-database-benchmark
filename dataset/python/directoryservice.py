# Copyright (c) 2012-2022, Mark Peek <mark@peek.org>
# All rights reserved.
#
# See LICENSE file for full license.
#
# *** Do not modify - this file is autogenerated ***


from . import AWSObject, AWSProperty, PropsDictType
from .validators import boolean


class VpcSettings(AWSProperty):
    """
    `VpcSettings <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-directoryservice-simplead-vpcsettings.html>`__
    """

    props: PropsDictType = {
        "SubnetIds": ([str], True),
        "VpcId": (str, True),
    }


class MicrosoftAD(AWSObject):
    """
    `MicrosoftAD <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-directoryservice-microsoftad.html>`__
    """

    resource_type = "AWS::DirectoryService::MicrosoftAD"

    props: PropsDictType = {
        "CreateAlias": (boolean, False),
        "Edition": (str, False),
        "EnableSso": (boolean, False),
        "Name": (str, True),
        "Password": (str, True),
        "ShortName": (str, False),
        "VpcSettings": (VpcSettings, True),
    }


class SimpleAD(AWSObject):
    """
    `SimpleAD <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-directoryservice-simplead.html>`__
    """

    resource_type = "AWS::DirectoryService::SimpleAD"

    props: PropsDictType = {
        "CreateAlias": (boolean, False),
        "Description": (str, False),
        "EnableSso": (boolean, False),
        "Name": (str, True),
        "Password": (str, False),
        "ShortName": (str, False),
        "Size": (str, True),
        "VpcSettings": (VpcSettings, True),
    }
