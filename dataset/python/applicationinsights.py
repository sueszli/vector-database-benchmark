# Copyright (c) 2012-2022, Mark Peek <mark@peek.org>
# All rights reserved.
#
# See LICENSE file for full license.
#
# *** Do not modify - this file is autogenerated ***


from . import AWSObject, AWSProperty, PropsDictType, Tags
from .validators import boolean, integer


class Alarm(AWSProperty):
    """
    `Alarm <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-alarm.html>`__
    """

    props: PropsDictType = {
        "AlarmName": (str, True),
        "Severity": (str, False),
    }


class AlarmMetric(AWSProperty):
    """
    `AlarmMetric <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-alarmmetric.html>`__
    """

    props: PropsDictType = {
        "AlarmMetricName": (str, True),
    }


class HAClusterPrometheusExporter(AWSProperty):
    """
    `HAClusterPrometheusExporter <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-haclusterprometheusexporter.html>`__
    """

    props: PropsDictType = {
        "PrometheusPort": (str, False),
    }


class HANAPrometheusExporter(AWSProperty):
    """
    `HANAPrometheusExporter <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-hanaprometheusexporter.html>`__
    """

    props: PropsDictType = {
        "AgreeToInstallHANADBClient": (boolean, True),
        "HANAPort": (str, True),
        "HANASID": (str, True),
        "HANASecretName": (str, True),
        "PrometheusPort": (str, False),
    }


class JMXPrometheusExporter(AWSProperty):
    """
    `JMXPrometheusExporter <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-jmxprometheusexporter.html>`__
    """

    props: PropsDictType = {
        "HostPort": (str, False),
        "JMXURL": (str, False),
        "PrometheusPort": (str, False),
    }


class Log(AWSProperty):
    """
    `Log <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-log.html>`__
    """

    props: PropsDictType = {
        "Encoding": (str, False),
        "LogGroupName": (str, False),
        "LogPath": (str, False),
        "LogType": (str, True),
        "PatternSet": (str, False),
    }


class WindowsEvent(AWSProperty):
    """
    `WindowsEvent <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-windowsevent.html>`__
    """

    props: PropsDictType = {
        "EventLevels": ([str], True),
        "EventName": (str, True),
        "LogGroupName": (str, True),
        "PatternSet": (str, False),
    }


class ConfigurationDetails(AWSProperty):
    """
    `ConfigurationDetails <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-configurationdetails.html>`__
    """

    props: PropsDictType = {
        "AlarmMetrics": ([AlarmMetric], False),
        "Alarms": ([Alarm], False),
        "HAClusterPrometheusExporter": (HAClusterPrometheusExporter, False),
        "HANAPrometheusExporter": (HANAPrometheusExporter, False),
        "JMXPrometheusExporter": (JMXPrometheusExporter, False),
        "Logs": ([Log], False),
        "WindowsEvents": ([WindowsEvent], False),
    }


class SubComponentConfigurationDetails(AWSProperty):
    """
    `SubComponentConfigurationDetails <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-subcomponentconfigurationdetails.html>`__
    """

    props: PropsDictType = {
        "AlarmMetrics": ([AlarmMetric], False),
        "Logs": ([Log], False),
        "WindowsEvents": ([WindowsEvent], False),
    }


class SubComponentTypeConfiguration(AWSProperty):
    """
    `SubComponentTypeConfiguration <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-subcomponenttypeconfiguration.html>`__
    """

    props: PropsDictType = {
        "SubComponentConfigurationDetails": (SubComponentConfigurationDetails, True),
        "SubComponentType": (str, True),
    }


class ComponentConfiguration(AWSProperty):
    """
    `ComponentConfiguration <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-componentconfiguration.html>`__
    """

    props: PropsDictType = {
        "ConfigurationDetails": (ConfigurationDetails, False),
        "SubComponentTypeConfigurations": ([SubComponentTypeConfiguration], False),
    }


class ComponentMonitoringSetting(AWSProperty):
    """
    `ComponentMonitoringSetting <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-componentmonitoringsetting.html>`__
    """

    props: PropsDictType = {
        "ComponentARN": (str, False),
        "ComponentConfigurationMode": (str, True),
        "ComponentName": (str, False),
        "CustomComponentConfiguration": (ComponentConfiguration, False),
        "DefaultOverwriteComponentConfiguration": (ComponentConfiguration, False),
        "Tier": (str, True),
    }


class CustomComponent(AWSProperty):
    """
    `CustomComponent <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-customcomponent.html>`__
    """

    props: PropsDictType = {
        "ComponentName": (str, True),
        "ResourceList": ([str], True),
    }


class LogPattern(AWSProperty):
    """
    `LogPattern <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-logpattern.html>`__
    """

    props: PropsDictType = {
        "Pattern": (str, True),
        "PatternName": (str, True),
        "Rank": (integer, True),
    }


class LogPatternSet(AWSProperty):
    """
    `LogPatternSet <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-properties-applicationinsights-application-logpatternset.html>`__
    """

    props: PropsDictType = {
        "LogPatterns": ([LogPattern], True),
        "PatternSetName": (str, True),
    }


class Application(AWSObject):
    """
    `Application <http://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-resource-applicationinsights-application.html>`__
    """

    resource_type = "AWS::ApplicationInsights::Application"

    props: PropsDictType = {
        "AutoConfigurationEnabled": (boolean, False),
        "CWEMonitorEnabled": (boolean, False),
        "ComponentMonitoringSettings": ([ComponentMonitoringSetting], False),
        "CustomComponents": ([CustomComponent], False),
        "GroupingType": (str, False),
        "LogPatternSets": ([LogPatternSet], False),
        "OpsCenterEnabled": (boolean, False),
        "OpsItemSNSTopicArn": (str, False),
        "ResourceGroupName": (str, True),
        "Tags": (Tags, False),
    }
