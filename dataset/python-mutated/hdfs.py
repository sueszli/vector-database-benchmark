from __future__ import annotations
from airflow.sensors.base import BaseSensorOperator
_EXCEPTION_MESSAGE = 'The old HDFS Sensors have been removed in 4.0.0 version of the apache.hdfs provider.\nPlease convert your DAGs to use the WebHdfsSensor or downgrade the provider to below 4.*\nif you want to continue using it.\nIf you want to use earlier provider you can downgrade to latest released 3.* version\nusing `pip install apache-airflow-providers-apache-hdfs==3.2.1` (no constraints)\n'

class HdfsSensor(BaseSensorOperator):
    """
    This Sensor has been removed and is not functional.

    Please convert your DAGs to use the WebHdfsSensor or downgrade the provider
    to below 4.* if you want to continue using it. If you want to use earlier
    provider you can downgrade to latest released 3.* version using
    `pip install apache-airflow-providers-apache-hdfs==3.2.1` (no constraints).
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise Exception(_EXCEPTION_MESSAGE)

class HdfsRegexSensor(HdfsSensor):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise Exception(_EXCEPTION_MESSAGE)

class HdfsFolderSensor(HdfsSensor):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        raise Exception(_EXCEPTION_MESSAGE)