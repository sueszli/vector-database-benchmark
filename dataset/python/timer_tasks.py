# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

from __future__ import absolute_import
import os
import time

from celery import Celery
from django.conf import settings
from sqlalchemy.orm import create_session

from kams_erp.models.kqs_products import KqsProdukty
from kams_erp.utils.database_connector import DatabaseConnector
from kams_erp.utils.xml_rpc_connector import XmlRpcConnector
from kams_erp.timer_tasks.settings import BROKER_URL, CELERY_RESULT_BACKEND





# set the default Django settings module for the 'celery' program.

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_dependencies.settings')

app = Celery('django_dependencies', broker=BROKER_URL, backend=CELERY_RESULT_BACKEND,
             include=['django_dependencies'])

# Using a string here means the worker will not have to
# pickle the object when using Windows.
app.config_from_object('django.conf:settings')
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)


# Optional configuration, see the application user guide.
# app.conf.update(
#     CELERY_TASK_RESULT_EXPIRES=3600,
# )

@app.task(trail=True, name='django_dependencies.timer_tasks.fetch_orders')
def fetch_orders():
    """
    Method to fetch information about orders from KQS and insert to Odoo.
    :return: Fetched data.
    """
    connector = XmlRpcConnector()

    partner_record = [{
        'name': 'Fabien2',
        'email': 'example@odoo.com'
    }]

    result = connector.create('product.template', partner_record)
    print "Michal proo skill" + str(result)
    return time.time()


@app.task(trail=True, name='django_dependencies.timer_tasks.fetch_products')
def fetch_products():
    """
    Method to fetch information about orders from KQS and insert to Odoo.
    :return: Fetched data.
    """

    # Create a session to use the tables
    session = create_session(bind=DatabaseConnector().get_engine())
    query = session.query(KqsProdukty)
    result = query.all()

    # for record in result:
    #     if not self.read('product.template', [[['name', '=', record.nazwa]]]):
    #         print record.nazwa

    session.close()

