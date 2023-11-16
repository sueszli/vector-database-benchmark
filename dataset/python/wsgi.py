# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

"""
WSGI config for django_dependencies project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/1.9/howto/deployment/wsgi/
"""

from django.core.wsgi import get_wsgi_application
import os
import sys

sys.path.append('/opt/odoo/my-modules')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kams_erp.models.settings")
application = get_wsgi_application()
