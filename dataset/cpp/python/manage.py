#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @COPYRIGHT_begin
#
# Copyright [2015] Michał Szczygieł (m4gik), M4GiK Software
#
# @COPYRIGHT_end

import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "django_dependencies.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
