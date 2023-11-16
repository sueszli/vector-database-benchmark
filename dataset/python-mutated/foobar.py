""" Using absolute import, do from module imports.

"""
from __future__ import absolute_import, print_function
from foobar import util
from . import local

class Foobar(object):

    def __init__(self):
        if False:
            return 10
        print(util.someFunction())