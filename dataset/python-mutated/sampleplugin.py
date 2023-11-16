from __future__ import absolute_import
import plugin.util.randomutil

class SamplePlugin(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__count = 10

    def do_work(self):
        if False:
            i = 10
            return i + 15
        return plugin.util.randomutil.get_random_numbers(self.__count)