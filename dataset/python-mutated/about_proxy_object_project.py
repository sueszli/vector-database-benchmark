from runner.koan import *

class Proxy:

    def __init__(self, target_object):
        if False:
            i = 10
            return i + 15
        self._obj = target_object

class AboutProxyObjectProject(Koan):

    def test_proxy_method_returns_wrapped_object(self):
        if False:
            print('Hello World!')
        tv = Proxy(Television())
        self.assertTrue(isinstance(tv, Proxy))

    def test_tv_methods_still_perform_their_function(self):
        if False:
            i = 10
            return i + 15
        tv = Proxy(Television())
        tv.channel = 10
        tv.power()
        self.assertEqual(10, tv.channel)
        self.assertTrue(tv.is_on())

    def test_proxy_records_messages_sent_to_tv(self):
        if False:
            return 10
        tv = Proxy(Television())
        tv.power()
        tv.channel = 10
        self.assertEqual(['power', 'channel'], tv.messages())

    def test_proxy_handles_invalid_messages(self):
        if False:
            return 10
        tv = Proxy(Television())
        with self.assertRaises(AttributeError):
            tv.no_such_method()

    def test_proxy_reports_methods_have_been_called(self):
        if False:
            i = 10
            return i + 15
        tv = Proxy(Television())
        tv.power()
        tv.power()
        self.assertTrue(tv.was_called('power'))
        self.assertFalse(tv.was_called('channel'))

    def test_proxy_counts_method_calls(self):
        if False:
            while True:
                i = 10
        tv = Proxy(Television())
        tv.power()
        tv.channel = 48
        tv.power()
        self.assertEqual(2, tv.number_of_times_called('power'))
        self.assertEqual(1, tv.number_of_times_called('channel'))
        self.assertEqual(0, tv.number_of_times_called('is_on'))

    def test_proxy_can_record_more_than_just_tv_objects(self):
        if False:
            return 10
        proxy = Proxy('Py Ohio 2010')
        result = proxy.upper()
        self.assertEqual('PY OHIO 2010', result)
        result = proxy.split()
        self.assertEqual(['Py', 'Ohio', '2010'], result)
        self.assertEqual(['upper', 'split'], proxy.messages())

class Television:

    def __init__(self):
        if False:
            return 10
        self._channel = None
        self._power = None

    @property
    def channel(self):
        if False:
            while True:
                i = 10
        return self._channel

    @channel.setter
    def channel(self, value):
        if False:
            print('Hello World!')
        self._channel = value

    def power(self):
        if False:
            return 10
        if self._power == 'on':
            self._power = 'off'
        else:
            self._power = 'on'

    def is_on(self):
        if False:
            for i in range(10):
                print('nop')
        return self._power == 'on'

class TelevisionTest(Koan):

    def test_it_turns_on(self):
        if False:
            for i in range(10):
                print('nop')
        tv = Television()
        tv.power()
        self.assertTrue(tv.is_on())

    def test_it_also_turns_off(self):
        if False:
            print('Hello World!')
        tv = Television()
        tv.power()
        tv.power()
        self.assertFalse(tv.is_on())

    def test_edge_case_on_off(self):
        if False:
            for i in range(10):
                print('nop')
        tv = Television()
        tv.power()
        tv.power()
        tv.power()
        self.assertTrue(tv.is_on())
        tv.power()
        self.assertFalse(tv.is_on())

    def test_can_set_the_channel(self):
        if False:
            while True:
                i = 10
        tv = Television()
        tv.channel = 11
        self.assertEqual(11, tv.channel)