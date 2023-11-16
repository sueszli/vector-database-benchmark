"""Unit tests for scheduled tasks."""
from django.test import TestCase
from plugin import InvenTreePlugin, registry
from plugin.helpers import MixinImplementationError
from plugin.mixins import ScheduleMixin
from plugin.registry import call_function

class ExampleScheduledTaskPluginTests(TestCase):
    """Tests for provided ScheduledTaskPlugin."""

    def test_function(self):
        if False:
            for i in range(10):
                print('nop')
        'Check if the scheduling works.'
        self.assertIn('schedule', registry.plugins)
        plg = registry.plugins['schedule']
        self.assertTrue(plg)
        self.assertEqual(plg.member_func(), False)
        self.assertEqual(plg.get_task_names(), ['plugin.schedule.member', 'plugin.schedule.hello', 'plugin.schedule.world'])
        plg.register_tasks()
        from django_q.models import Schedule
        scheduled_plugin_tasks = Schedule.objects.filter(name__istartswith='plugin.')
        self.assertEqual(len(scheduled_plugin_tasks), 3)
        hello_schedule = Schedule.objects.get(name='plugin.schedule.hello')
        self.assertEqual(hello_schedule.minutes, 45)
        plg.scheduled_tasks['hello']['minutes'] = 15
        plg.register_tasks()
        hello_schedule = Schedule.objects.get(name='plugin.schedule.hello')
        scheduled_plugin_tasks = Schedule.objects.filter(name__istartswith='plugin.')
        self.assertEqual(hello_schedule.minutes, 15)
        self.assertEqual(len(scheduled_plugin_tasks), 3)
        scheduled_plugin_tasks[1].delete()
        scheduled_plugin_tasks = Schedule.objects.filter(name__istartswith='plugin.')
        self.assertEqual(len(scheduled_plugin_tasks), 2)
        plg.unregister_tasks()
        scheduled_plugin_tasks = Schedule.objects.filter(name__istartswith='plugin.')
        self.assertEqual(len(scheduled_plugin_tasks), 0)

    def test_calling(self):
        if False:
            i = 10
            return i + 15
        'Check if a function can be called without errors.'
        self.assertEqual(call_function('schedule', 'member_func'), False)
        self.assertEqual(call_function('does_not_exsist', 'member_func'), None)

class ScheduledTaskPluginTests(TestCase):
    """Tests for ScheduledTaskPluginTests mixin base."""

    def test_init(self):
        if False:
            return 10
        'Check that all MixinImplementationErrors raise.'

        class Base(ScheduleMixin, InvenTreePlugin):
            NAME = 'APlugin'

        class NoSchedules(Base):
            """Plugin without schedules."""
            pass
        with self.assertRaises(MixinImplementationError):
            NoSchedules()

        class WrongFuncSchedules(Base):
            """Plugin with broken functions.

            This plugin is missing a func
            """
            SCHEDULED_TASKS = {'test': {'schedule': 'I', 'minutes': 30}}

            def test(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        with self.assertRaises(MixinImplementationError):
            WrongFuncSchedules()

        class WrongFuncSchedules1(WrongFuncSchedules):
            """Plugin with broken functions.

            This plugin is missing a schedule
            """
            SCHEDULED_TASKS = {'test': {'func': 'test', 'minutes': 30}}
        with self.assertRaises(MixinImplementationError):
            WrongFuncSchedules1()

        class WrongFuncSchedules2(WrongFuncSchedules):
            """Plugin with broken functions.

            This plugin is missing a schedule
            """
            SCHEDULED_TASKS = {'test': {'func': 'test', 'minutes': 30}}
        with self.assertRaises(MixinImplementationError):
            WrongFuncSchedules2()

        class WrongFuncSchedules3(WrongFuncSchedules):
            """Plugin with broken functions.

            This plugin has a broken schedule
            """
            SCHEDULED_TASKS = {'test': {'func': 'test', 'schedule': 'XX', 'minutes': 30}}
        with self.assertRaises(MixinImplementationError):
            WrongFuncSchedules3()

        class WrongFuncSchedules4(WrongFuncSchedules):
            """Plugin with broken functions.

            This plugin is missing a minute marker for its schedule
            """
            SCHEDULED_TASKS = {'test': {'func': 'test', 'schedule': 'I'}}
        with self.assertRaises(MixinImplementationError):
            WrongFuncSchedules4()