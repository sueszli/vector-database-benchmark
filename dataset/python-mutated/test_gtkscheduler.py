import os
import threading
import unittest
from datetime import timedelta
from time import sleep
import pytest
from reactivex.internal.basic import default_now
from reactivex.scheduler.mainloop import GtkScheduler
gi = pytest.importorskip('gi')
from gi.repository import GLib, Gtk
gi.require_version('Gtk', '3.0')
if 'GNOME_DESKTOP_SESSION_ID' in os.environ:
    del os.environ['GNOME_DESKTOP_SESSION_ID']

class TestGtkScheduler(unittest.TestCase):

    def test_gtk_schedule_now(self):
        if False:
            while True:
                i = 10
        scheduler = GtkScheduler(GLib)
        diff = scheduler.now - default_now()
        assert abs(diff) < timedelta(milliseconds=1)

    def test_gtk_schedule_now_units(self):
        if False:
            print('Hello World!')
        scheduler = GtkScheduler(GLib)
        diff = scheduler.now
        sleep(0.1)
        diff = scheduler.now - diff
        assert timedelta(milliseconds=80) < diff < timedelta(milliseconds=180)

    def test_gtk_schedule_action(self):
        if False:
            while True:
                i = 10
        scheduler = GtkScheduler(GLib)
        gate = threading.Semaphore(0)
        ran = False

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal ran
            ran = True
        scheduler.schedule(action)

        def done(data):
            if False:
                while True:
                    i = 10
            Gtk.main_quit()
            gate.release()
            return False
        GLib.timeout_add(50, done, None)
        Gtk.main()
        gate.acquire()
        assert ran is True

    def test_gtk_schedule_action_relative(self):
        if False:
            while True:
                i = 10
        scheduler = GtkScheduler(GLib)
        gate = threading.Semaphore(0)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                while True:
                    i = 10
            nonlocal endtime
            endtime = default_now()
        scheduler.schedule_relative(0.1, action)

        def done(data):
            if False:
                i = 10
                return i + 15
            Gtk.main_quit()
            gate.release()
            return False
        GLib.timeout_add(200, done, None)
        Gtk.main()
        gate.acquire()
        assert endtime is not None
        diff = endtime - starttime
        assert diff > timedelta(milliseconds=80)

    def test_gtk_schedule_action_absolute(self):
        if False:
            return 10
        scheduler = GtkScheduler(GLib)
        gate = threading.Semaphore(0)
        starttime = default_now()
        endtime = None

        def action(scheduler, state):
            if False:
                print('Hello World!')
            nonlocal endtime
            endtime = default_now()
        due = scheduler.now + timedelta(milliseconds=100)
        scheduler.schedule_absolute(due, action)

        def done(data):
            if False:
                for i in range(10):
                    print('nop')
            Gtk.main_quit()
            gate.release()
            return False
        GLib.timeout_add(200, done, None)
        Gtk.main()
        gate.acquire()
        assert endtime is not None
        diff = endtime - starttime
        assert diff > timedelta(milliseconds=80)

    def test_gtk_schedule_action_cancel(self):
        if False:
            i = 10
            return i + 15
        ran = False
        scheduler = GtkScheduler(GLib)
        gate = threading.Semaphore(0)

        def action(scheduler, state):
            if False:
                while True:
                    i = 10
            nonlocal ran
            ran = True
        d = scheduler.schedule_relative(0.1, action)
        d.dispose()

        def done(data):
            if False:
                i = 10
                return i + 15
            Gtk.main_quit()
            gate.release()
            return False
        GLib.timeout_add(200, done, None)
        Gtk.main()
        gate.acquire()
        assert ran is False

    def test_gtk_schedule_action_periodic(self):
        if False:
            while True:
                i = 10
        scheduler = GtkScheduler(GLib)
        gate = threading.Semaphore(0)
        period = 0.05
        counter = 3

        def action(state):
            if False:
                print('Hello World!')
            nonlocal counter
            if state:
                counter -= 1
                return state - 1
        scheduler.schedule_periodic(period, action, counter)

        def done(data):
            if False:
                print('Hello World!')
            Gtk.main_quit()
            gate.release()
            return False
        GLib.timeout_add(300, done, None)
        Gtk.main()
        gate.acquire()
        assert counter == 0