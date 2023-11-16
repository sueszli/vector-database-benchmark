"""
Animations tests
================
"""
import pytest

@pytest.fixture(scope='module')
def ec_cls():
    if False:
        while True:
            i = 10

    class EventCounter:

        def __init__(self, anim):
            if False:
                print('Hello World!')
            self.n_start = 0
            self.n_progress = 0
            self.n_complete = 0
            anim.bind(on_start=self.on_start, on_progress=self.on_progress, on_complete=self.on_complete)

        def on_start(self, anim, widget):
            if False:
                for i in range(10):
                    print('nop')
            self.n_start += 1

        def on_progress(self, anim, widget, progress):
            if False:
                return 10
            self.n_progress += 1

        def on_complete(self, anim, widget):
            if False:
                i = 10
                return i + 15
            self.n_complete += 1

        def assert_(self, n_start, n_progress_greater_than_zero, n_complete):
            if False:
                for i in range(10):
                    print('nop')
            assert self.n_start == n_start
            if n_progress_greater_than_zero:
                assert self.n_progress > 0
            else:
                assert self.n_progress == 0
            assert self.n_complete == n_complete
    return EventCounter

@pytest.fixture(autouse=True)
def cleanup():
    if False:
        for i in range(10):
            print('nop')
    from kivy.animation import Animation
    Animation.cancel_all(None)

def no_animations_being_played():
    if False:
        return 10
    from kivy.animation import Animation
    return len(Animation._instances) == 0

def sleep(t):
    if False:
        print('Hello World!')
    from time import time, sleep
    from kivy.clock import Clock
    tick = Clock.tick
    deadline = time() + t
    while time() < deadline:
        sleep(0.01)
        tick()

class TestAnimation:

    def test_start_animation(self):
        if False:
            print('Hello World!')
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(1.5)
        assert w.x == pytest.approx(100)
        assert no_animations_being_played()

    def test_animation_duration_0(self):
        if False:
            while True:
                i = 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        assert no_animations_being_played()

    def test_cancel_all(self):
        if False:
            return 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a1 = Animation(x=100)
        a2 = Animation(y=100)
        w1 = Widget()
        w2 = Widget()
        a1.start(w1)
        a1.start(w2)
        a2.start(w1)
        a2.start(w2)
        assert not no_animations_being_played()
        Animation.cancel_all(None)
        assert no_animations_being_played()

    def test_cancel_all_2(self):
        if False:
            return 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a1 = Animation(x=100)
        a2 = Animation(y=100)
        w1 = Widget()
        w2 = Widget()
        a1.start(w1)
        a1.start(w2)
        a2.start(w1)
        a2.start(w2)
        assert not no_animations_being_played()
        Animation.cancel_all(None, 'x', 'z')
        assert not no_animations_being_played()
        Animation.cancel_all(None, 'y')
        assert no_animations_being_played()

    def test_stop_animation(self):
        if False:
            while True:
                i = 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(0.5)
        a.stop(w)
        assert w.x != pytest.approx(100)
        assert w.x != pytest.approx(0)
        assert no_animations_being_played()

    def test_stop_all(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w)
        assert no_animations_being_played()

    def test_stop_all_2(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=1)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w, 'x')
        assert no_animations_being_played()

    def test_duration(self):
        if False:
            i = 10
            return i + 15
        from kivy.animation import Animation
        a = Animation(x=100, d=1)
        assert a.duration == 1

    def test_transition(self):
        if False:
            while True:
                i = 10
        from kivy.animation import Animation, AnimationTransition
        a = Animation(x=100, t='out_bounce')
        assert a.transition is AnimationTransition.out_bounce

    def test_animated_properties(self):
        if False:
            return 10
        from kivy.animation import Animation
        a = Animation(x=100)
        assert a.animated_properties == {'x': 100}

    def test_animated_instruction(self):
        if False:
            print('Hello World!')
        from kivy.graphics import Scale
        from kivy.animation import Animation
        a = Animation(x=100, d=1)
        instruction = Scale(3, 3, 3)
        a.start(instruction)
        assert a.animated_properties == {'x': 100}
        assert instruction.x == pytest.approx(3)
        sleep(1.5)
        assert instruction.x == pytest.approx(100)
        assert no_animations_being_played()

    def test_weakref(self):
        if False:
            return 10
        import gc
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        w = Widget()
        a = Animation(x=100)
        a.start(w.proxy_ref)
        del w
        gc.collect()
        try:
            sleep(1.0)
        except ReferenceError:
            pass
        assert no_animations_being_played()

class TestSequence:

    def test_cancel_all(self):
        if False:
            i = 10
            return i + 15
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.cancel_all(w)
        assert no_animations_being_played()

    def test_cancel_all_2(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.cancel_all(w, 'x')
        assert no_animations_being_played()

    def test_stop_all(self):
        if False:
            i = 10
            return i + 15
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w)
        assert no_animations_being_played()

    def test_stop_all_2(self):
        if False:
            while True:
                i = 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        a.start(w)
        sleep(0.5)
        Animation.stop_all(w, 'x')
        assert no_animations_being_played()

    def test_count_events(self, ec_cls):
        if False:
            while True:
                i = 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=0.5) + Animation(x=0, d=0.5)
        w = Widget()
        ec = ec_cls(a)
        ec1 = ec_cls(a.anim1)
        ec2 = ec_cls(a.anim2)
        a.start(w)
        ec.assert_(1, False, 0)
        ec1.assert_(1, False, 0)
        ec2.assert_(0, False, 0)
        sleep(0.2)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(0, False, 0)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 0)
        sleep(0.5)
        ec.assert_(1, True, 1)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 1)
        assert no_animations_being_played()

    def test_have_properties_to_animate(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        w = Widget()
        assert not a.have_properties_to_animate(w)
        a.start(w)
        assert a.have_properties_to_animate(w)
        a.stop(w)
        assert not a.have_properties_to_animate(w)
        assert no_animations_being_played()

    def test_animated_properties(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.animation import Animation
        a = Animation(x=100, y=200) + Animation(x=0)
        assert a.animated_properties == {'x': 0, 'y': 200}

    def test_transition(self):
        if False:
            for i in range(10):
                print('nop')
        from kivy.animation import Animation
        a = Animation(x=100) + Animation(x=0)
        with pytest.raises(AttributeError):
            a.transition

class TestRepetitiveSequence:

    def test_stop(self):
        if False:
            return 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) + Animation(x=0)
        a.repeat = True
        w = Widget()
        a.start(w)
        a.stop(w)
        assert no_animations_being_played()

    def test_count_events(self, ec_cls):
        if False:
            while True:
                i = 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100, d=0.5) + Animation(x=0, d=0.5)
        a.repeat = True
        w = Widget()
        ec = ec_cls(a)
        ec1 = ec_cls(a.anim1)
        ec2 = ec_cls(a.anim2)
        a.start(w)
        ec.assert_(1, False, 0)
        ec1.assert_(1, False, 0)
        ec2.assert_(0, False, 0)
        sleep(0.2)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(0, False, 0)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 0)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(2, True, 1)
        ec2.assert_(1, True, 1)
        sleep(0.5)
        ec.assert_(1, True, 0)
        ec1.assert_(2, True, 2)
        ec2.assert_(2, True, 1)
        a.stop(w)
        ec.assert_(1, True, 1)
        ec1.assert_(2, True, 2)
        ec2.assert_(2, True, 2)
        assert no_animations_being_played()

class TestParallel:

    def test_have_properties_to_animate(self):
        if False:
            i = 10
            return i + 15
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) & Animation(y=100)
        w = Widget()
        assert not a.have_properties_to_animate(w)
        a.start(w)
        assert a.have_properties_to_animate(w)
        a.stop(w)
        assert not a.have_properties_to_animate(w)
        assert no_animations_being_played()

    def test_cancel_property(self):
        if False:
            return 10
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) & Animation(y=100)
        w = Widget()
        a.start(w)
        a.cancel_property(w, 'x')
        assert not no_animations_being_played()
        a.stop(w)
        assert no_animations_being_played()

    def test_animated_properties(self):
        if False:
            i = 10
            return i + 15
        from kivy.animation import Animation
        a = Animation(x=100) & Animation(y=100)
        assert a.animated_properties == {'x': 100, 'y': 100}

    def test_transition(self):
        if False:
            return 10
        from kivy.animation import Animation
        a = Animation(x=100) & Animation(y=100)
        with pytest.raises(AttributeError):
            a.transition

    def test_count_events(self, ec_cls):
        if False:
            for i in range(10):
                print('nop')
        from kivy.animation import Animation
        from kivy.uix.widget import Widget
        a = Animation(x=100) & Animation(y=100, d=0.5)
        w = Widget()
        ec = ec_cls(a)
        ec1 = ec_cls(a.anim1)
        ec2 = ec_cls(a.anim2)
        a.start(w)
        ec.assert_(1, False, 0)
        ec1.assert_(1, False, 0)
        ec2.assert_(1, False, 0)
        sleep(0.2)
        ec.assert_(1, False, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(1, True, 0)
        sleep(0.5)
        ec.assert_(1, False, 0)
        ec1.assert_(1, True, 0)
        ec2.assert_(1, True, 1)
        sleep(0.5)
        ec.assert_(1, False, 1)
        ec1.assert_(1, True, 1)
        ec2.assert_(1, True, 1)
        assert no_animations_being_played()