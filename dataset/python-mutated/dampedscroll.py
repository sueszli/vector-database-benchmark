"""
Damped scroll effect
====================

.. versionadded:: 1.7.0

This damped scroll effect will use the
:attr:`~kivy.effects.scroll.ScrollEffect.overscroll` to calculate the scroll
value, and slows going back to the upper or lower limit.

"""
__all__ = ('DampedScrollEffect',)
from kivy.effects.scroll import ScrollEffect
from kivy.properties import NumericProperty, BooleanProperty
from kivy.metrics import sp

class DampedScrollEffect(ScrollEffect):
    """DampedScrollEffect class. See the module documentation for more
    information.
    """
    edge_damping = NumericProperty(0.25)
    'Edge damping.\n\n    :attr:`edge_damping` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 0.25\n    '
    spring_constant = NumericProperty(2.0)
    'Spring constant.\n\n    :attr:`spring_constant` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to 2.0\n    '
    min_overscroll = NumericProperty(0.5)
    'An overscroll less than this amount will be normalized to 0.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`min_overscroll` is a :class:`~kivy.properties.NumericProperty` and\n    defaults to .5.\n    '
    round_value = BooleanProperty(True)
    'If True, when the motion stops, :attr:`value` is rounded to the nearest\n    integer.\n\n    .. versionadded:: 1.8.0\n\n    :attr:`round_value` is a :class:`~kivy.properties.BooleanProperty` and\n    defaults to True.\n    '

    def update_velocity(self, dt):
        if False:
            while True:
                i = 10
        if abs(self.velocity) <= self.min_velocity and self.overscroll == 0:
            self.velocity = 0
            if self.round_value:
                self.value = round(self.value)
            return
        total_force = self.velocity * self.friction * dt / self.std_dt
        if abs(self.overscroll) > self.min_overscroll:
            total_force += self.velocity * self.edge_damping
            total_force += self.overscroll * self.spring_constant
        else:
            self.overscroll = 0
        stop_overscroll = ''
        if not self.is_manual:
            if self.overscroll > 0 and self.velocity < 0:
                stop_overscroll = 'max'
            elif self.overscroll < 0 and self.velocity > 0:
                stop_overscroll = 'min'
        self.velocity = self.velocity - total_force
        if not self.is_manual:
            self.apply_distance(self.velocity * dt)
            if stop_overscroll == 'min' and self.value > self.min:
                self.value = self.min
                self.velocity = 0
                return
            if stop_overscroll == 'max' and self.value < self.max:
                self.value = self.max
                self.velocity = 0
                return
        self.trigger_velocity_update()

    def on_value(self, *args):
        if False:
            print('Hello World!')
        scroll_min = self.min
        scroll_max = self.max
        if scroll_min > scroll_max:
            (scroll_min, scroll_max) = (scroll_max, scroll_min)
        if self.value < scroll_min:
            self.overscroll = self.value - scroll_min
        elif self.value > scroll_max:
            self.overscroll = self.value - scroll_max
        else:
            self.overscroll = 0
        self.scroll = self.value

    def on_overscroll(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.trigger_velocity_update()

    def apply_distance(self, distance):
        if False:
            while True:
                i = 10
        os = abs(self.overscroll)
        if os:
            distance /= 1.0 + os / sp(200.0)
        super(DampedScrollEffect, self).apply_distance(distance)