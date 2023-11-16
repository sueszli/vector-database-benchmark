import pytest
import functools
non_relative_type_widget_cls_names = ('Widget',)
relative_type_widget_cls_names = ('RelativeLayout', 'Scatter', 'ScrollView')
all_widget_cls_names = non_relative_type_widget_cls_names + relative_type_widget_cls_names

@functools.lru_cache(maxsize=1)
def get_relative_type_widget_classes():
    if False:
        print('Hello World!')
    from kivy.factory import Factory
    return tuple((Factory.get(cls_name) for cls_name in relative_type_widget_cls_names))

def is_relative_type(widget):
    if False:
        while True:
            i = 10
    return isinstance(widget, get_relative_type_widget_classes())

@pytest.mark.parametrize('widget_cls_name', relative_type_widget_cls_names)
def test_to_local_and_to_parent__relative(widget_cls_name, kivy_clock):
    if False:
        for i in range(10):
            print('nop')
    from kivy.factory import Factory
    widget = Factory.get(widget_cls_name)(pos=(100, 100))
    kivy_clock.tick()
    assert widget.to_local(0, 0) == (-100, -100)
    assert widget.to_parent(0, 0) == (100, 100)

@pytest.mark.parametrize('widget_cls_name', non_relative_type_widget_cls_names)
def test_to_local_and_to_parent__not_relative(widget_cls_name, kivy_clock):
    if False:
        for i in range(10):
            print('nop')
    from kivy.factory import Factory
    widget = Factory.get(widget_cls_name)(pos=(100, 100))
    kivy_clock.tick()
    assert widget.to_local(0, 0) == (0, 0)
    assert widget.to_parent(0, 0) == (0, 0)

@pytest.mark.parametrize('root_widget_cls_name', all_widget_cls_names)
@pytest.mark.parametrize('target_widget_cls_name', all_widget_cls_names)
def test_to_window_and_to_widget(root_widget_cls_name, target_widget_cls_name, kivy_clock):
    if False:
        i = 10
        return i + 15
    from textwrap import dedent
    from kivy.lang import Builder
    root = Builder.load_string(dedent("\n        {}:\n            pos: 100, 0\n\n            # In case the root widget is ScrollView, this cushion is needed,\n            # because ScrollView's direct child is always at pos(0, 0)\n            Widget:\n                pos: 0, 0\n\n                {}:\n                    id: target\n                    pos: 0, 100\n        ").format(root_widget_cls_name, target_widget_cls_name))
    kivy_clock.tick()
    target = root.ids.target
    if is_relative_type(root):
        assert target.to_window(*target.pos) == (100, 100)
        assert target.to_widget(0, 0) == ((-100, -100) if is_relative_type(target) else (-100, 0))
    else:
        assert target.to_window(*target.pos) == (0, 100)
        assert target.to_widget(0, 0) == ((0, -100) if is_relative_type(target) else (0, 0))