"""Decorators handling min- and max- widths and heights."""
import functools

def handle_min_max_width(function):
    if False:
        while True:
            i = 10
    'Decorate a function setting used width, handling {min,max}-width.'

    @functools.wraps(function)
    def wrapper(box, *args):
        if False:
            i = 10
            return i + 15
        computed_margins = (box.margin_left, box.margin_right)
        result = function(box, *args)
        if box.width > box.max_width:
            box.width = box.max_width
            (box.margin_left, box.margin_right) = computed_margins
            result = function(box, *args)
        if box.width < box.min_width:
            box.width = box.min_width
            (box.margin_left, box.margin_right) = computed_margins
            result = function(box, *args)
        return result
    wrapper.without_min_max = function
    return wrapper

def handle_min_max_height(function):
    if False:
        for i in range(10):
            print('nop')
    'Decorate a function setting used height, handling {min,max}-height.'

    @functools.wraps(function)
    def wrapper(box, *args):
        if False:
            return 10
        computed_margins = (box.margin_top, box.margin_bottom)
        result = function(box, *args)
        if box.height > box.max_height:
            box.height = box.max_height
            (box.margin_top, box.margin_bottom) = computed_margins
            result = function(box, *args)
        if box.height < box.min_height:
            box.height = box.min_height
            (box.margin_top, box.margin_bottom) = computed_margins
            result = function(box, *args)
        return result
    wrapper.without_min_max = function
    return wrapper