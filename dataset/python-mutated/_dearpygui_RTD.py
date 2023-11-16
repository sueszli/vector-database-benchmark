from typing import List, Any, Callable, Union, Tuple
from contextlib import contextmanager
import warnings
import functools
import inspect
import dearpygui._dearpygui as internal_dpg
from dearpygui._dearpygui import mvBuffer
from dearpygui._dearpygui import mvVec4
from dearpygui._dearpygui import mvMat4

def run_callbacks(jobs):
    if False:
        return 10
    ' New in 1.2. Runs callbacks from the callback queue and checks arguments. '
    if jobs is None:
        pass
    else:
        for job in jobs:
            if job[0] is None:
                pass
            else:
                sig = inspect.signature(job[0])
                args = []
                for arg in range(len(sig.parameters)):
                    args.append(job[arg + 1])
                job[0](*args)

def get_major_version():
    if False:
        print('Hello World!')
    ' return Dear PyGui Major Version '
    return internal_dpg.get_app_configuration()['major_version']

def get_minor_version():
    if False:
        for i in range(10):
            print('nop')
    ' return Dear PyGui Minor Version '
    return internal_dpg.get_app_configuration()['minor_version']

def get_dearpygui_version():
    if False:
        for i in range(10):
            print('nop')
    ' return Dear PyGui Version '
    return internal_dpg.get_app_configuration()['version']

def configure_item(item: Union[int, str], **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    'Configures an item after creation.'
    internal_dpg.configure_item(item, **kwargs)

def configure_app(**kwargs) -> None:
    if False:
        print('Hello World!')
    'Configures an item after creation.'
    internal_dpg.configure_app(**kwargs)

def configure_viewport(item: Union[int, str], **kwargs) -> None:
    if False:
        i = 10
        return i + 15
    'Configures a viewport after creation.'
    internal_dpg.configure_viewport(item, **kwargs)

def start_dearpygui():
    if False:
        for i in range(10):
            print('nop')
    'Prepares viewport (if not done already). sets up, cleans up, and runs main event loop.\n\n    Returns:\n        None\n    '
    if not internal_dpg.is_viewport_ok():
        raise RuntimeError('Viewport was not created and shown.')
        return
    while internal_dpg.is_dearpygui_running():
        internal_dpg.render_dearpygui_frame()

@contextmanager
def mutex():
    if False:
        for i in range(10):
            print('nop')
    ' Handles locking/unlocking render thread mutex. '
    try:
        yield internal_dpg.lock_mutex()
    finally:
        internal_dpg.unlock_mutex()

@contextmanager
def popup(parent: Union[int, str], mousebutton: int=internal_dpg.mvMouseButton_Right, modal: bool=False, tag: Union[int, str]=0, min_size: Union[List[int], Tuple[int, ...]]=[100, 100], max_size: Union[List[int], Tuple[int, ...]]=[30000, 30000], no_move: bool=False, no_background: bool=False) -> int:
    if False:
        print('Hello World!')
    "A window that will be displayed when a parent item is hovered and the corresponding mouse button has been clicked. By default a popup will shrink fit the items it contains.\n    This is useful for context windows, and simple modal window popups.\n    When popups are used a modal they have more avaliable settings (i.e. title, resize, width, height) These\n    can be set by using configure item. \n    This is a light wrapper over window. For more control over a modal|popup window use a normal window with the modal|popup keyword \n    and set the item handler and mouse events manually.\n\n    Args:\n        parent: The UI item that will need to be hovered.\n        **mousebutton: The mouse button that will trigger the window to popup.\n        **modal: Will force the user to interact with the popup.\n        **min_size: New in 1.4. Minimum window size.\n        **max_size: New in 1.4. Maximum window size.\n        **no_move: New in 1.4. Prevents the window from moving based on user input.\n        **no_background: New in 1.4. Sets Background and border alpha to transparent.\n\n    Returns:\n        item's uuid\n    "
    try:
        if tag == 0:
            _internal_popup_id = internal_dpg.generate_uuid()
        else:
            _internal_popup_id = tag
        _handler_reg_id = internal_dpg.add_item_handler_registry()
        internal_dpg.add_item_clicked_handler(mousebutton, parent=internal_dpg.last_item(), callback=lambda : internal_dpg.configure_item(_internal_popup_id, show=True))
        internal_dpg.bind_item_handler_registry(parent, _handler_reg_id)
        if modal:
            internal_dpg.add_window(modal=True, show=False, tag=_internal_popup_id, autosize=True, min_size=min_size, max_size=max_size, no_move=no_move, no_background=no_background)
        else:
            internal_dpg.add_window(popup=True, show=False, tag=_internal_popup_id, autosize=True, min_size=min_size, max_size=max_size, no_move=no_move, no_background=no_background)
        internal_dpg.push_container_stack(internal_dpg.last_container())
        yield _internal_popup_id
    finally:
        internal_dpg.pop_container_stack()

def show_style_editor() -> None:
    if False:
        while True:
            i = 10
    'Shows the standard style editor window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Style)

def show_metrics() -> None:
    if False:
        return 10
    'Shows the standard metrics window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Metrics)

def show_about() -> None:
    if False:
        while True:
            i = 10
    'Shows the standard about window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_About)

def show_debug() -> None:
    if False:
        while True:
            i = 10
    'Shows the standard debug window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Debug)

def show_documentation() -> None:
    if False:
        i = 10
        return i + 15
    'Shows the standard documentation window\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Doc)

def show_font_manager() -> None:
    if False:
        print('Hello World!')
    'Shows a debug tool for the font manager\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_Font)

def show_item_registry() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Shows the item hierarchy of your application\n\n    Returns:\n        None\n    '
    internal_dpg.show_tool(internal_dpg.mvTool_ItemRegistry)

def get_item_slot(item: Union[int, str]) -> Union[int, None]:
    if False:
        for i in range(10):
            print('nop')
    "Returns an item's target slot.\n\n    Returns:\n        slot as a int\n    "
    return internal_dpg.get_item_info(item)['target']

def is_item_container(item: Union[int, str]) -> Union[bool, None]:
    if False:
        while True:
            i = 10
    'Checks if item is a container.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_info(item)['container']

def get_item_parent(item: Union[int, str]) -> Union[int, None]:
    if False:
        print('Hello World!')
    "Gets the item's parent.\n\n    Returns:\n        parent as a int or None\n    "
    return internal_dpg.get_item_info(item)['parent']

def get_item_children(item: Union[int, str], slot: int=-1) -> Union[dict, List[int], None]:
    if False:
        print('Hello World!')
    "Provides access to the item's children slots.\n\n    Returns:\n        A 2-D tuple of children slots ex. ((child_slot_1),(child_slot_2),(child_slot_3),...) or a single slot if slot is used.\n    "
    if slot < 0 or slot > 4:
        return internal_dpg.get_item_info(item)['children']
    return internal_dpg.get_item_info(item)['children'][slot]

def get_item_type(item: Union[int, str]) -> Union[str]:
    if False:
        for i in range(10):
            print('nop')
    "Gets the item's type.\n\n    Returns:\n        type as a string or None\n    "
    return internal_dpg.get_item_info(item)['type']

def get_item_theme(item: Union[int, str]) -> int:
    if False:
        i = 10
        return i + 15
    "Gets the item's theme.\n\n    Returns:\n        theme's uuid\n    "
    return internal_dpg.get_item_info(item)['theme']

def get_item_font(item: Union[int, str]) -> int:
    if False:
        print('Hello World!')
    "Gets the item's font.\n\n    Returns:\n        font's uuid\n    "
    return internal_dpg.get_item_info(item)['font']

def get_item_disabled_theme(item: Union[int, str]) -> int:
    if False:
        i = 10
        return i + 15
    "Gets the item's disabled theme.\n\n    Returns:\n        theme's uuid\n    "
    return internal_dpg.get_item_info(item)['disabled_theme']

def enable_item(item: Union[int, str]):
    if False:
        return 10
    'Enables the item.\n\n    Args:\n        **item: Item to enable.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, enabled=True)

def disable_item(item: Union[int, str]):
    if False:
        for i in range(10):
            print('nop')
    'Disables the item.\n\n    Args:\n        **item: Item to disable.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, enabled=False)

def set_item_label(item: Union[int, str], label: str):
    if False:
        while True:
            i = 10
    'Sets the item\'s displayed label, anything after the characters "##" in the name will not be shown.\n\n    Args:\n        item: Item label will be applied to.\n        label: Displayed name to be applied.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, label=label)

def set_item_source(item: Union[int, str], source: Union[int, str]):
    if False:
        i = 10
        return i + 15
    'Sets the item\'s value, to the source\'s value. Widget\'s value will now be "linked" to source\'s value.\n\n    Args:\n        item: Item to me linked.\n        source: Source to link to.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, source=source)

def set_item_pos(item: Union[int, str], pos: List[float]):
    if False:
        print('Hello World!')
    "Sets the item's position.\n\n    Args:\n        item: Item the absolute position will be applied to.\n        pos: X and Y positions relative to parent of the item.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, pos=pos)

def set_item_width(item: Union[int, str], width: int):
    if False:
        print('Hello World!')
    "Sets the item's width.\n\n    Args:\n        item: Item the Width will be applied to.\n        width: Width to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, width=width)

def set_item_height(item: Union[int, str], height: int):
    if False:
        while True:
            i = 10
    "Sets the item's height.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, height=height)

def set_item_indent(item: Union[int, str], indent: int):
    if False:
        return 10
    "Sets the item's indent.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, indent=indent)

def set_item_track_offset(item: Union[int, str], offset: float):
    if False:
        i = 10
        return i + 15
    "Sets the item's track offset.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, track_offset=offset)

def set_item_payload_type(item: Union[int, str], payload_type: str):
    if False:
        return 10
    "Sets the item's payload type.\n\n    Args:\n        item: Item the Height will be applied to.\n        height: Height to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, payload_type=str)

def set_item_callback(item: Union[int, str], callback: Callable):
    if False:
        for i in range(10):
            print('nop')
    "Sets the item's callack.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, callback=callback)

def set_item_drag_callback(item: Union[int, str], callback: Callable):
    if False:
        i = 10
        return i + 15
    "Sets the item's drag callack.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, drag_callback=callback)

def set_item_drop_callback(item: Union[int, str], callback: Callable):
    if False:
        return 10
    "Sets the item's drop callack.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, drop_callback=callback)

def track_item(item: Union[int, str]):
    if False:
        while True:
            i = 10
    'Track item in scroll region.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, tracked=True)

def untrack_item(item: Union[int, str]):
    if False:
        for i in range(10):
            print('nop')
    'Track item in scroll region.\n\n    Args:\n        item: Item the callback will be applied to.\n        callback: Callback to be applied.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, tracked=False)

def set_item_user_data(item: Union[int, str], user_data: Any):
    if False:
        return 10
    "Sets the item's callack_data to any python object.\n\n    Args:\n        item: Item the callback will be applied to.\n        user_data: Callback_data to be applied.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_item(item, user_data=user_data)

def show_item(item: Union[int, str]):
    if False:
        i = 10
        return i + 15
    'Shows the item.\n\n    Args:\n        item: Item to show.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_item(item, show=True)

def hide_item(item: Union[int, str], *, children_only: bool=False):
    if False:
        i = 10
        return i + 15
    'Hides the item.\n\n    Args:\n        **item: Item to hide.\n\n    Returns:\n        None\n    '
    if children_only:
        children = get_item_children(item)
        for child in children:
            internal_dpg.configure_item(child, show=False)
    else:
        internal_dpg.configure_item(item, show=False)

def get_item_label(item: Union[int, str]) -> Union[str, None]:
    if False:
        for i in range(10):
            print('nop')
    "Gets the item's label.\n\n    Returns:\n        label as a string or None\n    "
    return internal_dpg.get_item_configuration(item)['label']

def get_item_filter_key(item: Union[int, str]) -> Union[str, None]:
    if False:
        return 10
    "Gets the item's filter key.\n\n    Returns:\n        filter key as a string or None\n    "
    return internal_dpg.get_item_configuration(item)['filter_key']

def is_item_tracked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        return 10
    'Checks if item is tracked.\n\n    Returns:\n        tracked as a bool or None\n    '
    return internal_dpg.get_item_configuration(item)['tracked']

def is_item_search_delayed(item: Union[int, str]) -> Union[bool, None]:
    if False:
        return 10
    'Checks if item is search delayed.\n\n    Returns:\n        tracked as a bool or None\n    '
    return internal_dpg.get_item_configuration(item)['delay_search']

def get_item_indent(item: Union[int, str]) -> Union[int, None]:
    if False:
        print('Hello World!')
    "Gets the item's indent.\n\n    Returns:\n        indent as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['indent']

def get_item_track_offset(item: Union[int, str]) -> Union[float, None]:
    if False:
        while True:
            i = 10
    "Gets the item's track offset.\n\n    Returns:\n        track offset as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['track_offset']

def get_item_width(item: Union[int, str]) -> Union[int, None]:
    if False:
        for i in range(10):
            print('nop')
    "Gets the item's width.\n\n    Returns:\n        width as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['width']

def get_item_height(item: Union[int, str]) -> Union[int, None]:
    if False:
        i = 10
        return i + 15
    "Gets the item's height.\n\n    Returns:\n        height as a int or None\n    "
    return internal_dpg.get_item_configuration(item)['height']

def get_item_callback(item: Union[int, str]) -> Union[Callable, None]:
    if False:
        i = 10
        return i + 15
    "Gets the item's callback.\n\n    Returns:\n        callback as a callable or None\n    "
    return internal_dpg.get_item_configuration(item)['callback']

def get_item_drag_callback(item: Union[int, str]) -> Union[Callable, None]:
    if False:
        while True:
            i = 10
    "Gets the item's drag callback.\n\n    Returns:\n        callback as a callable or None\n    "
    return internal_dpg.get_item_configuration(item)['drag_callback']

def get_item_drop_callback(item: Union[int, str]) -> Union[Callable, None]:
    if False:
        while True:
            i = 10
    "Gets the item's drop callback.\n\n    Returns:\n        callback as a callable or None\n    "
    return internal_dpg.get_item_configuration(item)['drop_callback']

def get_item_user_data(item: Union[int, str]) -> Union[Any, None]:
    if False:
        for i in range(10):
            print('nop')
    "Gets the item's callback data.\n\n    Returns:\n        callback data as a python object or None\n    "
    return internal_dpg.get_item_configuration(item)['user_data']

def get_item_source(item: Union[int, str]) -> Union[str, None]:
    if False:
        while True:
            i = 10
    "Gets the item's source.\n\n    Returns:\n        source as a string or None\n    "
    return internal_dpg.get_item_configuration(item)['source']

def is_item_hovered(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is hovered.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['hovered']

def is_item_active(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is active.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['active']

def is_item_focused(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is focused.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['focused']

def is_item_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        while True:
            i = 10
    'Checks if item is clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['clicked']

def is_item_left_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        return 10
    'Checks if item is left clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['left_clicked']

def is_item_right_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is right clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['right_clicked']

def is_item_middle_clicked(item: Union[int, str]) -> Union[bool, None]:
    if False:
        i = 10
        return i + 15
    'Checks if item is middle clicked.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['middle_clicked']

def is_item_visible(item: Union[int, str]) -> Union[bool, None]:
    if False:
        while True:
            i = 10
    'Checks if item is visible.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['visible']

def is_item_edited(item: Union[int, str]) -> Union[bool, None]:
    if False:
        while True:
            i = 10
    'Checks if item is edited.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['edited']

def is_item_activated(item: Union[int, str]) -> Union[bool, None]:
    if False:
        while True:
            i = 10
    'Checks if item is activated.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['activated']

def is_item_deactivated(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is deactivated.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['deactivated']

def is_item_deactivated_after_edit(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is deactivated_after_edit.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['deactivated_after_edit']

def is_item_toggled_open(item: Union[int, str]) -> Union[bool, None]:
    if False:
        for i in range(10):
            print('nop')
    'Checks if item is toggled_open.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['toggled_open']

def is_item_ok(item: Union[int, str]) -> Union[bool, None]:
    if False:
        i = 10
        return i + 15
    'Checks if item is ok and can be used.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_state(item)['ok']

def is_item_shown(item: Union[int, str]) -> Union[bool, None]:
    if False:
        return 10
    'Checks if item is shown.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_configuration(item)['show']

def is_item_enabled(item: Union[int, str]) -> Union[bool, None]:
    if False:
        print('Hello World!')
    'Checks if item is enabled.\n\n    Returns:\n        status as a bool\n    '
    return internal_dpg.get_item_configuration(item)['enabled']

def get_item_pos(item: Union[int, str]) -> List[int]:
    if False:
        i = 10
        return i + 15
    "Returns item's position.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['pos']

def get_available_content_region(item: Union[int, str]) -> List[int]:
    if False:
        print('Hello World!')
    "Returns item's available content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['content_region_avail']

def get_item_rect_size(item: Union[int, str]) -> List[int]:
    if False:
        while True:
            i = 10
    "Returns item's available content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['rect_size']

def get_item_rect_min(item: Union[int, str]) -> List[int]:
    if False:
        for i in range(10):
            print('nop')
    "Returns item's minimum content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['rect_min']

def get_item_rect_max(item: Union[int, str]) -> List[int]:
    if False:
        while True:
            i = 10
    "Returns item's maximum content region.\n\n    Returns:\n        position\n    "
    return internal_dpg.get_item_state(item)['rect_max']

def set_viewport_clear_color(color: List[int]):
    if False:
        i = 10
        return i + 15
    "Sets the viewport's clear color.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, clear_color=color)

def set_viewport_small_icon(icon: str):
    if False:
        return 10
    "Sets the viewport's small icon. Must be ico for windows.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, small_icon=icon)

def set_viewport_large_icon(icon: str):
    if False:
        for i in range(10):
            print('nop')
    "Sets the viewport's large icon. Must be ico for windows.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, large_icon=icon)

def set_viewport_pos(pos: List[float]):
    if False:
        print('Hello World!')
    "Sets the viewport's position.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, x_pos=pos[0], y_pos=pos[1])

def set_viewport_width(width: int):
    if False:
        i = 10
        return i + 15
    "Sets the viewport's width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, width=width)

def set_viewport_height(height: int):
    if False:
        print('Hello World!')
    "Sets the viewport's height.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, height=height)

def set_viewport_min_width(width: int):
    if False:
        while True:
            i = 10
    "Sets the viewport's minimum width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, min_width=width)

def set_viewport_max_width(width: int):
    if False:
        for i in range(10):
            print('nop')
    "Sets the viewport's max width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, max_width=width)

def set_viewport_min_height(height: int):
    if False:
        while True:
            i = 10
    "Sets the viewport's minimum height.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, min_height=height)

def set_viewport_max_height(height: int):
    if False:
        print('Hello World!')
    "Sets the viewport's max width.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, max_height=height)

def set_viewport_title(title: str):
    if False:
        while True:
            i = 10
    "Sets the viewport's title.\n\n    Returns:\n        None\n    "
    internal_dpg.configure_viewport(0, title=title)

def set_viewport_always_top(value: bool):
    if False:
        return 10
    'Sets the viewport always on top.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, always_on_top=value)

def set_viewport_resizable(value: bool):
    if False:
        print('Hello World!')
    'Sets the viewport resizable.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, resizable=value)

def set_viewport_vsync(value: bool):
    if False:
        i = 10
        return i + 15
    'Sets the viewport vsync.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, vsync=value)

def set_viewport_decorated(value: bool):
    if False:
        i = 10
        return i + 15
    'Sets the viewport to be decorated.\n\n    Returns:\n        None\n    '
    internal_dpg.configure_viewport(0, decorated=value)

def get_viewport_clear_color() -> List[int]:
    if False:
        for i in range(10):
            print('nop')
    "Gets the viewport's clear color.\n\n    Returns:\n        List[int]\n    "
    return internal_dpg.get_viewport_configuration()['clear_color']

def get_viewport_pos() -> List[float]:
    if False:
        i = 10
        return i + 15
    "Gets the viewport's position.\n\n    Returns:\n        viewport position.\n    "
    config = internal_dpg.get_viewport_configuration()
    x_pos = config['x_pos']
    y_pos = config['y_pos']
    return [x_pos, y_pos]

def get_viewport_width() -> int:
    if False:
        for i in range(10):
            print('nop')
    "Gets the viewport's width.\n\n    Returns:\n        viewport width\n    "
    return internal_dpg.get_viewport_configuration()['width']

def get_viewport_client_width() -> int:
    if False:
        while True:
            i = 10
    "Gets the viewport's client width.\n\n    Returns:\n        viewport width\n    "
    return internal_dpg.get_viewport_configuration()['client_width']

def get_viewport_client_height() -> int:
    if False:
        i = 10
        return i + 15
    "Gets the viewport's client height.\n\n    Returns:\n        viewport width\n    "
    return internal_dpg.get_viewport_configuration()['client_height']

def get_viewport_height() -> int:
    if False:
        while True:
            i = 10
    "Gets the viewport's height.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['height']

def get_viewport_min_width() -> int:
    if False:
        return 10
    "Gets the viewport's minimum width.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['min_width']

def get_viewport_max_width() -> int:
    if False:
        return 10
    "Gets the viewport's max width.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['max_width']

def get_viewport_min_height() -> int:
    if False:
        i = 10
        return i + 15
    "Gets the viewport's minimum height.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['min_height']

def get_viewport_max_height() -> int:
    if False:
        i = 10
        return i + 15
    "Gets the viewport's max width.\n\n    Returns:\n        int\n    "
    return internal_dpg.get_viewport_configuration()['max_height']

def get_viewport_title() -> str:
    if False:
        return 10
    "Gets the viewport's title.\n\n    Returns:\n        str\n    "
    return internal_dpg.get_viewport_configuration()['title']

def is_viewport_always_top() -> bool:
    if False:
        i = 10
        return i + 15
    'Checks the viewport always on top flag.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['always_on_top']

def is_viewport_resizable() -> bool:
    if False:
        print('Hello World!')
    'Checks the viewport resizable flag.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['resizable']

def is_viewport_vsync_on() -> bool:
    if False:
        while True:
            i = 10
    'Checks the viewport vsync flag.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['vsync']

def is_viewport_decorated() -> bool:
    if False:
        i = 10
        return i + 15
    'Checks if the viewport is docorated.\n\n    Returns:\n        bool\n    '
    return internal_dpg.get_viewport_configuration()['decorated']

def deprecated(reason):
    if False:
        i = 10
        return i + 15
    string_types = (type(b''), type(u''))
    if isinstance(reason, string_types):

        def decorator(func1):
            if False:
                return 10
            fmt1 = 'Call to deprecated function {name} ({reason}).'

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(fmt1.format(name=func1.__name__, reason=reason), category=DeprecationWarning, stacklevel=2)
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)
            return new_func1
        return decorator
    elif inspect.isfunction(reason):
        func2 = reason
        fmt2 = 'Call to deprecated function {name}.'

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(fmt2.format(name=func2.__name__), category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)
        return new_func2

@deprecated("Use 'configure_app(docking=True, docking_space=dock_space)'.")
def enable_docking(dock_space=False):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    internal_dpg.configure_app(docking=True, docking_space=dock_space)

@deprecated("Use 'configure_app(init_file=file)'.")
def set_init_file(file='dpg.ini'):
    if False:
        while True:
            i = 10
    ' deprecated function '
    internal_dpg.configure_app(init_file=file)

@deprecated("Use 'configure_app(init_file=file, load_init_file=True)'.")
def load_init_file(file):
    if False:
        print('Hello World!')
    ' deprecated function '
    internal_dpg.configure_app(init_file=file, load_init_file=True)

@deprecated('Use: `is_viewport_ok(...)`')
def is_viewport_created():
    if False:
        print('Hello World!')
    ' deprecated function '
    return internal_dpg.is_viewport_ok()

@deprecated('Use: \ncreate_viewport()\nsetup_dearpygui()\nshow_viewport()')
def setup_viewport():
    if False:
        return 10
    ' deprecated function '
    internal_dpg.create_viewport()
    internal_dpg.setup_dearpygui()
    internal_dpg.show_viewport()

@deprecated('Use: `bind_item_theme(...)`')
def set_item_theme(item, theme):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.bind_item_theme(item, theme)

@deprecated('Use: `bind_item_type_disabled_theme(...)`')
def set_item_type_disabled_theme(item, theme):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.bind_item_type_disabled_theme(item, theme)

@deprecated('Use: `bind_item_type_theme(...)`')
def set_item_type_theme(item, theme):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.bind_item_type_theme(item, theme)

@deprecated('Use: `bind_item_font(...)`')
def set_item_font(item, font):
    if False:
        print('Hello World!')
    ' deprecated function '
    return internal_dpg.bind_item_font(item, font)

@deprecated('Use: `add_item_activated_handler(...)`')
def add_activated_handler(parent, **kwargs):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    return internal_dpg.add_item_activated_handler(parent, **kwargs)

@deprecated('Use: `add_item_active_handler(...)`')
def add_active_handler(parent, **kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_item_active_handler(parent, **kwargs)

@deprecated('Use: `add_item_clicked_handler(...)`')
def add_clicked_handler(parent, button=-1, **kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.add_item_clicked_handler(parent, button, **kwargs)

@deprecated('Use: `add_item_deactived_after_edit_handler(...)`')
def add_deactivated_after_edit_handler(parent, **kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_item_deactivated_after_edit_handler(parent, **kwargs)

@deprecated('Use: `add_item_deactivated_handler(...)`')
def add_deactivated_handler(parent, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    return internal_dpg.add_item_deactivated_handler(parent, **kwargs)

@deprecated('Use: `add_item_edited_handler(...)`')
def add_edited_handler(parent, **kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_item_edited_handler(parent, **kwargs)

@deprecated('Use: `add_item_focus_handler(...)`')
def add_focus_handler(parent, **kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_item_focus_handler(parent, **kwargs)

@deprecated('Use: `add_item_hover_handler(...)`')
def add_hover_handler(parent, **kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_item_hover_handler(parent, **kwargs)

@deprecated('Use: `add_item_resize_handler(...)`')
def add_resize_handler(parent, **kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.add_item_resize_handler(parent, **kwargs)

@deprecated('Use: `add_item_toggled_open_handler(...)`')
def add_toggled_open_handler(parent, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    return internal_dpg.add_item_toggled_open_handler(parent, **kwargs)

@deprecated('Use: `add_item_visible_handler(...)`')
def add_visible_handler(parent, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ' deprecated function '
    return internal_dpg.add_item_visible_handler(parent, **kwargs)

@deprecated('Use: `bind_colormap(...)`')
def set_colormap(item, source):
    if False:
        print('Hello World!')
    ' deprecated function '
    return internal_dpg.bind_colormap(item, source)

@deprecated('Use: `bind_theme(0)`')
def reset_default_theme(item, source):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.bind_theme(item, source)

@deprecated
def set_staging_mode(mode):
    if False:
        print('Hello World!')
    ' deprecated function '
    pass

@deprecated
def add_table_next_column(**kwargs):
    if False:
        while True:
            i = 10
    ' deprecated function '
    pass

@deprecated('Use: add_stage')
def add_staging_container(**kwargs):
    if False:
        return 10
    ' deprecated function '
    return internal_dpg.add_stage(**kwargs)

@deprecated('Use: stage')
@contextmanager
def staging_container(**kwargs):
    if False:
        print('Hello World!')
    "\n\tdeprecated function\n\tArgs:\n\t\t**label (str): Overrides 'name' as label.\n\t\t**user_data (Any): User data for callbacks.\n\t\t**use_internal_label (bool): Use generated internal label instead of user specified (appends ### uuid).\n\t\t**id (Union[int, str]): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        warnings.warn("'staging_container' is deprecated and was changed to 'stage'", DeprecationWarning, 2)
        widget = internal_dpg.add_stage_container(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@deprecated('Use: add_spacer(...)')
def add_spacing(**kwargs):
    if False:
        print('Hello World!')
    "\t(deprecated function) Adds vertical spacing. \n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks.\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tcount (int, optional): Number of spacings to add the size is dependant on the curret style.\n\tReturns:\n\t\tUnion[int, str]\n\t"
    if 'count' in kwargs.keys():
        count = kwargs['count']
        kwargs.pop('count', None)
        internal_dpg.add_group(**kwargs)
        internal_dpg.push_container_stack(internal_dpg.last_container())
        for i in range(count):
            internal_dpg.add_spacer()
        result_id = internal_dpg.pop_container_stack()
    else:
        result_id = internal_dpg.add_spacer(**kwargs)
    return result_id

@deprecated('Use: add_spacer(...)')
def add_dummy(**kwargs):
    if False:
        return 10
    "\t(deprecated function) Adds a spacer or 'dummy' object.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks.\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_spacer(**kwargs)

@deprecated('Use: `destroy_context()`')
def cleanup_dearpygui():
    if False:
        print('Hello World!')
    ' deprecated function '
    return internal_dpg.destroy_context()

@deprecated('Use: group(horizontal=True)')
def add_same_line(**kwargs):
    if False:
        i = 10
        return i + 15
    ' deprecated function '
    last_item = internal_dpg.last_item()
    group = internal_dpg.add_group(horizontal=True, **kwargs)
    internal_dpg.move_item(last_item, parent=group)
    internal_dpg.capture_next_item(lambda s: internal_dpg.move_item(s, parent=group))
    return group

@deprecated('Use: `add_child_window()`')
def add_child(**kwargs):
    if False:
        print('Hello World!')
    "\t(deprecated function) Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_child_window(**kwargs)

@deprecated('Use: `child_window()`')
@contextmanager
def child(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t(deprecated function) Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_child_window(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@deprecated('Use: Just not recommended')
def setup_registries() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Adds default registries for fonts, handlers, textures, colormaps, and values.'
    internal_dpg.add_font_registry(tag=internal_dpg.mvReservedUUID_0)
    internal_dpg.add_handler_registry(tag=internal_dpg.mvReservedUUID_1)
    internal_dpg.add_texture_registry(tag=internal_dpg.mvReservedUUID_2)
    internal_dpg.add_value_registry(tag=internal_dpg.mvReservedUUID_3)
    internal_dpg.add_colormap_registry(tag=internal_dpg.mvReservedUUID_4)

@deprecated('Use: `set_frame_callback()`')
def set_start_callback(callback):
    if False:
        while True:
            i = 10
    ' deprecated function '
    return internal_dpg.set_frame_callback(3, callback)

@contextmanager
def child_window(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_child_window(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def clipper(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Helper to manually clip large list of items. Increases performance by not searching or drawing widgets outside of the clipped region.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_clipper(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def collapsing_header(**kwargs):
    if False:
        return 10
    "\t Adds a collapsing header to add items to. Must be closed with the end command.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Adds the ability to hide this widget by pressing the (x) in the top right of widget.\n\t\tdefault_open (bool, optional): Sets the collapseable header open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_collapsing_header(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def colormap_registry(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a colormap registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_colormap_registry(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def custom_series(x, y, channel_count, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a custom series to a plot. New in 1.6.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tchannel_count (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\ty1 (Any, optional): \n\t\ty2 (Any, optional): \n\t\ty3 (Any, optional): \n\t\ttooltip (bool, optional): Show tooltip when plot is hovered.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_custom_series(x, y, channel_count, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def drag_payload(**kwargs):
    if False:
        print('Hello World!')
    "\t User data payload for drag and drop operations.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdrag_data (Any, optional): Drag data\n\t\tdrop_data (Any, optional): Drop data\n\t\tpayload_type (str, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_drag_payload(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def draw_layer(**kwargs):
    if False:
        while True:
            i = 10
    "\t New in 1.1. Creates a layer useful for grouping drawlist items.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tperspective_divide (bool, optional): New in 1.1. apply perspective divide\n\t\tdepth_clipping (bool, optional): New in 1.1. apply depth clipping\n\t\tcull_mode (int, optional): New in 1.1. culling mode, mvCullMode_* constants. Only works with triangles currently.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_draw_layer(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def draw_node(**kwargs):
    if False:
        while True:
            i = 10
    "\t New in 1.1. Creates a drawing node to associate a transformation matrix. Child node matricies will concatenate.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_draw_node(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def drawlist(width, height, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a drawing canvas.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_drawlist(width, height, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def file_dialog(**kwargs):
    if False:
        print('Hello World!')
    "\t Displays a file or directory selector depending on keywords. Displays a file dialog by default. Callback will be ran when the file or directory picker is closed. The app_data arguemnt will be populated with information related to the file and directory as a dictionary.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_path (str, optional): Path that the file dialog will default to when opened.\n\t\tdefault_filename (str, optional): Default name that will show in the file name input.\n\t\tfile_count (int, optional): Number of visible files in the dialog.\n\t\tmodal (bool, optional): Forces user interaction with the file selector.\n\t\tdirectory_selector (bool, optional): Shows only directory/paths as options. Allows selection of directory/paths only.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tcancel_callback (Callable, optional): Callback called when cancel button is clicked.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_file_dialog(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def filter_set(**kwargs):
    if False:
        while True:
            i = 10
    "\t Helper to parse and apply text filters (e.g. aaaaa[, bbbbb][, ccccc])\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_filter_set(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def font(file, size, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds font to a font registry.\n\n\tArgs:\n\t\tfile (str): \n\t\tsize (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tpixel_snapH (bool, optional): Align every glyph to pixel boundary. Useful e.g. if you are merging a non-pixel aligned font with the default font, or rendering text piece-by-piece (e.g. for coloring).\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\t\tdefault_font (bool, optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_font(file, size, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def font_registry(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a font registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_font_registry(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def group(**kwargs):
    if False:
        while True:
            i = 10
    "\t Creates a group that other widgets can belong to. The group allows item commands to be issued for all of its members.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\thorizontal (bool, optional): Forces child widgets to be added in a horizontal layout.\n\t\thorizontal_spacing (float, optional): Spacing for the horizontal layout.\n\t\txoffset (float, optional): Offset from containing window x item location within group.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_group(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def handler_registry(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_handler_registry(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def item_handler_registry(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds an item handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_item_handler_registry(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def menu(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a menu to an existing menu bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_menu(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def menu_bar(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a menu bar to a window.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_menu_bar(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def node(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a node to a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdraggable (bool, optional): Allow node to be draggable.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_node(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def node_attribute(**kwargs):
    if False:
        return 10
    "\t Adds a node attribute to a node.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tattribute_type (int, optional): mvNode_Attr_Input, mvNode_Attr_Output, or mvNode_Attr_Static.\n\t\tshape (int, optional): Pin shape.\n\t\tcategory (str, optional): Category\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_node_attribute(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def node_editor(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdelink_callback (Callable, optional): Callback ran when a link is detached.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tminimap (bool, optional): Shows or hides the Minimap. New in 1.6.\n\t\tminimap_location (int, optional): mvNodeMiniMap_Location_* constants. New in 1.6.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_node_editor(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def plot(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a plot which is used to hold series, and can be drawn to with draw commands.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_title (bool, optional): the plot title will not be displayed\n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_box_select (bool, optional): the user will not be able to box-select with right-click drag\n\t\tno_mouse_pos (bool, optional): the mouse position, in plot coordinates, will not be displayed inside of the plot\n\t\tno_highlight (bool, optional): plot items will not be highlighted when their legend entry is hovered\n\t\tno_child (bool, optional): a child window region will not be used to capture mouse scroll (can boost performance for single ImGui window applications)\n\t\tquery (bool, optional): the user will be able to draw query rects with middle - mouse or CTRL + right - click drag\n\t\tcrosshairs (bool, optional): the default mouse cursor will be replaced with a crosshair when hovered\n\t\tanti_aliased (bool, optional): plot lines will be software anti-aliased (not recommended for high density plots, prefer MSAA)\n\t\tequal_aspects (bool, optional): primary x and y axes will be constrained to have the same units/pixel (does not apply to auxiliary y-axes)\n\t\tuse_local_time (bool, optional): axis labels will be formatted for your timezone when\n\t\tuse_ISO8601 (bool, optional): dates will be formatted according to ISO 8601 where applicable (e.g. YYYY-MM-DD, YYYY-MM, --MM-DD, etc.)\n\t\tuse_24hour_clock (bool, optional): times will be formatted using a 24 hour clock\n\t\tpan_button (int, optional): enables panning when held\n\t\tpan_mod (int, optional): optional modifier that must be held for panning\n\t\tfit_button (int, optional): fits visible data when double clicked\n\t\tcontext_menu_button (int, optional): opens plot context menu (if enabled) when clicked\n\t\tbox_select_button (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_mod (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_cancel_button (int, optional): cancels active box selection when pressed\n\t\tquery_button (int, optional): begins query selection when pressed and end query selection when released\n\t\tquery_mod (int, optional): optional modifier that must be held for query selection\n\t\tquery_toggle_mod (int, optional): when held, active box selections turn into queries\n\t\thorizontal_mod (int, optional): expands active box selection/query horizontally to plot edge when held\n\t\tvertical_mod (int, optional): expands active box selection/query vertically to plot edge when held\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_plot(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def plot_axis(axis, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds an axis to a plot.\n\n\tArgs:\n\t\taxis (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tno_gridlines (bool, optional): \n\t\tno_tick_marks (bool, optional): \n\t\tno_tick_labels (bool, optional): \n\t\tlog_scale (bool, optional): \n\t\tinvert (bool, optional): \n\t\tlock_min (bool, optional): \n\t\tlock_max (bool, optional): \n\t\ttime (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_plot_axis(axis, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def stage(**kwargs):
    if False:
        return 10
    "\t Adds a stage.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_stage(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def subplots(rows, columns, **kwargs):
    if False:
        return 10
    "\t Adds a collection of plots.\n\n\tArgs:\n\t\trows (int): \n\t\tcolumns (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\trow_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tcolumn_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tno_title (bool, optional): \n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_resize (bool, optional): resize splitters between subplot cells will be not be provided\n\t\tno_align (bool, optional): subplot edges will not be aligned vertically or horizontally\n\t\tlink_rows (bool, optional): link the y-axis limits of all plots in each row (does not apply auxiliary y-axes)\n\t\tlink_columns (bool, optional): link the x-axis limits of all plots in each column\n\t\tlink_all_x (bool, optional): link the x-axis limits in every plot in the subplot\n\t\tlink_all_y (bool, optional): link the y-axis limits in every plot in the subplot (does not apply to auxiliary y-axes)\n\t\tcolumn_major (bool, optional): subplots are added in column major order instead of the default row major order\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_subplots(rows, columns, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tab(**kwargs):
    if False:
        return 10
    "\t Adds a tab to a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Creates a button on the tab that can hide the tab.\n\t\tno_tooltip (bool, optional): Disable tooltip for the given tab.\n\t\torder_mode (bool, optional): set using a constant: mvTabOrder_Reorderable: allows reordering, mvTabOrder_Fixed: fixed ordering, mvTabOrder_Leading: adds tab to front, mvTabOrder_Trailing: adds tab to back\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_tab(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tab_bar(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\treorderable (bool, optional): Allows for the user to change the order of the tabs.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_tab_bar(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def table(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\theader_row (bool, optional): show headers at the top of the columns\n\t\tclipper (bool, optional): Use clipper (rows must be same height).\n\t\tinner_width (int, optional): \n\t\tpolicy (int, optional): \n\t\tfreeze_rows (int, optional): \n\t\tfreeze_columns (int, optional): \n\t\tsort_multi (bool, optional): Hold shift when clicking headers to sort on multiple column.\n\t\tsort_tristate (bool, optional): Allow no sorting, disable default sorting.\n\t\tresizable (bool, optional): Enable resizing columns\n\t\treorderable (bool, optional): Enable reordering columns in header row (need calling TableSetupColumn() + TableHeadersRow() to display headers)\n\t\thideable (bool, optional): Enable hiding/disabling columns in context menu.\n\t\tsortable (bool, optional): Enable sorting. Call TableGetSortSpecs() to obtain sort specs. Also see ImGuiTableFlags_SortMulti and ImGuiTableFlags_SortTristate.\n\t\tcontext_menu_in_body (bool, optional): Right-click on columns body/contents will display table context menu. By default it is available in TableHeadersRow().\n\t\trow_background (bool, optional): Set each RowBg color with ImGuiCol_TableRowBg or ImGuiCol_TableRowBgAlt (equivalent of calling TableSetBgColor with ImGuiTableBgFlags_RowBg0 on each row manually)\n\t\tborders_innerH (bool, optional): Draw horizontal borders between rows.\n\t\tborders_outerH (bool, optional): Draw horizontal borders at the top and bottom.\n\t\tborders_innerV (bool, optional): Draw vertical borders between columns.\n\t\tborders_outerV (bool, optional): Draw vertical borders on the left and right sides.\n\t\tno_host_extendX (bool, optional): Make outer width auto-fit to columns, overriding outer_size.x value. Only available when ScrollX/ScrollY are disabled and Stretch columns are not used.\n\t\tno_host_extendY (bool, optional): Make outer height stop exactly at outer_size.y (prevent auto-extending table past the limit). Only available when ScrollX/ScrollY are disabled. Data below the limit will be clipped and not visible.\n\t\tno_keep_columns_visible (bool, optional): Disable keeping column always minimally visible when ScrollX is off and table gets too small. Not recommended if columns are resizable.\n\t\tprecise_widths (bool, optional): Disable distributing remainder width to stretched columns (width allocation on a 100-wide table with 3 columns: Without this flag: 33,33,34. With this flag: 33,33,33). With larger number of columns, resizing will appear to be less smooth.\n\t\tno_clip (bool, optional): Disable clipping rectangle for every individual columns.\n\t\tpad_outerX (bool, optional): Default if BordersOuterV is on. Enable outer-most padding. Generally desirable if you have headers.\n\t\tno_pad_outerX (bool, optional): Default if BordersOuterV is off. Disable outer-most padding.\n\t\tno_pad_innerX (bool, optional): Disable inner padding between columns (double inner padding if BordersOuterV is on, single inner padding if BordersOuterV is off).\n\t\tscrollX (bool, optional): Enable horizontal scrolling. Require 'outer_size' parameter of BeginTable() to specify the container size. Changes default sizing policy. Because this create a child window, ScrollY is currently generally recommended when using ScrollX.\n\t\tscrollY (bool, optional): Enable vertical scrolling.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_table(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def table_cell(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_table_cell(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def table_row(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a table row.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_table_row(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def template_registry(**kwargs):
    if False:
        return 10
    "\t Adds a template registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_template_registry(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def texture_registry(**kwargs):
    if False:
        return 10
    "\t Adds a dynamic texture.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_texture_registry(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def theme(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a theme.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\t\tdefault_theme (bool, optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_theme(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def theme_component(item_type=0, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a theme component.\n\n\tArgs:\n\t\titem_type (int, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tenabled_state (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_theme_component(item_type, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tooltip(parent, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a tooltip window.\n\n\tArgs:\n\t\tparent (Union[int, str]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay (float, optional): Activation delay: time, in seconds, during which the mouse should stay still in order to display the tooltip.  May be zero for instant activation.\n\t\thide_on_activity (bool, optional): Hide the tooltip if the user has moved the mouse.  If False, the tooltip will follow mouse pointer.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_tooltip(parent, **kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def tree_node(**kwargs):
    if False:
        return 10
    "\t Adds a tree node to add items to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_open (bool, optional): Sets the tree node open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tselectable (bool, optional): Makes the tree selectable.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_tree_node(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def value_registry(**kwargs):
    if False:
        return 10
    "\t Adds a value registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_value_registry(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def viewport_drawlist(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t A container that is used to present draw items or layers directly to the viewport. By default this will draw to the back of the viewport. Layers and draw items should be added to this widget as children.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tfront (bool, optional): Draws to the front of the view port instead of the back.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_viewport_drawlist(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def viewport_menu_bar(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a menubar to the viewport.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_viewport_menu_bar(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

@contextmanager
def window(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Creates a new window for following items to be added to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tcollapsed (bool, optional): Collapse the window.\n\t\tautosize (bool, optional): Autosized the window to fit it's items.\n\t\tno_resize (bool, optional): Allows for the window size to be changed or fixed.\n\t\tno_title_bar (bool, optional): Title name for the title bar of the window.\n\t\tno_move (bool, optional): Allows for the window's position to be changed or fixed.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars. (window can still scroll with mouse or programmatically)\n\t\tno_collapse (bool, optional): Disable user collapsing window by double-clicking on it.\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear. (off by default)\n\t\tno_focus_on_appearing (bool, optional): Disable taking focus when transitioning from hidden to visible state.\n\t\tno_bring_to_front_on_focus (bool, optional): Disable bringing window to front when taking focus. (e.g. clicking on it or programmatically giving it focus)\n\t\tno_close (bool, optional): Disable user closing the window by removing the close button.\n\t\tno_background (bool, optional): Sets Background and border alpha to transparent.\n\t\tmodal (bool, optional): Fills area behind window according to the theme and disables user ability to interact with anything except the window.\n\t\tpopup (bool, optional): Fills area behind window according to the theme, removes title bar, collapse and close. Window can be closed by selecting area in the background behind the window.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tno_open_over_existing_popup (bool, optional): Don't open if there's already a popup\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\ton_close (Callable, optional): Callback ran when window is closed.\n\t\tid (Union[int, str], optional): (deprecated)\n\tYields:\n\t\tUnion[int, str]\n\t"
    try:
        widget = internal_dpg.add_window(**kwargs)
        internal_dpg.push_container_stack(widget)
        yield widget
    finally:
        internal_dpg.pop_container_stack()

def add_2d_histogram_series(x, y, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a 2d histogram series.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\txbins (int, optional): \n\t\tybins (int, optional): \n\t\txmin_range (float, optional): \n\t\txmax_range (float, optional): \n\t\tymin_range (float, optional): \n\t\tymax_range (float, optional): \n\t\tdensity (bool, optional): \n\t\toutliers (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_2d_histogram_series(x, y, **kwargs)

def add_3d_slider(**kwargs):
    if False:
        return 10
    "\t Adds a 3D box slider.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tmax_x (float, optional): Applies upper limit to slider.\n\t\tmax_y (float, optional): Applies upper limit to slider.\n\t\tmax_z (float, optional): Applies upper limit to slider.\n\t\tmin_x (float, optional): Applies lower limit to slider.\n\t\tmin_y (float, optional): Applies lower limit to slider.\n\t\tmin_z (float, optional): Applies lower limit to slider.\n\t\tscale (float, optional): Size of the widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_3d_slider(**kwargs)

def add_alias(alias, item):
    if False:
        i = 10
        return i + 15
    '\t Adds an alias.\n\n\tArgs:\n\t\talias (str): \n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.add_alias(alias, item)

def add_area_series(x, y, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds an area series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tcontribute_to_bounds (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_area_series(x, y, **kwargs)

def add_bar_series(x, y, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a bar series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tweight (float, optional): \n\t\thorizontal (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_bar_series(x, y, **kwargs)

def add_bool_value(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a bool value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (bool, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_bool_value(**kwargs)

def add_button(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a button.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tsmall (bool, optional): Shrinks the size of the button to the text of the label it contains. Useful for embedding in text.\n\t\tarrow (bool, optional): Displays an arrow in place of the text string. This requires the direction keyword.\n\t\tdirection (int, optional): Sets the cardinal direction for the arrow by using constants mvDir_Left, mvDir_Up, mvDir_Down, mvDir_Right, mvDir_None. Arrow keyword must be set to True.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_button(**kwargs)

def add_candle_series(dates, opens, closes, lows, highs, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a candle series to a plot.\n\n\tArgs:\n\t\tdates (Any): \n\t\topens (Any): \n\t\tcloses (Any): \n\t\tlows (Any): \n\t\thighs (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tbull_color (Union[List[int], Tuple[int, ...]], optional): \n\t\tbear_color (Union[List[int], Tuple[int, ...]], optional): \n\t\tweight (float, optional): \n\t\ttooltip (bool, optional): \n\t\ttime_unit (int, optional): mvTimeUnit_* constants. Default mvTimeUnit_Day.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_candle_series(dates, opens, closes, lows, highs, **kwargs)

def add_char_remap(source, target, **kwargs):
    if False:
        return 10
    "\t Remaps a character.\n\n\tArgs:\n\t\tsource (int): \n\t\ttarget (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_char_remap(source, target, **kwargs)

def add_checkbox(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a checkbox.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (bool, optional): Sets the default value of the checkmark\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_checkbox(**kwargs)

def add_child_window(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds an embedded child window. Will show scrollbars when items do not fit.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tborder (bool, optional): Shows/Hides the border around the sides.\n\t\tautosize_x (bool, optional): Autosize the window to its parents size in x.\n\t\tautosize_y (bool, optional): Autosize the window to its parents size in y.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars (window can still scroll with mouse or programmatically).\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear (off by default).\n\t\tmenubar (bool, optional): Shows/Hides the menubar at the top.\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_child_window(**kwargs)

def add_clipper(**kwargs):
    if False:
        while True:
            i = 10
    "\t Helper to manually clip large list of items. Increases performance by not searching or drawing widgets outside of the clipped region.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_clipper(**kwargs)

def add_collapsing_header(**kwargs):
    if False:
        return 10
    "\t Adds a collapsing header to add items to. Must be closed with the end command.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Adds the ability to hide this widget by pressing the (x) in the top right of widget.\n\t\tdefault_open (bool, optional): Sets the collapseable header open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_collapsing_header(**kwargs)

def add_color_button(default_value=(0, 0, 0, 255), **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a color button.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_alpha (bool, optional): Removes the displayed slider that can change alpha channel.\n\t\tno_border (bool, optional): Disable border around the image.\n\t\tno_drag_drop (bool, optional): Disable ability to drag and drop small preview (color square) to apply colors to other items.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_color_button(default_value, **kwargs)

def add_color_edit(default_value=(0, 0, 0, 255), **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds an RGBA color editor. Left clicking the small color preview will provide a color picker. Click and draging the small color preview will copy the color to be applied on any other color widget.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_alpha (bool, optional): Removes the displayed slider that can change alpha channel.\n\t\tno_picker (bool, optional): Disable picker popup when color square is clicked.\n\t\tno_options (bool, optional): Disable toggling options menu when right-clicking on inputs/small preview.\n\t\tno_small_preview (bool, optional): Disable colored square preview next to the inputs. (e.g. to show only the inputs). This only displays if the side preview is not shown.\n\t\tno_inputs (bool, optional): Disable inputs sliders/text widgets. (e.g. to show only the small preview colored square)\n\t\tno_tooltip (bool, optional): Disable tooltip when hovering the preview.\n\t\tno_label (bool, optional): Disable display of inline text label.\n\t\tno_drag_drop (bool, optional): Disable ability to drag and drop small preview (color square) to apply colors to other items.\n\t\talpha_bar (bool, optional): Show vertical alpha bar/gradient in picker.\n\t\talpha_preview (int, optional): mvColorEdit_AlphaPreviewNone, mvColorEdit_AlphaPreview, or mvColorEdit_AlphaPreviewHalf\n\t\tdisplay_mode (int, optional): mvColorEdit_rgb, mvColorEdit_hsv, or mvColorEdit_hex\n\t\tdisplay_type (int, optional): mvColorEdit_uint8 or mvColorEdit_float\n\t\tinput_mode (int, optional): mvColorEdit_input_rgb or mvColorEdit_input_hsv\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_color_edit(default_value, **kwargs)

def add_color_picker(default_value=(0, 0, 0, 255), **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds an RGB color picker. Right click the color picker for options. Click and drag the color preview to copy the color and drop on any other color widget to apply. Right Click allows the style of the color picker to be changed.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_alpha (bool, optional): Removes the displayed slider that can change alpha channel.\n\t\tno_side_preview (bool, optional): Disable bigger color preview on right side of the picker, use small colored square preview instead , unless small preview is also hidden.\n\t\tno_small_preview (bool, optional): Disable colored square preview next to the inputs. (e.g. to show only the inputs). This only displays if the side preview is not shown.\n\t\tno_inputs (bool, optional): Disable inputs sliders/text widgets. (e.g. to show only the small preview colored square)\n\t\tno_tooltip (bool, optional): Disable tooltip when hovering the preview.\n\t\tno_label (bool, optional): Disable display of inline text label.\n\t\talpha_bar (bool, optional): Show vertical alpha bar/gradient in picker.\n\t\tdisplay_rgb (bool, optional): Override _display_ type among RGB/HSV/Hex.\n\t\tdisplay_hsv (bool, optional): Override _display_ type among RGB/HSV/Hex.\n\t\tdisplay_hex (bool, optional): Override _display_ type among RGB/HSV/Hex.\n\t\tpicker_mode (int, optional): mvColorPicker_bar or mvColorPicker_wheel\n\t\talpha_preview (int, optional): mvColorEdit_AlphaPreviewNone, mvColorEdit_AlphaPreview, or mvColorEdit_AlphaPreviewHalf\n\t\tdisplay_type (int, optional): mvColorEdit_uint8 or mvColorEdit_float\n\t\tinput_mode (int, optional): mvColorEdit_input_rgb or mvColorEdit_input_hsv\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_color_picker(default_value, **kwargs)

def add_color_value(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a color value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_color_value(**kwargs)

def add_colormap(colors, qualitative, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a legend that pairs colors with normalized value 0.0->1.0. Each color will be  This is typically used with a heat series. (ex. [[0, 0, 0, 255], [255, 255, 255, 255]] will be mapped to a soft transition from 0.0-1.0)\n\n\tArgs:\n\t\tcolors (Any): colors that will be mapped to the normalized value 0.0->1.0\n\t\tqualitative (bool): Qualitative will create hard transitions for color boundries across the value range when enabled.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_colormap(colors, qualitative, **kwargs)

def add_colormap_button(default_value=(0, 0, 0, 255), **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a button that a color map can be bound to.\n\n\tArgs:\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_colormap_button(default_value, **kwargs)

def add_colormap_registry(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a colormap registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_colormap_registry(**kwargs)

def add_colormap_scale(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a legend that pairs values with colors. This is typically used with a heat series. \n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tcolormap (Union[int, str], optional): mvPlotColormap_* constants or mvColorMap uuid from a color map registry\n\t\tmin_scale (float, optional): Sets the min number of the color scale. Typically is the same as the min scale from the heat series.\n\t\tmax_scale (float, optional): Sets the max number of the color scale. Typically is the same as the max scale from the heat series.\n\t\tid (Union[int, str], optional): (deprecated)\n\t\tdrag_callback (Callable, optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_colormap_scale(**kwargs)

def add_colormap_slider(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a color slider that a color map can be bound to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\t\tdrag_callback (Callable, optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_colormap_slider(**kwargs)

def add_combo(items=(), **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a combo dropdown that allows a user to select a single option from a drop down window. All items will be shown as selectables on the dropdown.\n\n\tArgs:\n\t\titems (Union[List[str], Tuple[str, ...]], optional): A tuple of items to be shown in the drop down window. Can consist of any combination of types but will convert all items to strings to be shown.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): Sets a selected item from the drop down by specifying the string value.\n\t\tpopup_align_left (bool, optional): Align the contents on the popup toward the left.\n\t\tno_arrow_button (bool, optional): Display the preview box without the square arrow button indicating dropdown activity.\n\t\tno_preview (bool, optional): Display only the square arrow button and not the selected value.\n\t\theight_mode (int, optional): Controlls the number of items shown in the dropdown by the constants mvComboHeight_Small, mvComboHeight_Regular, mvComboHeight_Large, mvComboHeight_Largest\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_combo(items, **kwargs)

def add_custom_series(x, y, channel_count, **kwargs):
    if False:
        return 10
    "\t Adds a custom series to a plot. New in 1.6.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tchannel_count (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\ty1 (Any, optional): \n\t\ty2 (Any, optional): \n\t\ty3 (Any, optional): \n\t\ttooltip (bool, optional): Show tooltip when plot is hovered.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_custom_series(x, y, channel_count, **kwargs)

def add_date_picker(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a data picker.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (dict, optional): \n\t\tlevel (int, optional): Use avaliable constants. mvDatePickerLevel_Day, mvDatePickerLevel_Month, mvDatePickerLevel_Year\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_date_picker(**kwargs)

def add_double4_value(**kwargs):
    if False:
        return 10
    "\t Adds a double value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Any, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_double4_value(**kwargs)

def add_double_value(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a double value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (float, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_double_value(**kwargs)

def add_drag_double(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds drag for a single double value. Useful when drag float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_double(**kwargs)

def add_drag_doublex(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds drag input for a set of double values up to 4. Useful when drag float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Any, optional): \n\t\tsize (int, optional): Number of doubles to be displayed.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_doublex(**kwargs)

def add_drag_float(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds drag for a single float value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_float(**kwargs)

def add_drag_floatx(**kwargs):
    if False:
        return 10
    "\t Adds drag input for a set of float values up to 4. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tsize (int, optional): Number of floats to be displayed.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (float, optional): Applies a limit only to draging entry only.\n\t\tmax_value (float, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_floatx(**kwargs)

def add_drag_int(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds drag for a single int value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (int, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (int, optional): Applies a limit only to draging entry only.\n\t\tmax_value (int, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_int(**kwargs)

def add_drag_intx(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds drag input for a set of int values up to 4. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the drag. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tsize (int, optional): Number of ints to be displayed.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tspeed (float, optional): Sets the sensitivity the float will be modified while dragging.\n\t\tmin_value (int, optional): Applies a limit only to draging entry only.\n\t\tmax_value (int, optional): Applies a limit only to draging entry only.\n\t\tno_input (bool, optional): Disable direct entry methods or Enter key allowing to input text directly into the widget.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_intx(**kwargs)

def add_drag_line(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a drag line to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_value (Any, optional): \n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tshow_label (bool, optional): \n\t\tvertical (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_line(**kwargs)

def add_drag_payload(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t User data payload for drag and drop operations.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdrag_data (Any, optional): Drag data\n\t\tdrop_data (Any, optional): Drop data\n\t\tpayload_type (str, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_payload(**kwargs)

def add_drag_point(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a drag point to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_value (Any, optional): \n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tshow_label (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drag_point(**kwargs)

def add_draw_layer(**kwargs):
    if False:
        while True:
            i = 10
    "\t New in 1.1. Creates a layer useful for grouping drawlist items.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tperspective_divide (bool, optional): New in 1.1. apply perspective divide\n\t\tdepth_clipping (bool, optional): New in 1.1. apply depth clipping\n\t\tcull_mode (int, optional): New in 1.1. culling mode, mvCullMode_* constants. Only works with triangles currently.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_draw_layer(**kwargs)

def add_draw_node(**kwargs):
    if False:
        while True:
            i = 10
    "\t New in 1.1. Creates a drawing node to associate a transformation matrix. Child node matricies will concatenate.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_draw_node(**kwargs)

def add_drawlist(width, height, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a drawing canvas.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_drawlist(width, height, **kwargs)

def add_dynamic_texture(width, height, default_value, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a dynamic texture.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tdefault_value (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_dynamic_texture(width, height, default_value, **kwargs)

def add_error_series(x, y, negative, positive, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds an error series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tnegative (Any): \n\t\tpositive (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcontribute_to_bounds (bool, optional): \n\t\thorizontal (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_error_series(x, y, negative, positive, **kwargs)

def add_file_dialog(**kwargs):
    if False:
        print('Hello World!')
    "\t Displays a file or directory selector depending on keywords. Displays a file dialog by default. Callback will be ran when the file or directory picker is closed. The app_data arguemnt will be populated with information related to the file and directory as a dictionary.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_path (str, optional): Path that the file dialog will default to when opened.\n\t\tdefault_filename (str, optional): Default name that will show in the file name input.\n\t\tfile_count (int, optional): Number of visible files in the dialog.\n\t\tmodal (bool, optional): Forces user interaction with the file selector.\n\t\tdirectory_selector (bool, optional): Shows only directory/paths as options. Allows selection of directory/paths only.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tcancel_callback (Callable, optional): Callback called when cancel button is clicked.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_file_dialog(**kwargs)

def add_file_extension(extension, **kwargs):
    if False:
        while True:
            i = 10
    "\t Creates a file extension filter option in the file dialog.\n\n\tArgs:\n\t\textension (str): Extension that will show as an when the parent is a file dialog.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcustom_text (str, optional): Replaces the displayed text in the drop down for this extension.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): Color for the text that will be shown with specified extensions.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_file_extension(extension, **kwargs)

def add_filter_set(**kwargs):
    if False:
        print('Hello World!')
    "\t Helper to parse and apply text filters (e.g. aaaaa[, bbbbb][, ccccc])\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_filter_set(**kwargs)

def add_float4_value(**kwargs):
    if False:
        return 10
    "\t Adds a float4 value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_float4_value(**kwargs)

def add_float_value(**kwargs):
    if False:
        return 10
    "\t Adds a float value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (float, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_float_value(**kwargs)

def add_float_vect_value(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a float vect value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_float_vect_value(**kwargs)

def add_font(file, size, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds font to a font registry.\n\n\tArgs:\n\t\tfile (str): \n\t\tsize (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tpixel_snapH (bool, optional): Align every glyph to pixel boundary. Useful e.g. if you are merging a non-pixel aligned font with the default font, or rendering text piece-by-piece (e.g. for coloring).\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\t\tdefault_font (bool, optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_font(file, size, **kwargs)

def add_font_chars(chars, **kwargs):
    if False:
        return 10
    "\t Adds specific font characters to a font.\n\n\tArgs:\n\t\tchars (Union[List[int], Tuple[int, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_font_chars(chars, **kwargs)

def add_font_range(first_char, last_char, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a range of font characters to a font.\n\n\tArgs:\n\t\tfirst_char (int): \n\t\tlast_char (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_font_range(first_char, last_char, **kwargs)

def add_font_range_hint(hint, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a range of font characters (mvFontRangeHint_ constants).\n\n\tArgs:\n\t\thint (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_font_range_hint(hint, **kwargs)

def add_font_registry(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a font registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_font_registry(**kwargs)

def add_group(**kwargs):
    if False:
        while True:
            i = 10
    "\t Creates a group that other widgets can belong to. The group allows item commands to be issued for all of its members.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\thorizontal (bool, optional): Forces child widgets to be added in a horizontal layout.\n\t\thorizontal_spacing (float, optional): Spacing for the horizontal layout.\n\t\txoffset (float, optional): Offset from containing window x item location within group.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_group(**kwargs)

def add_handler_registry(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_handler_registry(**kwargs)

def add_heat_series(x, rows, cols, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a heat series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\trows (int): \n\t\tcols (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tscale_min (float, optional): Sets the color scale min. Typically paired with the color scale widget scale_min.\n\t\tscale_max (float, optional): Sets the color scale max. Typically paired with the color scale widget scale_max.\n\t\tbounds_min (Any, optional): \n\t\tbounds_max (Any, optional): \n\t\tformat (str, optional): \n\t\tcontribute_to_bounds (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_heat_series(x, rows, cols, **kwargs)

def add_histogram_series(x, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a histogram series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tbins (int, optional): \n\t\tbar_scale (float, optional): \n\t\tmin_range (float, optional): \n\t\tmax_range (float, optional): \n\t\tcumlative (bool, optional): \n\t\tdensity (bool, optional): \n\t\toutliers (bool, optional): \n\t\tcontribute_to_bounds (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_histogram_series(x, **kwargs)

def add_hline_series(x, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds an infinite horizontal line series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_hline_series(x, **kwargs)

def add_image(texture_tag, **kwargs):
    if False:
        return 10
    "\t Adds an image from a specified texture. uv_min and uv_max represent the normalized texture coordinates of the original image that will be shown. Using range (0.0,0.0)->(1.0,1.0) for texture coordinates will generally display the entire texture.\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): The texture_tag should come from a texture that was added to a texture registry.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\ttint_color (Union[List[float], Tuple[float, ...]], optional): Applies a color tint to the entire texture.\n\t\tborder_color (Union[List[float], Tuple[float, ...]], optional): Displays a border of the specified color around the texture. If the theme style has turned off the border it will not be shown.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates min point.\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates max point.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_image(texture_tag, **kwargs)

def add_image_button(texture_tag, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds an button with a texture. uv_min and uv_max represent the normalized texture coordinates of the original image that will be shown. Using range (0.0,0.0)->(1.0,1.0) texture coordinates will generally display the entire texture\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): The texture_tag should come from a texture that was added to a texture registry.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tframe_padding (int, optional): Empty space around the outside of the texture. Button will show around the texture.\n\t\ttint_color (Union[List[float], Tuple[float, ...]], optional): Applies a color tint to the entire texture.\n\t\tbackground_color (Union[List[float], Tuple[float, ...]], optional): Displays a border of the specified color around the texture.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates min point.\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): Normalized texture coordinates max point.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_image_button(texture_tag, **kwargs)

def add_image_series(texture_tag, bounds_min, bounds_max, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds an image series to a plot.\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): \n\t\tbounds_min (Any): \n\t\tbounds_max (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): normalized texture coordinates\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): normalized texture coordinates\n\t\ttint_color (Union[List[int], Tuple[int, ...]], optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_image_series(texture_tag, bounds_min, bounds_max, **kwargs)

def add_input_double(**kwargs):
    if False:
        return 10
    "\t Adds input for an double. Useful when input float is not accurate enough. +/- buttons can be activated by setting the value of step.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input. By default this limits the step buttons. Use min_clamped to limit manual input.\n\t\tmax_value (float, optional): Value for upper limit of input. By default this limits the step buttons. Use max_clamped to limit manual input.\n\t\tstep (float, optional): Increment to change value by when the step buttons are pressed. Setting this and step_fast to a value of 0 or less will turn off step buttons.\n\t\tstep_fast (float, optional): Increment to change value by when ctrl + step buttons are pressed. Setting this and step to a value of 0 or less will turn off step buttons.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_input_double(**kwargs)

def add_input_doublex(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds multi double input for up to 4 double values. Useful when input float mulit is not accurate enough.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Any, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input for each cell. Use min_clamped to turn on.\n\t\tmax_value (float, optional): Value for upper limit of input for each cell. Use max_clamped to turn on.\n\t\tsize (int, optional): Number of components displayed for input.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_input_doublex(**kwargs)

def add_input_float(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds input for an float. +/- buttons can be activated by setting the value of step.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input. By default this limits the step buttons. Use min_clamped to limit manual input.\n\t\tmax_value (float, optional): Value for upper limit of input. By default this limits the step buttons. Use max_clamped to limit manual input.\n\t\tstep (float, optional): Increment to change value by when the step buttons are pressed. Setting this and step_fast to a value of 0 or less will turn off step buttons.\n\t\tstep_fast (float, optional): Increment to change value by when ctrl + step buttons are pressed. Setting this and step to a value of 0 or less will turn off step buttons.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_input_float(**kwargs)

def add_input_floatx(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds multi float input for up to 4 float values.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tmin_value (float, optional): Value for lower limit of input for each cell. Use min_clamped to turn on.\n\t\tmax_value (float, optional): Value for upper limit of input for each cell. Use max_clamped to turn on.\n\t\tsize (int, optional): Number of components displayed for input.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_input_floatx(**kwargs)

def add_input_int(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds input for an int. +/- buttons can be activated by setting the value of step.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (int, optional): \n\t\tmin_value (int, optional): Value for lower limit of input. By default this limits the step buttons. Use min_clamped to limit manual input.\n\t\tmax_value (int, optional): Value for upper limit of input. By default this limits the step buttons. Use max_clamped to limit manual input.\n\t\tstep (int, optional): Increment to change value by when the step buttons are pressed. Setting this and step_fast to a value of 0 or less will turn off step buttons.\n\t\tstep_fast (int, optional): Increment to change value by when ctrl + step buttons are pressed. Setting this and step to a value of 0 or less will turn off step buttons.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_input_int(**kwargs)

def add_input_intx(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds multi int input for up to 4 integer values.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tmin_value (int, optional): Value for lower limit of input for each cell. Use min_clamped to turn on.\n\t\tmax_value (int, optional): Value for upper limit of input for each cell. Use max_clamped to turn on.\n\t\tsize (int, optional): Number of components displayed for input.\n\t\tmin_clamped (bool, optional): Activates and deactivates the enforcment of min_value.\n\t\tmax_clamped (bool, optional): Activates and deactivates the enforcment of max_value.\n\t\ton_enter (bool, optional): Only runs callback on enter.\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_input_intx(**kwargs)

def add_input_text(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds input for text.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): \n\t\thint (str, optional): Displayed only when value is an empty string. Will reappear if input value is set to empty string. Will not show if default value is anything other than default empty string.\n\t\tmultiline (bool, optional): Allows for multiline text input.\n\t\tno_spaces (bool, optional): Filter out spaces and tabs.\n\t\tuppercase (bool, optional): Automatically make all inputs uppercase.\n\t\ttab_input (bool, optional): Allows tabs to be input into the string value instead of changing item focus.\n\t\tdecimal (bool, optional): Only allow characters 0123456789.+-*/\n\t\thexadecimal (bool, optional): Only allow characters 0123456789ABCDEFabcdef\n\t\treadonly (bool, optional): Activates read only mode where no text can be input but text can still be highlighted.\n\t\tpassword (bool, optional): Display all input characters as '*'.\n\t\tscientific (bool, optional): Only allow characters 0123456789.+-*/eE (Scientific notation input)\n\t\ton_enter (bool, optional): Only runs callback on enter key press.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_input_text(**kwargs)

def add_int4_value(**kwargs):
    if False:
        return 10
    "\t Adds a int4 value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_int4_value(**kwargs)

def add_int_value(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a int value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (int, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_int_value(**kwargs)

def add_item_activated_handler(**kwargs):
    if False:
        return 10
    "\t Adds a activated handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_activated_handler(**kwargs)

def add_item_active_handler(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a active handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_active_handler(**kwargs)

def add_item_clicked_handler(button=-1, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a clicked handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_clicked_handler(button, **kwargs)

def add_item_deactivated_after_edit_handler(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a deactivated after edit handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_deactivated_after_edit_handler(**kwargs)

def add_item_deactivated_handler(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a deactivated handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_deactivated_handler(**kwargs)

def add_item_double_clicked_handler(button=-1, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a double click handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_double_clicked_handler(button, **kwargs)

def add_item_edited_handler(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds an edited handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_edited_handler(**kwargs)

def add_item_focus_handler(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a focus handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_focus_handler(**kwargs)

def add_item_handler_registry(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds an item handler registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_handler_registry(**kwargs)

def add_item_hover_handler(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a hover handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_hover_handler(**kwargs)

def add_item_resize_handler(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a resize handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_resize_handler(**kwargs)

def add_item_toggled_open_handler(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a togged open handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_toggled_open_handler(**kwargs)

def add_item_visible_handler(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a visible handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_item_visible_handler(**kwargs)

def add_key_down_handler(key=-1, **kwargs):
    if False:
        return 10
    "\t Adds a key down handler.\n\n\tArgs:\n\t\tkey (int, optional): Submits callback for all keys\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_key_down_handler(key, **kwargs)

def add_key_press_handler(key=-1, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a key press handler.\n\n\tArgs:\n\t\tkey (int, optional): Submits callback for all keys\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_key_press_handler(key, **kwargs)

def add_key_release_handler(key=-1, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a key release handler.\n\n\tArgs:\n\t\tkey (int, optional): Submits callback for all keys\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_key_release_handler(key, **kwargs)

def add_knob_float(**kwargs):
    if False:
        return 10
    "\t Adds a knob that rotates based on change in x mouse position.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tmin_value (float, optional): Applies lower limit to value.\n\t\tmax_value (float, optional): Applies upper limit to value.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_knob_float(**kwargs)

def add_line_series(x, y, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a line series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_line_series(x, y, **kwargs)

def add_listbox(items=(), **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a listbox. If height is not large enough to show all items a scroll bar will appear.\n\n\tArgs:\n\t\titems (Union[List[str], Tuple[str, ...]], optional): A tuple of items to be shown in the listbox. Can consist of any combination of types. All items will be displayed as strings.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): String value of the item that will be selected by default.\n\t\tnum_items (int, optional): Expands the height of the listbox to show specified number of items.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_listbox(items, **kwargs)

def add_loading_indicator(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a rotating animated loading symbol.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tstyle (int, optional): 0 is rotating dots style, 1 is rotating bar style.\n\t\tcircle_count (int, optional): Number of dots show if dots or size of circle if circle.\n\t\tspeed (float, optional): Speed the anamation will rotate.\n\t\tradius (float, optional): Radius size of the loading indicator.\n\t\tthickness (float, optional): Thickness of the circles or line.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): Color of the growing center circle.\n\t\tsecondary_color (Union[List[int], Tuple[int, ...]], optional): Background of the dots in dot mode.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_loading_indicator(**kwargs)

def add_menu(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a menu to an existing menu bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_menu(**kwargs)

def add_menu_bar(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a menu bar to a window.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_menu_bar(**kwargs)

def add_menu_item(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a menu item to an existing menu. Menu items act similar to selectables and has a bool value. When placed in a menu the checkmark will reflect its value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (bool, optional): This value also controls the checkmark when shown.\n\t\tshortcut (str, optional): Displays text on the menu item. Typically used to show a shortcut key command.\n\t\tcheck (bool, optional): Displays a checkmark on the menu item when it is selected and placed in a menu.\n\t\tid (Union[int, str], optional): (deprecated)\n\t\tdrag_callback (Callable, optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_menu_item(**kwargs)

def add_mouse_click_handler(button=-1, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a mouse click handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_mouse_click_handler(button, **kwargs)

def add_mouse_double_click_handler(button=-1, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a mouse double click handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_mouse_double_click_handler(button, **kwargs)

def add_mouse_down_handler(button=-1, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a mouse down handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_mouse_down_handler(button, **kwargs)

def add_mouse_drag_handler(button=-1, threshold=10.0, **kwargs):
    if False:
        return 10
    "\t Adds a mouse drag handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tthreshold (float, optional): The threshold the mouse must be dragged before the callback is ran\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_mouse_drag_handler(button, threshold, **kwargs)

def add_mouse_move_handler(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a mouse move handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_mouse_move_handler(**kwargs)

def add_mouse_release_handler(button=-1, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a mouse release handler.\n\n\tArgs:\n\t\tbutton (int, optional): Submits callback for all mouse buttons\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_mouse_release_handler(button, **kwargs)

def add_mouse_wheel_handler(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a mouse wheel handler.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_mouse_wheel_handler(**kwargs)

def add_node(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a node to a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdraggable (bool, optional): Allow node to be draggable.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_node(**kwargs)

def add_node_attribute(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a node attribute to a node.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tattribute_type (int, optional): mvNode_Attr_Input, mvNode_Attr_Output, or mvNode_Attr_Static.\n\t\tshape (int, optional): Pin shape.\n\t\tcategory (str, optional): Category\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_node_attribute(**kwargs)

def add_node_editor(**kwargs):
    if False:
        return 10
    "\t Adds a node editor.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdelink_callback (Callable, optional): Callback ran when a link is detached.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tminimap (bool, optional): Shows or hides the Minimap. New in 1.6.\n\t\tminimap_location (int, optional): mvNodeMiniMap_Location_* constants. New in 1.6.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_node_editor(**kwargs)

def add_node_link(attr_1, attr_2, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a node link between 2 node attributes.\n\n\tArgs:\n\t\tattr_1 (Union[int, str]): \n\t\tattr_2 (Union[int, str]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_node_link(attr_1, attr_2, **kwargs)

def add_pie_series(x, y, radius, values, labels, **kwargs):
    if False:
        return 10
    "\t Adds an pie series to a plot.\n\n\tArgs:\n\t\tx (float): \n\t\ty (float): \n\t\tradius (float): \n\t\tvalues (Any): \n\t\tlabels (Union[List[str], Tuple[str, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tformat (str, optional): \n\t\tangle (float, optional): \n\t\tnormalize (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_pie_series(x, y, radius, values, labels, **kwargs)

def add_plot(**kwargs):
    if False:
        return 10
    "\t Adds a plot which is used to hold series, and can be drawn to with draw commands.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_title (bool, optional): the plot title will not be displayed\n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_box_select (bool, optional): the user will not be able to box-select with right-click drag\n\t\tno_mouse_pos (bool, optional): the mouse position, in plot coordinates, will not be displayed inside of the plot\n\t\tno_highlight (bool, optional): plot items will not be highlighted when their legend entry is hovered\n\t\tno_child (bool, optional): a child window region will not be used to capture mouse scroll (can boost performance for single ImGui window applications)\n\t\tquery (bool, optional): the user will be able to draw query rects with middle - mouse or CTRL + right - click drag\n\t\tcrosshairs (bool, optional): the default mouse cursor will be replaced with a crosshair when hovered\n\t\tanti_aliased (bool, optional): plot lines will be software anti-aliased (not recommended for high density plots, prefer MSAA)\n\t\tequal_aspects (bool, optional): primary x and y axes will be constrained to have the same units/pixel (does not apply to auxiliary y-axes)\n\t\tuse_local_time (bool, optional): axis labels will be formatted for your timezone when\n\t\tuse_ISO8601 (bool, optional): dates will be formatted according to ISO 8601 where applicable (e.g. YYYY-MM-DD, YYYY-MM, --MM-DD, etc.)\n\t\tuse_24hour_clock (bool, optional): times will be formatted using a 24 hour clock\n\t\tpan_button (int, optional): enables panning when held\n\t\tpan_mod (int, optional): optional modifier that must be held for panning\n\t\tfit_button (int, optional): fits visible data when double clicked\n\t\tcontext_menu_button (int, optional): opens plot context menu (if enabled) when clicked\n\t\tbox_select_button (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_mod (int, optional): begins box selection when pressed and confirms selection when released\n\t\tbox_select_cancel_button (int, optional): cancels active box selection when pressed\n\t\tquery_button (int, optional): begins query selection when pressed and end query selection when released\n\t\tquery_mod (int, optional): optional modifier that must be held for query selection\n\t\tquery_toggle_mod (int, optional): when held, active box selections turn into queries\n\t\thorizontal_mod (int, optional): expands active box selection/query horizontally to plot edge when held\n\t\tvertical_mod (int, optional): expands active box selection/query vertically to plot edge when held\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_plot(**kwargs)

def add_plot_annotation(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds an annotation to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdefault_value (Any, optional): \n\t\toffset (Union[List[float], Tuple[float, ...]], optional): \n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tclamped (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_plot_annotation(**kwargs)

def add_plot_axis(axis, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds an axis to a plot.\n\n\tArgs:\n\t\taxis (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tno_gridlines (bool, optional): \n\t\tno_tick_marks (bool, optional): \n\t\tno_tick_labels (bool, optional): \n\t\tlog_scale (bool, optional): \n\t\tinvert (bool, optional): \n\t\tlock_min (bool, optional): \n\t\tlock_max (bool, optional): \n\t\ttime (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_plot_axis(axis, **kwargs)

def add_plot_legend(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a plot legend to a plot.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tlocation (int, optional): location, mvPlot_Location_*\n\t\thorizontal (bool, optional): \n\t\toutside (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_plot_legend(**kwargs)

def add_progress_bar(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a progress bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\toverlay (str, optional): Overlayed text onto the bar that typically used to display the value of the progress.\n\t\tdefault_value (float, optional): Normalized value to fill the bar from 0.0 to 1.0.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_progress_bar(**kwargs)

def add_radio_button(items=(), **kwargs):
    if False:
        return 10
    "\t Adds a set of radio buttons. If items keyword is empty, nothing will be shown.\n\n\tArgs:\n\t\titems (Union[List[str], Tuple[str, ...]], optional): A tuple of items to be shown as radio options. Can consist of any combination of types. All types will be shown as strings.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (str, optional): Default selected radio option. Set by using the string value of the item.\n\t\thorizontal (bool, optional): Displays the radio options horizontally.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_radio_button(items, **kwargs)

def add_raw_texture(width, height, default_value, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a raw texture.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tdefault_value (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tformat (int, optional): Data format.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_raw_texture(width, height, default_value, **kwargs)

def add_scatter_series(x, y, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a scatter series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_scatter_series(x, y, **kwargs)

def add_selectable(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a selectable. Similar to a button but can indicate its selected state.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (bool, optional): \n\t\tspan_columns (bool, optional): Forces the selectable to span the width of all columns if placed in a table.\n\t\tdisable_popup_close (bool, optional): Disable closing a modal or popup window.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_selectable(**kwargs)

def add_separator(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a horizontal line separator.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_separator(**kwargs)

def add_series_value(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a plot series value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (Any, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_series_value(**kwargs)

def add_shade_series(x, y1, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a shade series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty1 (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\ty2 (Any, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_shade_series(x, y1, **kwargs)

def add_simple_plot(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a simple plot for visualization of a 1 dimensional set of values.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\toverlay (str, optional): overlays text (similar to a plot title)\n\t\thistogram (bool, optional): \n\t\tautosize (bool, optional): \n\t\tmin_scale (float, optional): \n\t\tmax_scale (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_simple_plot(**kwargs)

def add_slider_double(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds slider for a single double value. Useful when slider float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tvertical (bool, optional): Sets orientation of the slidebar and slider to vertical.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_slider_double(**kwargs)

def add_slider_doublex(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds multi slider for up to 4 double values. Usueful for when multi slide float is not accurate enough. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Any, optional): \n\t\tsize (int, optional): Number of doubles to be displayed.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_slider_doublex(**kwargs)

def add_slider_float(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds slider for a single float value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (float, optional): \n\t\tvertical (bool, optional): Sets orientation of the slidebar and slider to vertical.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the float will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_slider_float(**kwargs)

def add_slider_floatx(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds multi slider for up to 4 float values. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[float], Tuple[float, ...]], optional): \n\t\tsize (int, optional): Number of floats to be displayed.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (float, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (float, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_slider_floatx(**kwargs)

def add_slider_int(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds slider for a single int value. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (int, optional): \n\t\tvertical (bool, optional): Sets orientation of the slidebar and slider to vertical.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (int, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (int, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_slider_int(**kwargs)

def add_slider_intx(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds multi slider for up to 4 int values. Directly entry can be done with double click or CTRL+Click. Min and Max alone are a soft limit for the slider. Use clamped keyword to also apply limits to the direct entry modes.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (Union[List[int], Tuple[int, ...]], optional): \n\t\tsize (int, optional): Number of ints to be displayed.\n\t\tno_input (bool, optional): Disable direct entry methods double-click or ctrl+click or Enter key allowing to input text directly into the item.\n\t\tclamped (bool, optional): Applies the min and max limits to direct entry methods also such as double click and CTRL+Click.\n\t\tmin_value (int, optional): Applies a limit only to sliding entry only.\n\t\tmax_value (int, optional): Applies a limit only to sliding entry only.\n\t\tformat (str, optional): Determines the format the int will be displayed as use python string formatting.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_slider_intx(**kwargs)

def add_spacer(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a spacer item that can be used to help with layouts or can be used as a placeholder item.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_spacer(**kwargs)

def add_stage(**kwargs):
    if False:
        return 10
    "\t Adds a stage.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_stage(**kwargs)

def add_stair_series(x, y, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a stair series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_stair_series(x, y, **kwargs)

def add_static_texture(width, height, default_value, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a static texture.\n\n\tArgs:\n\t\twidth (int): \n\t\theight (int): \n\t\tdefault_value (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_static_texture(width, height, default_value, **kwargs)

def add_stem_series(x, y, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a stem series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\ty (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_stem_series(x, y, **kwargs)

def add_string_value(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a string value.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tdefault_value (str, optional): \n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_string_value(**kwargs)

def add_subplots(rows, columns, **kwargs):
    if False:
        return 10
    "\t Adds a collection of plots.\n\n\tArgs:\n\t\trows (int): \n\t\tcolumns (int): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\trow_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tcolumn_ratios (Union[List[float], Tuple[float, ...]], optional): \n\t\tno_title (bool, optional): \n\t\tno_menus (bool, optional): the user will not be able to open context menus with right-click\n\t\tno_resize (bool, optional): resize splitters between subplot cells will be not be provided\n\t\tno_align (bool, optional): subplot edges will not be aligned vertically or horizontally\n\t\tlink_rows (bool, optional): link the y-axis limits of all plots in each row (does not apply auxiliary y-axes)\n\t\tlink_columns (bool, optional): link the x-axis limits of all plots in each column\n\t\tlink_all_x (bool, optional): link the x-axis limits in every plot in the subplot\n\t\tlink_all_y (bool, optional): link the y-axis limits in every plot in the subplot (does not apply to auxiliary y-axes)\n\t\tcolumn_major (bool, optional): subplots are added in column major order instead of the default row major order\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_subplots(rows, columns, **kwargs)

def add_tab(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a tab to a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tclosable (bool, optional): Creates a button on the tab that can hide the tab.\n\t\tno_tooltip (bool, optional): Disable tooltip for the given tab.\n\t\torder_mode (bool, optional): set using a constant: mvTabOrder_Reorderable: allows reordering, mvTabOrder_Fixed: fixed ordering, mvTabOrder_Leading: adds tab to front, mvTabOrder_Trailing: adds tab to back\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_tab(**kwargs)

def add_tab_bar(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\treorderable (bool, optional): Allows for the user to change the order of the tabs.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_tab_bar(**kwargs)

def add_tab_button(**kwargs):
    if False:
        return 10
    "\t Adds a tab button to a tab bar.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tno_reorder (bool, optional): Disable reordering this tab or having another tab cross over this tab. Fixes the position of this tab in relation to the order of neighboring tabs at start. \n\t\tleading (bool, optional): Enforce the tab position to the left of the tab bar (after the tab list popup button).\n\t\ttrailing (bool, optional): Enforce the tab position to the right of the tab bar (before the scrolling buttons).\n\t\tno_tooltip (bool, optional): Disable tooltip for the given tab.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_tab_button(**kwargs)

def add_table(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\theader_row (bool, optional): show headers at the top of the columns\n\t\tclipper (bool, optional): Use clipper (rows must be same height).\n\t\tinner_width (int, optional): \n\t\tpolicy (int, optional): \n\t\tfreeze_rows (int, optional): \n\t\tfreeze_columns (int, optional): \n\t\tsort_multi (bool, optional): Hold shift when clicking headers to sort on multiple column.\n\t\tsort_tristate (bool, optional): Allow no sorting, disable default sorting.\n\t\tresizable (bool, optional): Enable resizing columns\n\t\treorderable (bool, optional): Enable reordering columns in header row (need calling TableSetupColumn() + TableHeadersRow() to display headers)\n\t\thideable (bool, optional): Enable hiding/disabling columns in context menu.\n\t\tsortable (bool, optional): Enable sorting. Call TableGetSortSpecs() to obtain sort specs. Also see ImGuiTableFlags_SortMulti and ImGuiTableFlags_SortTristate.\n\t\tcontext_menu_in_body (bool, optional): Right-click on columns body/contents will display table context menu. By default it is available in TableHeadersRow().\n\t\trow_background (bool, optional): Set each RowBg color with ImGuiCol_TableRowBg or ImGuiCol_TableRowBgAlt (equivalent of calling TableSetBgColor with ImGuiTableBgFlags_RowBg0 on each row manually)\n\t\tborders_innerH (bool, optional): Draw horizontal borders between rows.\n\t\tborders_outerH (bool, optional): Draw horizontal borders at the top and bottom.\n\t\tborders_innerV (bool, optional): Draw vertical borders between columns.\n\t\tborders_outerV (bool, optional): Draw vertical borders on the left and right sides.\n\t\tno_host_extendX (bool, optional): Make outer width auto-fit to columns, overriding outer_size.x value. Only available when ScrollX/ScrollY are disabled and Stretch columns are not used.\n\t\tno_host_extendY (bool, optional): Make outer height stop exactly at outer_size.y (prevent auto-extending table past the limit). Only available when ScrollX/ScrollY are disabled. Data below the limit will be clipped and not visible.\n\t\tno_keep_columns_visible (bool, optional): Disable keeping column always minimally visible when ScrollX is off and table gets too small. Not recommended if columns are resizable.\n\t\tprecise_widths (bool, optional): Disable distributing remainder width to stretched columns (width allocation on a 100-wide table with 3 columns: Without this flag: 33,33,34. With this flag: 33,33,33). With larger number of columns, resizing will appear to be less smooth.\n\t\tno_clip (bool, optional): Disable clipping rectangle for every individual columns.\n\t\tpad_outerX (bool, optional): Default if BordersOuterV is on. Enable outer-most padding. Generally desirable if you have headers.\n\t\tno_pad_outerX (bool, optional): Default if BordersOuterV is off. Disable outer-most padding.\n\t\tno_pad_innerX (bool, optional): Disable inner padding between columns (double inner padding if BordersOuterV is on, single inner padding if BordersOuterV is off).\n\t\tscrollX (bool, optional): Enable horizontal scrolling. Require 'outer_size' parameter of BeginTable() to specify the container size. Changes default sizing policy. Because this create a child window, ScrollY is currently generally recommended when using ScrollX.\n\t\tscrollY (bool, optional): Enable vertical scrolling.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_table(**kwargs)

def add_table_cell(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a table.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_table_cell(**kwargs)

def add_table_column(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a table column.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tenabled (bool, optional): Turns off functionality of widget and applies the disabled theme.\n\t\tinit_width_or_weight (float, optional): \n\t\tdefault_hide (bool, optional): Default as a hidden/disabled column.\n\t\tdefault_sort (bool, optional): Default as a sorting column.\n\t\twidth_stretch (bool, optional): Column will stretch. Preferable with horizontal scrolling disabled (default if table sizing policy is _SizingStretchSame or _SizingStretchProp).\n\t\twidth_fixed (bool, optional): Column will not stretch. Preferable with horizontal scrolling enabled (default if table sizing policy is _SizingFixedFit and table is resizable).\n\t\tno_resize (bool, optional): Disable manual resizing.\n\t\tno_reorder (bool, optional): Disable manual reordering this column, this will also prevent other columns from crossing over this column.\n\t\tno_hide (bool, optional): Disable ability to hide/disable this column.\n\t\tno_clip (bool, optional): Disable clipping for this column (all NoClip columns will render in a same draw command).\n\t\tno_sort (bool, optional): Disable ability to sort on this field (even if ImGuiTableFlags_Sortable is set on the table).\n\t\tno_sort_ascending (bool, optional): Disable ability to sort in the ascending direction.\n\t\tno_sort_descending (bool, optional): Disable ability to sort in the descending direction.\n\t\tno_header_width (bool, optional): Disable header text width contribution to automatic column width.\n\t\tprefer_sort_ascending (bool, optional): Make the initial sort direction Ascending when first sorting on this column (default).\n\t\tprefer_sort_descending (bool, optional): Make the initial sort direction Descending when first sorting on this column.\n\t\tindent_enable (bool, optional): Use current Indent value when entering cell (default for column 0).\n\t\tindent_disable (bool, optional): Ignore current Indent value when entering cell (default for columns > 0). Indentation changes _within_ the cell will still be honored.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_table_column(**kwargs)

def add_table_row(**kwargs):
    if False:
        return 10
    "\t Adds a table row.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\theight (int, optional): Height of the item.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_table_row(**kwargs)

def add_template_registry(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a template registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_template_registry(**kwargs)

def add_text(default_value='', **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds text. Text can have an optional label that will display to the right of the text.\n\n\tArgs:\n\t\tdefault_value (str, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\twrap (int, optional): Number of pixels from the start of the item until wrapping starts.\n\t\tbullet (bool, optional): Places a bullet to the left of the text.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): Color of the text (rgba).\n\t\tshow_label (bool, optional): Displays the label to the right of the text.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_text(default_value, **kwargs)

def add_text_point(x, y, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a label series to a plot.\n\n\tArgs:\n\t\tx (float): \n\t\ty (float): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tx_offset (int, optional): \n\t\ty_offset (int, optional): \n\t\tvertical (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_text_point(x, y, **kwargs)

def add_texture_registry(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a dynamic texture.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_texture_registry(**kwargs)

def add_theme(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a theme.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\t\tdefault_theme (bool, optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_theme(**kwargs)

def add_theme_color(target=0, value=(0, 0, 0, 255), **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a theme color.\n\n\tArgs:\n\t\ttarget (int, optional): \n\t\tvalue (Union[List[int], Tuple[int, ...]], optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcategory (int, optional): Options include mvThemeCat_Core, mvThemeCat_Plots, mvThemeCat_Nodes.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_theme_color(target, value, **kwargs)

def add_theme_component(item_type=0, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a theme component.\n\n\tArgs:\n\t\titem_type (int, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tenabled_state (bool, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_theme_component(item_type, **kwargs)

def add_theme_style(target=0, x=1.0, y=-1.0, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds a theme style.\n\n\tArgs:\n\t\ttarget (int, optional): \n\t\tx (float, optional): \n\t\ty (float, optional): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tcategory (int, optional): Options include mvThemeCat_Core, mvThemeCat_Plots, mvThemeCat_Nodes.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_theme_style(target, x, y, **kwargs)

def add_time_picker(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a time picker.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tcallback (Callable, optional): Registers a callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_value (dict, optional): \n\t\thour24 (bool, optional): Show 24 hour clock instead of 12 hour.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_time_picker(**kwargs)

def add_tooltip(parent, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a tooltip window.\n\n\tArgs:\n\t\tparent (Union[int, str]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay (float, optional): Activation delay: time, in seconds, during which the mouse should stay still in order to display the tooltip.  May be zero for instant activation.\n\t\thide_on_activity (bool, optional): Hide the tooltip if the user has moved the mouse.  If False, the tooltip will follow mouse pointer.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_tooltip(parent, **kwargs)

def add_tree_node(**kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a tree node to add items to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tpayload_type (str, optional): Sender string type must be the same as the target for the target to run the payload_callback.\n\t\tdrag_callback (Callable, optional): Registers a drag callback for drag and drop.\n\t\tdrop_callback (Callable, optional): Registers a drop callback for drag and drop.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\ttracked (bool, optional): Scroll tracking\n\t\ttrack_offset (float, optional): 0.0f:top, 0.5f:center, 1.0f:bottom\n\t\tdefault_open (bool, optional): Sets the tree node open by default.\n\t\topen_on_double_click (bool, optional): Need double-click to open node.\n\t\topen_on_arrow (bool, optional): Only open when clicking on the arrow part.\n\t\tleaf (bool, optional): No collapsing, no arrow (use as a convenience for leaf nodes).\n\t\tbullet (bool, optional): Display a bullet instead of arrow.\n\t\tselectable (bool, optional): Makes the tree selectable.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_tree_node(**kwargs)

def add_value_registry(**kwargs):
    if False:
        print('Hello World!')
    "\t Adds a value registry.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_value_registry(**kwargs)

def add_viewport_drawlist(**kwargs):
    if False:
        return 10
    "\t A container that is used to present draw items or layers directly to the viewport. By default this will draw to the back of the viewport. Layers and draw items should be added to this widget as children.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tfilter_key (str, optional): Used by filter widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tfront (bool, optional): Draws to the front of the view port instead of the back.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_viewport_drawlist(**kwargs)

def add_viewport_menu_bar(**kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a menubar to the viewport.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_viewport_menu_bar(**kwargs)

def add_vline_series(x, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds an infinite vertical line series to a plot.\n\n\tArgs:\n\t\tx (Any): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tsource (Union[int, str], optional): Overrides 'id' as value storage key.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_vline_series(x, **kwargs)

def add_window(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Creates a new window for following items to be added to.\n\n\tArgs:\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\twidth (int, optional): Width of the item.\n\t\theight (int, optional): Height of the item.\n\t\tindent (int, optional): Offsets the widget to the right the specified number multiplied by the indent style.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tpos (Union[List[int], Tuple[int, ...]], optional): Places the item relative to window coordinates, [0,0] is top left.\n\t\tdelay_search (bool, optional): Delays searching container for specified items until the end of the app. Possible optimization when a container has many children that are not accessed often.\n\t\tmin_size (Union[List[int], Tuple[int, ...]], optional): Minimum window size.\n\t\tmax_size (Union[List[int], Tuple[int, ...]], optional): Maximum window size.\n\t\tmenubar (bool, optional): Shows or hides the menubar.\n\t\tcollapsed (bool, optional): Collapse the window.\n\t\tautosize (bool, optional): Autosized the window to fit it's items.\n\t\tno_resize (bool, optional): Allows for the window size to be changed or fixed.\n\t\tno_title_bar (bool, optional): Title name for the title bar of the window.\n\t\tno_move (bool, optional): Allows for the window's position to be changed or fixed.\n\t\tno_scrollbar (bool, optional):  Disable scrollbars. (window can still scroll with mouse or programmatically)\n\t\tno_collapse (bool, optional): Disable user collapsing window by double-clicking on it.\n\t\thorizontal_scrollbar (bool, optional): Allow horizontal scrollbar to appear. (off by default)\n\t\tno_focus_on_appearing (bool, optional): Disable taking focus when transitioning from hidden to visible state.\n\t\tno_bring_to_front_on_focus (bool, optional): Disable bringing window to front when taking focus. (e.g. clicking on it or programmatically giving it focus)\n\t\tno_close (bool, optional): Disable user closing the window by removing the close button.\n\t\tno_background (bool, optional): Sets Background and border alpha to transparent.\n\t\tmodal (bool, optional): Fills area behind window according to the theme and disables user ability to interact with anything except the window.\n\t\tpopup (bool, optional): Fills area behind window according to the theme, removes title bar, collapse and close. Window can be closed by selecting area in the background behind the window.\n\t\tno_saved_settings (bool, optional): Never load/save settings in .ini file.\n\t\tno_open_over_existing_popup (bool, optional): Don't open if there's already a popup\n\t\tno_scroll_with_mouse (bool, optional): Disable user vertically scrolling with mouse wheel.\n\t\ton_close (Callable, optional): Callback ran when window is closed.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.add_window(**kwargs)

def apply_transform(item, transform):
    if False:
        for i in range(10):
            print('nop')
    '\t New in 1.1. Applies a transformation matrix to a layer.\n\n\tArgs:\n\t\titem (Union[int, str]): Drawing node to apply transform to.\n\t\ttransform (Any): Transformation matrix.\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.apply_transform(item, transform)

def bind_colormap(item, source):
    if False:
        while True:
            i = 10
    '\t Sets the color map for widgets that accept it.\n\n\tArgs:\n\t\titem (Union[int, str]): item that the color map will be applied to\n\t\tsource (Union[int, str]): The colormap tag. This should come from a colormap that was added to a colormap registry.  Built in color maps are accessible through their corresponding constants mvPlotColormap_Twilight, mvPlotColormap_***\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_colormap(item, source)

def bind_font(font):
    if False:
        i = 10
        return i + 15
    '\t Binds a global font.\n\n\tArgs:\n\t\tfont (Union[int, str]): \n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.bind_font(font)

def bind_item_font(item, font):
    if False:
        while True:
            i = 10
    "\t Sets an item's font.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tfont (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.bind_item_font(item, font)

def bind_item_handler_registry(item, handler_registry):
    if False:
        return 10
    '\t Binds an item handler registry to an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\thandler_registry (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_item_handler_registry(item, handler_registry)

def bind_item_theme(item, theme):
    if False:
        return 10
    '\t Binds a theme to an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\ttheme (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_item_theme(item, theme)

def bind_theme(theme):
    if False:
        for i in range(10):
            print('nop')
    '\t Binds a global theme.\n\n\tArgs:\n\t\ttheme (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.bind_theme(theme)

def capture_next_item(callback, **kwargs):
    if False:
        print('Hello World!')
    '\t Captures the next item.\n\n\tArgs:\n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.capture_next_item(callback, **kwargs)

def clear_selected_links(node_editor):
    if False:
        i = 10
        return i + 15
    "\t Clears a node editor's selected links.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.clear_selected_links(node_editor)

def clear_selected_nodes(node_editor):
    if False:
        for i in range(10):
            print('nop')
    "\t Clears a node editor's selected nodes.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.clear_selected_nodes(node_editor)

def create_context():
    if False:
        i = 10
        return i + 15
    '\t Creates the Dear PyGui context.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.create_context()

def create_fps_matrix(eye, pitch, yaw):
    if False:
        while True:
            i = 10
    "\t New in 1.1. Create a 'first person shooter' matrix.\n\n\tArgs:\n\t\teye (Union[List[float], Tuple[float, ...]]): eye position\n\t\tpitch (float): pitch (in radians)\n\t\tyaw (float): yaw (in radians)\n\tReturns:\n\t\tAny\n\t"
    return internal_dpg.create_fps_matrix(eye, pitch, yaw)

def create_lookat_matrix(eye, target, up):
    if False:
        print('Hello World!')
    "\t New in 1.1. Creates a 'Look at matrix'.\n\n\tArgs:\n\t\teye (Union[List[float], Tuple[float, ...]]): eye position\n\t\ttarget (Union[List[float], Tuple[float, ...]]): target position\n\t\tup (Union[List[float], Tuple[float, ...]]): up vector\n\tReturns:\n\t\tAny\n\t"
    return internal_dpg.create_lookat_matrix(eye, target, up)

def create_orthographic_matrix(left, right, bottom, top, zNear, zFar):
    if False:
        while True:
            i = 10
    '\t New in 1.1. Creates an orthographic matrix.\n\n\tArgs:\n\t\tleft (float): left plane\n\t\tright (float): right plane\n\t\tbottom (float): bottom plane\n\t\ttop (float): top plane\n\t\tzNear (float): Near clipping plane.\n\t\tzFar (float): Far clipping plane.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_orthographic_matrix(left, right, bottom, top, zNear, zFar)

def create_perspective_matrix(fov, aspect, zNear, zFar):
    if False:
        print('Hello World!')
    '\t New in 1.1. Creates a perspective matrix.\n\n\tArgs:\n\t\tfov (float): Field of view (in radians)\n\t\taspect (float): Aspect ratio (width/height)\n\t\tzNear (float): Near clipping plane.\n\t\tzFar (float): Far clipping plane.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_perspective_matrix(fov, aspect, zNear, zFar)

def create_rotation_matrix(angle, axis):
    if False:
        for i in range(10):
            print('nop')
    '\t New in 1.1. Applies a transformation matrix to a layer.\n\n\tArgs:\n\t\tangle (float): angle to rotate\n\t\taxis (Union[List[float], Tuple[float, ...]]): axis to rotate around\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_rotation_matrix(angle, axis)

def create_scale_matrix(scales):
    if False:
        return 10
    '\t New in 1.1. Applies a transformation matrix to a layer.\n\n\tArgs:\n\t\tscales (Union[List[float], Tuple[float, ...]]): scale values per axis\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_scale_matrix(scales)

def create_translation_matrix(translation):
    if False:
        for i in range(10):
            print('nop')
    '\t New in 1.1. Creates a translation matrix.\n\n\tArgs:\n\t\ttranslation (Union[List[float], Tuple[float, ...]]): translation vector\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.create_translation_matrix(translation)

def create_viewport(**kwargs):
    if False:
        while True:
            i = 10
    "\t Creates a viewport. Viewports are required.\n\n\tArgs:\n\t\ttitle (str, optional): Sets the title of the viewport.\n\t\tsmall_icon (str, optional): Sets the small icon that is found in the viewport's decorator bar. Must be ***.ico on windows and either ***.ico or ***.png on mac.\n\t\tlarge_icon (str, optional): Sets the large icon that is found in the task bar while the app is running. Must be ***.ico on windows and either ***.ico or ***.png on mac.\n\t\twidth (int, optional): Sets the width of the drawable space on the viewport. Does not inclue the border.\n\t\theight (int, optional): Sets the height of the drawable space on the viewport. Does not inclue the border or decorator bar.\n\t\tx_pos (int, optional): Sets x position the viewport will be drawn in screen coordinates.\n\t\ty_pos (int, optional): Sets y position the viewport will be drawn in screen coordinates.\n\t\tmin_width (int, optional): Applies a minimuim limit to the width of the viewport.\n\t\tmax_width (int, optional): Applies a maximum limit to the width of the viewport.\n\t\tmin_height (int, optional): Applies a minimuim limit to the height of the viewport.\n\t\tmax_height (int, optional): Applies a maximum limit to the height of the viewport.\n\t\tresizable (bool, optional): Enables and Disables user ability to resize the viewport.\n\t\tvsync (bool, optional): Enables and Disables the renderloop vsync limit. vsync frame value is set by refresh rate of display.\n\t\talways_on_top (bool, optional): Forces the viewport to always be drawn ontop of all other viewports.\n\t\tdecorated (bool, optional): Enabled and disabled the decorator bar at the top of the viewport.\n\t\tclear_color (Union[List[float], Tuple[float, ...]], optional): Sets the color of the back of the viewport.\n\t\tdisable_close (bool, optional): Disables the viewport close button. can be used with set_exit_callback\n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.create_viewport(**kwargs)

def delete_item(item, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\t Deletes an item..\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tchildren_only (bool, optional): \n\t\tslot (int, optional): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.delete_item(item, **kwargs)

def destroy_context():
    if False:
        print('Hello World!')
    '\t Destroys the Dear PyGui context.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.destroy_context()

def does_alias_exist(alias):
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if an alias exist.\n\n\tArgs:\n\t\talias (str): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.does_alias_exist(alias)

def does_item_exist(item):
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if an item exist..\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.does_item_exist(item)

def draw_arrow(p1, p2, **kwargs):
    if False:
        return 10
    "\t Adds an arrow.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): Arrow tip.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): Arrow tail.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsize (int, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_arrow(p1, p2, **kwargs)

def draw_bezier_cubic(p1, p2, p3, p4, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a cubic bezier curve.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): First point in curve.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): Second point in curve.\n\t\tp3 (Union[List[float], Tuple[float, ...]]): Third point in curve.\n\t\tp4 (Union[List[float], Tuple[float, ...]]): Fourth point in curve.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate bezier curve.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_bezier_cubic(p1, p2, p3, p4, **kwargs)

def draw_bezier_quadratic(p1, p2, p3, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a quadratic bezier curve.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): First point in curve.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): Second point in curve.\n\t\tp3 (Union[List[float], Tuple[float, ...]]): Third point in curve.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate bezier curve.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_bezier_quadratic(p1, p2, p3, **kwargs)

def draw_circle(center, radius, **kwargs):
    if False:
        return 10
    "\t Adds a circle\n\n\tArgs:\n\t\tcenter (Union[List[float], Tuple[float, ...]]): \n\t\tradius (float): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate circle.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_circle(center, radius, **kwargs)

def draw_ellipse(pmin, pmax, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds an ellipse.\n\n\tArgs:\n\t\tpmin (Union[List[float], Tuple[float, ...]]): Min point of bounding rectangle.\n\t\tpmax (Union[List[float], Tuple[float, ...]]): Max point of bounding rectangle.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tsegments (int, optional): Number of segments to approximate bezier curve.\n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_ellipse(pmin, pmax, **kwargs)

def draw_image(texture_tag, pmin, pmax, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds an image (for a drawing).\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): \n\t\tpmin (Union[List[float], Tuple[float, ...]]): Point of to start drawing texture.\n\t\tpmax (Union[List[float], Tuple[float, ...]]): Point to complete drawing texture.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tuv_min (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv_max (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_image(texture_tag, pmin, pmax, **kwargs)

def draw_image_quad(texture_tag, p1, p2, p3, p4, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\t Adds an image (for a drawing).\n\n\tArgs:\n\t\ttexture_tag (Union[int, str]): \n\t\tp1 (Union[List[float], Tuple[float, ...]]): \n\t\tp2 (Union[List[float], Tuple[float, ...]]): \n\t\tp3 (Union[List[float], Tuple[float, ...]]): \n\t\tp4 (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tuv1 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv2 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv3 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tuv4 (Union[List[float], Tuple[float, ...]], optional): Normalized coordinates on texture that will be drawn.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_image_quad(texture_tag, p1, p2, p3, p4, **kwargs)

def draw_line(p1, p2, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a line.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): Start of line.\n\t\tp2 (Union[List[float], Tuple[float, ...]]): End of line.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_line(p1, p2, **kwargs)

def draw_polygon(points, **kwargs):
    if False:
        while True:
            i = 10
    "\t Adds a polygon.\n\n\tArgs:\n\t\tpoints (List[List[float]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_polygon(points, **kwargs)

def draw_polyline(points, **kwargs):
    if False:
        return 10
    "\t Adds a polyline.\n\n\tArgs:\n\t\tpoints (List[List[float]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tclosed (bool, optional): Will close the polyline by returning to the first point.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_polyline(points, **kwargs)

def draw_quad(p1, p2, p3, p4, **kwargs):
    if False:
        i = 10
        return i + 15
    "\t Adds a quad.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): \n\t\tp2 (Union[List[float], Tuple[float, ...]]): \n\t\tp3 (Union[List[float], Tuple[float, ...]]): \n\t\tp4 (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_quad(p1, p2, p3, p4, **kwargs)

def draw_rectangle(pmin, pmax, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds a rectangle.\n\n\tArgs:\n\t\tpmin (Union[List[float], Tuple[float, ...]]): Min point of bounding rectangle.\n\t\tpmax (Union[List[float], Tuple[float, ...]]): Max point of bounding rectangle.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tcolor_upper_left (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tcolor_upper_right (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tcolor_bottom_right (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tcolor_bottom_left (Union[List[int], Tuple[int, ...]], optional): 'multicolor' must be set to 'True'\n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tmulticolor (bool, optional): \n\t\trounding (float, optional): Number of pixels of the radius that will round the corners of the rectangle. Note: doesn't work with multicolor\n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_rectangle(pmin, pmax, **kwargs)

def draw_text(pos, text, **kwargs):
    if False:
        print('Hello World!')
    "\t Adds text (drawlist).\n\n\tArgs:\n\t\tpos (Union[List[float], Tuple[float, ...]]): Top left point of bounding text rectangle.\n\t\ttext (str): Text to draw.\n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tsize (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_text(pos, text, **kwargs)

def draw_triangle(p1, p2, p3, **kwargs):
    if False:
        return 10
    "\t Adds a triangle.\n\n\tArgs:\n\t\tp1 (Union[List[float], Tuple[float, ...]]): \n\t\tp2 (Union[List[float], Tuple[float, ...]]): \n\t\tp3 (Union[List[float], Tuple[float, ...]]): \n\t\tlabel (str, optional): Overrides 'name' as label.\n\t\tuser_data (Any, optional): User data for callbacks\n\t\tuse_internal_label (bool, optional): Use generated internal label instead of user specified (appends ### uuid).\n\t\ttag (Union[int, str], optional): Unique id used to programmatically refer to the item.If label is unused this will be the label.\n\t\tparent (Union[int, str], optional): Parent to add this item to. (runtime adding)\n\t\tbefore (Union[int, str], optional): This item will be displayed before the specified item in the parent.\n\t\tshow (bool, optional): Attempt to render widget.\n\t\tcolor (Union[List[int], Tuple[int, ...]], optional): \n\t\tfill (Union[List[int], Tuple[int, ...]], optional): \n\t\tthickness (float, optional): \n\t\tid (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tUnion[int, str]\n\t"
    return internal_dpg.draw_triangle(p1, p2, p3, **kwargs)

def empty_container_stack():
    if False:
        print('Hello World!')
    '\t Emptyes the container stack.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.empty_container_stack()

def fit_axis_data(axis):
    if False:
        while True:
            i = 10
    '\t Sets the axis boundaries max/min in the data series currently on the plot.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.fit_axis_data(axis)

def focus_item(item):
    if False:
        return 10
    '\t Focuses an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.focus_item(item)

def generate_uuid():
    if False:
        print('Hello World!')
    '\t Generate a new UUID.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.generate_uuid()

def get_active_window():
    if False:
        print('Hello World!')
    '\t Returns the active window.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.get_active_window()

def get_alias_id(alias):
    if False:
        for i in range(10):
            print('nop')
    '\t Returns the ID associated with an alias.\n\n\tArgs:\n\t\talias (str): \n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.get_alias_id(alias)

def get_aliases():
    if False:
        for i in range(10):
            print('nop')
    '\t Returns all aliases.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[str], Tuple[str, ...]]\n\t'
    return internal_dpg.get_aliases()

def get_all_items():
    if False:
        for i in range(10):
            print('nop')
    '\t Returns all items.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_all_items()

def get_app_configuration():
    if False:
        for i in range(10):
            print('nop')
    '\t Returns app configuration.\n\n\tArgs:\n\tReturns:\n\t\tdict\n\t'
    return internal_dpg.get_app_configuration()

def get_axis_limits(axis):
    if False:
        i = 10
        return i + 15
    '\t Get the specified axis limits.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tUnion[List[float], Tuple[float, ...]]\n\t'
    return internal_dpg.get_axis_limits(axis)

def get_callback_queue():
    if False:
        for i in range(10):
            print('nop')
    '\t New in 1.2. Returns and clears callback queue.\n\n\tArgs:\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.get_callback_queue()

def get_clipboard_text():
    if False:
        for i in range(10):
            print('nop')
    '\t New in 1.3. Gets the clipboard text.\n\n\tArgs:\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.get_clipboard_text()

def get_colormap_color(colormap, index):
    if False:
        while True:
            i = 10
    '\t Returns a color from a colormap given an index >= 0. (ex. 0 will be the first color in the color list of the color map) Modulo will be performed against the number of items in the color list.\n\n\tArgs:\n\t\tcolormap (Union[int, str]): The colormap tag. This should come from a colormap that was added to a colormap registry. Built in color maps are accessible through their corresponding constants mvPlotColormap_Twilight, mvPlotColormap_***\n\t\tindex (int): Desired position of the color in the colors list value of the colormap being quiered \n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_colormap_color(colormap, index)

def get_delta_time():
    if False:
        i = 10
        return i + 15
    '\t Returns time since last frame.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_delta_time()

def get_drawing_mouse_pos():
    if False:
        while True:
            i = 10
    '\t Returns mouse position in drawing.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_drawing_mouse_pos()

def get_file_dialog_info(file_dialog):
    if False:
        while True:
            i = 10
    '\t Returns information related to the file dialog. Typically used while the file dialog is in use to query data about the state or info related to the file dialog.\n\n\tArgs:\n\t\tfile_dialog (Union[int, str]): \n\tReturns:\n\t\tdict\n\t'
    return internal_dpg.get_file_dialog_info(file_dialog)

def get_frame_count():
    if False:
        for i in range(10):
            print('nop')
    '\t Returns frame count.\n\n\tArgs:\n\tReturns:\n\t\tint\n\t'
    return internal_dpg.get_frame_count()

def get_frame_rate():
    if False:
        while True:
            i = 10
    '\t Returns the average frame rate across 120 frames.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_frame_rate()

def get_global_font_scale():
    if False:
        print('Hello World!')
    '\t Returns global font scale.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_global_font_scale()

def get_item_alias(item):
    if False:
        for i in range(10):
            print('nop')
    "\t Returns an item's alias.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tstr\n\t"
    return internal_dpg.get_item_alias(item)

def get_item_configuration(item):
    if False:
        for i in range(10):
            print('nop')
    "\t Returns an item's configuration.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_item_configuration(item)

def get_item_info(item):
    if False:
        print('Hello World!')
    "\t Returns an item's information.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_item_info(item)

def get_item_state(item):
    if False:
        i = 10
        return i + 15
    "\t Returns an item's state.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_item_state(item)

def get_item_types():
    if False:
        print('Hello World!')
    '\t Returns an item types.\n\n\tArgs:\n\tReturns:\n\t\tdict\n\t'
    return internal_dpg.get_item_types()

def get_mouse_drag_delta():
    if False:
        for i in range(10):
            print('nop')
    '\t Returns mouse drag delta.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_mouse_drag_delta()

def get_mouse_pos(**kwargs):
    if False:
        while True:
            i = 10
    '\t Returns mouse position.\n\n\tArgs:\n\t\tlocal (bool, optional): \n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_mouse_pos(**kwargs)

def get_platform():
    if False:
        return 10
    '\t New in 1.6. Returns platform constant.\n\n\tArgs:\n\tReturns:\n\t\tint\n\t'
    return internal_dpg.get_platform()

def get_plot_mouse_pos():
    if False:
        while True:
            i = 10
    '\t Returns mouse position in plot.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_plot_mouse_pos()

def get_plot_query_area(plot):
    if False:
        i = 10
        return i + 15
    "\t Returns the last/current query area of the plot. (Requires plot 'query' kwarg to be enabled)\n\n\tArgs:\n\t\tplot (Union[int, str]): \n\tReturns:\n\t\tUnion[List[float], Tuple[float, ...]]\n\t"
    return internal_dpg.get_plot_query_area(plot)

def get_selected_links(node_editor):
    if False:
        i = 10
        return i + 15
    "\t Returns a node editor's selected links.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tList[List[str]]\n\t"
    return internal_dpg.get_selected_links(node_editor)

def get_selected_nodes(node_editor):
    if False:
        while True:
            i = 10
    "\t Returns a node editor's selected nodes.\n\n\tArgs:\n\t\tnode_editor (Union[int, str]): \n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t"
    return internal_dpg.get_selected_nodes(node_editor)

def get_text_size(text, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\t Returns width/height of text with specified font (must occur after 1st frame).\n\n\tArgs:\n\t\ttext (str): \n\t\twrap_width (float, optional): Wrap width to use (-1.0 turns wrap off).\n\t\tfont (Union[int, str], optional): Font to use.\n\tReturns:\n\t\tUnion[List[float], Tuple[float, ...]]\n\t'
    return internal_dpg.get_text_size(text, **kwargs)

def get_total_time():
    if False:
        for i in range(10):
            print('nop')
    '\t Returns total time since Dear PyGui has started.\n\n\tArgs:\n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_total_time()

def get_value(item):
    if False:
        while True:
            i = 10
    "\t Returns an item's value.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tAny\n\t"
    return internal_dpg.get_value(item)

def get_values(items):
    if False:
        for i in range(10):
            print('nop')
    '\t Returns values of a list of items.\n\n\tArgs:\n\t\titems (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.get_values(items)

def get_viewport_configuration(item):
    if False:
        while True:
            i = 10
    "\t Returns a viewport's configuration.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tdict\n\t"
    return internal_dpg.get_viewport_configuration(item)

def get_windows():
    if False:
        for i in range(10):
            print('nop')
    '\t Returns all windows.\n\n\tArgs:\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.get_windows()

def get_x_scroll(item):
    if False:
        for i in range(10):
            print('nop')
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_x_scroll(item)

def get_x_scroll_max(item):
    if False:
        for i in range(10):
            print('nop')
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_x_scroll_max(item)

def get_y_scroll(item):
    if False:
        print('Hello World!')
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_y_scroll(item)

def get_y_scroll_max(item):
    if False:
        return 10
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tfloat\n\t'
    return internal_dpg.get_y_scroll_max(item)

def highlight_table_cell(table, row, column, color):
    if False:
        i = 10
        return i + 15
    '\t Highlight specified table cell.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolumn (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.highlight_table_cell(table, row, column, color)

def highlight_table_column(table, column, color):
    if False:
        return 10
    '\t Highlight specified table column.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\tcolumn (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.highlight_table_column(table, column, color)

def highlight_table_row(table, row, color):
    if False:
        print('Hello World!')
    '\t Highlight specified table row.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.highlight_table_row(table, row, color)

def is_dearpygui_running():
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if Dear PyGui is running\n\n\tArgs:\n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_dearpygui_running()

def is_key_down(key):
    if False:
        i = 10
        return i + 15
    '\t Checks if key is down.\n\n\tArgs:\n\t\tkey (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_key_down(key)

def is_key_pressed(key):
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if key is pressed.\n\n\tArgs:\n\t\tkey (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_key_pressed(key)

def is_key_released(key):
    if False:
        return 10
    '\t Checks if key is released.\n\n\tArgs:\n\t\tkey (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_key_released(key)

def is_mouse_button_clicked(button):
    if False:
        return 10
    '\t Checks if mouse button is clicked.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_clicked(button)

def is_mouse_button_double_clicked(button):
    if False:
        return 10
    '\t Checks if mouse button is double clicked.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_double_clicked(button)

def is_mouse_button_down(button):
    if False:
        while True:
            i = 10
    '\t Checks if mouse button is down.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_down(button)

def is_mouse_button_dragging(button, threshold):
    if False:
        i = 10
        return i + 15
    '\t Checks if mouse button is down and dragging.\n\n\tArgs:\n\t\tbutton (int): \n\t\tthreshold (float): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_dragging(button, threshold)

def is_mouse_button_released(button):
    if False:
        for i in range(10):
            print('nop')
    '\t Checks if mouse button is released.\n\n\tArgs:\n\t\tbutton (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_mouse_button_released(button)

def is_plot_queried(plot):
    if False:
        print('Hello World!')
    "\t Returns true if the plot is currently being queried. (Requires plot 'query' kwarg to be enabled)\n\n\tArgs:\n\t\tplot (Union[int, str]): \n\tReturns:\n\t\tbool\n\t"
    return internal_dpg.is_plot_queried(plot)

def is_table_cell_highlighted(table, row, column):
    if False:
        return 10
    '\t Checks if a table cell is highlighted.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolumn (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_table_cell_highlighted(table, row, column)

def is_table_column_highlighted(table, column):
    if False:
        while True:
            i = 10
    '\t Checks if a table column is highlighted.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\tcolumn (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_table_column_highlighted(table, column)

def is_table_row_highlighted(table, row):
    if False:
        return 10
    '\t Checks if a table row is highlighted.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_table_row_highlighted(table, row)

def is_viewport_ok():
    if False:
        while True:
            i = 10
    '\t Checks if a viewport has been created and shown.\n\n\tArgs:\n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.is_viewport_ok()

def last_container():
    if False:
        while True:
            i = 10
    '\t Returns the last container item added.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.last_container()

def last_item():
    if False:
        print('Hello World!')
    '\t Returns the last item added.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.last_item()

def last_root():
    if False:
        while True:
            i = 10
    '\t Returns the last root added (registry or window).\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.last_root()

def load_image(file, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\t Loads an image. Returns width, height, channels, mvBuffer\n\n\tArgs:\n\t\tfile (str): \n\t\tgamma (float, optional): Gamma correction factor. (default is 1.0 to avoid automatic gamma correction on loading.\n\t\tgamma_scale_factor (float, optional): Gamma scale factor.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.load_image(file, **kwargs)

def lock_mutex():
    if False:
        return 10
    '\t Locks render thread mutex.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.lock_mutex()

def maximize_viewport():
    if False:
        print('Hello World!')
    '\t Maximizes the viewport.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.maximize_viewport()

def minimize_viewport():
    if False:
        print('Hello World!')
    '\t Minimizes a viewport.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.minimize_viewport()

def move_item(item, **kwargs):
    if False:
        i = 10
        return i + 15
    '\t Moves an item to a new location.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tparent (Union[int, str], optional): \n\t\tbefore (Union[int, str], optional): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.move_item(item, **kwargs)

def move_item_down(item):
    if False:
        return 10
    '\t Moves an item down.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.move_item_down(item)

def move_item_up(item):
    if False:
        while True:
            i = 10
    '\t Moves an item up.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.move_item_up(item)

def output_frame_buffer(file='', **kwargs):
    if False:
        return 10
    '\t Outputs frame buffer as a png if file is specified or through the second argument of a callback if specified. Render loop must have been started.\n\n\tArgs:\n\t\tfile (str, optional): \n\t\tcallback (Callable, optional): Callback will return framebuffer as an array through the second arg.\n\tReturns:\n\t\tAny\n\t'
    return internal_dpg.output_frame_buffer(file, **kwargs)

def pop_container_stack():
    if False:
        for i in range(10):
            print('nop')
    '\t Pops the top item off the parent stack and return its ID.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.pop_container_stack()

def push_container_stack(item):
    if False:
        for i in range(10):
            print('nop')
    '\t Pushes an item onto the container stack.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tbool\n\t'
    return internal_dpg.push_container_stack(item)

def remove_alias(alias):
    if False:
        i = 10
        return i + 15
    '\t Removes an alias.\n\n\tArgs:\n\t\talias (str): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.remove_alias(alias)

def render_dearpygui_frame():
    if False:
        while True:
            i = 10
    '\t Render a single Dear PyGui frame.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.render_dearpygui_frame()

def reorder_items(container, slot, new_order):
    if False:
        i = 10
        return i + 15
    "\t Reorders an item's children.\n\n\tArgs:\n\t\tcontainer (Union[int, str]): \n\t\tslot (int): \n\t\tnew_order (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.reorder_items(container, slot, new_order)

def reset_axis_ticks(axis):
    if False:
        return 10
    '\t Removes the manually set axis ticks and applies the default axis ticks\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.reset_axis_ticks(axis)

def reset_pos(item):
    if False:
        while True:
            i = 10
    "\t Resets an item's position after using 'set_item_pos'.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.reset_pos(item)

def sample_colormap(colormap, t):
    if False:
        for i in range(10):
            print('nop')
    '\t Returns a color from a colormap given t between 0.0-1.0.\n\n\tArgs:\n\t\tcolormap (Union[int, str]): The colormap tag. This should come from a colormap that was added to a colormap registry. Built in color maps are accessible through their corresponding constants mvPlotColormap_Twilight, mvPlotColormap_***\n\t\tt (float): Value of the colormap to sample between 0.0-1.0\n\tReturns:\n\t\tUnion[List[int], Tuple[int, ...]]\n\t'
    return internal_dpg.sample_colormap(colormap, t)

def save_image(file, width, height, data, **kwargs):
    if False:
        print('Hello World!')
    '\t Saves an image. Possible formats: png, bmp, tga, hdr, jpg.\n\n\tArgs:\n\t\tfile (str): \n\t\twidth (int): \n\t\theight (int): \n\t\tdata (Any): \n\t\tcomponents (int, optional): Number of components (1-4). Default of 4.\n\t\tquality (int, optional): Stride in bytes (only used for jpg).\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.save_image(file, width, height, data, **kwargs)

def save_init_file(file):
    if False:
        for i in range(10):
            print('nop')
    '\t Save dpg.ini file.\n\n\tArgs:\n\t\tfile (str): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.save_init_file(file)

def set_axis_limits(axis, ymin, ymax):
    if False:
        return 10
    '\t Sets limits on the axis for pan and zoom.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\t\tymin (float): \n\t\tymax (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_axis_limits(axis, ymin, ymax)

def set_axis_limits_auto(axis):
    if False:
        return 10
    '\t Removes all limits on specified axis.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_axis_limits_auto(axis)

def set_axis_ticks(axis, label_pairs):
    if False:
        return 10
    "\t Replaces axis ticks with 'label_pairs' argument.\n\n\tArgs:\n\t\taxis (Union[int, str]): \n\t\tlabel_pairs (Any): Tuples of label and value in the form '((label, axis_value), (label, axis_value), ...)'\n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_axis_ticks(axis, label_pairs)

def set_clip_space(item, top_left_x, top_left_y, width, height, min_depth, max_depth):
    if False:
        i = 10
        return i + 15
    "\t New in 1.1. Set the clip space for depth clipping and 'viewport' transformation.\n\n\tArgs:\n\t\titem (Union[int, str]): draw layer to set clip space\n\t\ttop_left_x (float): angle to rotate\n\t\ttop_left_y (float): angle to rotate\n\t\twidth (float): angle to rotate\n\t\theight (float): angle to rotate\n\t\tmin_depth (float): angle to rotate\n\t\tmax_depth (float): angle to rotate\n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_clip_space(item, top_left_x, top_left_y, width, height, min_depth, max_depth)

def set_clipboard_text(text):
    if False:
        return 10
    '\t New in 1.3. Sets the clipboard text.\n\n\tArgs:\n\t\ttext (str): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_clipboard_text(text)

def set_exit_callback(callback, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\t Sets a callback to run on last frame.\n\n\tArgs:\n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.set_exit_callback(callback, **kwargs)

def set_frame_callback(frame, callback, **kwargs):
    if False:
        while True:
            i = 10
    '\t Sets a callback to run on first frame.\n\n\tArgs:\n\t\tframe (int): \n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.set_frame_callback(frame, callback, **kwargs)

def set_global_font_scale(scale):
    if False:
        while True:
            i = 10
    '\t Sets global font scale.\n\n\tArgs:\n\t\tscale (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_global_font_scale(scale)

def set_item_alias(item, alias):
    if False:
        i = 10
        return i + 15
    "\t Sets an item's alias.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\talias (str): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_item_alias(item, alias)

def set_item_children(item, source, slot):
    if False:
        i = 10
        return i + 15
    "\t Sets an item's children.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tsource (Union[int, str]): \n\t\tslot (int): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_item_children(item, source, slot)

def set_primary_window(window, value):
    if False:
        print('Hello World!')
    '\t Sets the primary window.\n\n\tArgs:\n\t\twindow (Union[int, str]): \n\t\tvalue (bool): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_primary_window(window, value)

def set_table_row_color(table, row, color):
    if False:
        while True:
            i = 10
    '\t Set table row color.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolor (Union[List[int], Tuple[int, ...]]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_table_row_color(table, row, color)

def set_value(item, value):
    if False:
        for i in range(10):
            print('nop')
    "\t Set's an item's value.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tvalue (Any): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.set_value(item, value)

def set_viewport_resize_callback(callback, **kwargs):
    if False:
        while True:
            i = 10
    '\t Sets a callback to run on viewport resize.\n\n\tArgs:\n\t\tcallback (Callable): \n\t\tuser_data (Any, optional): New in 1.3. Optional user data to send to the callback\n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.set_viewport_resize_callback(callback, **kwargs)

def set_x_scroll(item, value):
    if False:
        i = 10
        return i + 15
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tvalue (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_x_scroll(item, value)

def set_y_scroll(item, value):
    if False:
        return 10
    '\t Undocumented\n\n\tArgs:\n\t\titem (Union[int, str]): \n\t\tvalue (float): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.set_y_scroll(item, value)

def setup_dearpygui():
    if False:
        i = 10
        return i + 15
    '\t Sets up Dear PyGui\n\n\tArgs:\n\t\tviewport (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.setup_dearpygui()

def show_imgui_demo():
    if False:
        for i in range(10):
            print('nop')
    '\t Shows the imgui demo.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.show_imgui_demo()

def show_implot_demo():
    if False:
        i = 10
        return i + 15
    '\t Shows the implot demo.\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.show_implot_demo()

def show_item_debug(item):
    if False:
        while True:
            i = 10
    "\t Shows an item's debug window\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t"
    return internal_dpg.show_item_debug(item)

def show_tool(tool):
    if False:
        i = 10
        return i + 15
    '\t Shows a built in tool.\n\n\tArgs:\n\t\ttool (Union[int, str]): \n\tReturns:\n\t\tstr\n\t'
    return internal_dpg.show_tool(tool)

def show_viewport(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\t Shows the main viewport.\n\n\tArgs:\n\t\tminimized (bool, optional): Sets the state of the viewport to minimized\n\t\tmaximized (bool, optional): Sets the state of the viewport to maximized\n\t\tviewport (Union[int, str], optional): (deprecated)\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.show_viewport(**kwargs)

def split_frame(**kwargs):
    if False:
        while True:
            i = 10
    '\t Waits one frame.\n\n\tArgs:\n\t\tdelay (int, optional): Minimal delay in in milliseconds\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.split_frame(**kwargs)

def stop_dearpygui():
    if False:
        for i in range(10):
            print('nop')
    '\t Stops Dear PyGui\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.stop_dearpygui()

def toggle_viewport_fullscreen():
    if False:
        while True:
            i = 10
    '\t Toggle viewport fullscreen mode..\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.toggle_viewport_fullscreen()

def top_container_stack():
    if False:
        print('Hello World!')
    '\t Returns the item on the top of the container stack.\n\n\tArgs:\n\tReturns:\n\t\tUnion[int, str]\n\t'
    return internal_dpg.top_container_stack()

def unhighlight_table_cell(table, row, column):
    if False:
        i = 10
        return i + 15
    '\t Unhighlight specified table cell.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\t\tcolumn (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unhighlight_table_cell(table, row, column)

def unhighlight_table_column(table, column):
    if False:
        i = 10
        return i + 15
    '\t Unhighlight specified table column.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\tcolumn (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unhighlight_table_column(table, column)

def unhighlight_table_row(table, row):
    if False:
        for i in range(10):
            print('nop')
    '\t Unhighlight specified table row.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unhighlight_table_row(table, row)

def unlock_mutex():
    if False:
        print('Hello World!')
    '\t Unlocks render thread mutex\n\n\tArgs:\n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unlock_mutex()

def unset_table_row_color(table, row):
    if False:
        for i in range(10):
            print('nop')
    '\t Remove user set table row color.\n\n\tArgs:\n\t\ttable (Union[int, str]): \n\t\trow (int): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unset_table_row_color(table, row)

def unstage(item):
    if False:
        i = 10
        return i + 15
    '\t Unstages an item.\n\n\tArgs:\n\t\titem (Union[int, str]): \n\tReturns:\n\t\tNone\n\t'
    return internal_dpg.unstage(item)
mvGraphicsBackend_D3D11 = internal_dpg.mvGraphicsBackend_D3D11
mvGraphicsBackend_D3D12 = internal_dpg.mvGraphicsBackend_D3D12
mvGraphicsBackend_VULKAN = internal_dpg.mvGraphicsBackend_VULKAN
mvGraphicsBackend_METAL = internal_dpg.mvGraphicsBackend_METAL
mvGraphicsBackend_OPENGL = internal_dpg.mvGraphicsBackend_OPENGL
mvMouseButton_Left = internal_dpg.mvMouseButton_Left
mvMouseButton_Right = internal_dpg.mvMouseButton_Right
mvMouseButton_Middle = internal_dpg.mvMouseButton_Middle
mvMouseButton_X1 = internal_dpg.mvMouseButton_X1
mvMouseButton_X2 = internal_dpg.mvMouseButton_X2
mvKey_0 = internal_dpg.mvKey_0
mvKey_1 = internal_dpg.mvKey_1
mvKey_2 = internal_dpg.mvKey_2
mvKey_3 = internal_dpg.mvKey_3
mvKey_4 = internal_dpg.mvKey_4
mvKey_5 = internal_dpg.mvKey_5
mvKey_6 = internal_dpg.mvKey_6
mvKey_7 = internal_dpg.mvKey_7
mvKey_8 = internal_dpg.mvKey_8
mvKey_9 = internal_dpg.mvKey_9
mvKey_A = internal_dpg.mvKey_A
mvKey_B = internal_dpg.mvKey_B
mvKey_C = internal_dpg.mvKey_C
mvKey_D = internal_dpg.mvKey_D
mvKey_E = internal_dpg.mvKey_E
mvKey_F = internal_dpg.mvKey_F
mvKey_G = internal_dpg.mvKey_G
mvKey_H = internal_dpg.mvKey_H
mvKey_I = internal_dpg.mvKey_I
mvKey_J = internal_dpg.mvKey_J
mvKey_K = internal_dpg.mvKey_K
mvKey_L = internal_dpg.mvKey_L
mvKey_M = internal_dpg.mvKey_M
mvKey_N = internal_dpg.mvKey_N
mvKey_O = internal_dpg.mvKey_O
mvKey_P = internal_dpg.mvKey_P
mvKey_Q = internal_dpg.mvKey_Q
mvKey_R = internal_dpg.mvKey_R
mvKey_S = internal_dpg.mvKey_S
mvKey_T = internal_dpg.mvKey_T
mvKey_U = internal_dpg.mvKey_U
mvKey_V = internal_dpg.mvKey_V
mvKey_W = internal_dpg.mvKey_W
mvKey_X = internal_dpg.mvKey_X
mvKey_Y = internal_dpg.mvKey_Y
mvKey_Z = internal_dpg.mvKey_Z
mvKey_Back = internal_dpg.mvKey_Back
mvKey_Tab = internal_dpg.mvKey_Tab
mvKey_Clear = internal_dpg.mvKey_Clear
mvKey_Return = internal_dpg.mvKey_Return
mvKey_Shift = internal_dpg.mvKey_Shift
mvKey_Control = internal_dpg.mvKey_Control
mvKey_Alt = internal_dpg.mvKey_Alt
mvKey_Pause = internal_dpg.mvKey_Pause
mvKey_Capital = internal_dpg.mvKey_Capital
mvKey_Escape = internal_dpg.mvKey_Escape
mvKey_Spacebar = internal_dpg.mvKey_Spacebar
mvKey_Prior = internal_dpg.mvKey_Prior
mvKey_Next = internal_dpg.mvKey_Next
mvKey_End = internal_dpg.mvKey_End
mvKey_Home = internal_dpg.mvKey_Home
mvKey_Left = internal_dpg.mvKey_Left
mvKey_Up = internal_dpg.mvKey_Up
mvKey_Right = internal_dpg.mvKey_Right
mvKey_Down = internal_dpg.mvKey_Down
mvKey_Select = internal_dpg.mvKey_Select
mvKey_Print = internal_dpg.mvKey_Print
mvKey_Execute = internal_dpg.mvKey_Execute
mvKey_PrintScreen = internal_dpg.mvKey_PrintScreen
mvKey_Insert = internal_dpg.mvKey_Insert
mvKey_Delete = internal_dpg.mvKey_Delete
mvKey_Help = internal_dpg.mvKey_Help
mvKey_LWin = internal_dpg.mvKey_LWin
mvKey_RWin = internal_dpg.mvKey_RWin
mvKey_Apps = internal_dpg.mvKey_Apps
mvKey_Sleep = internal_dpg.mvKey_Sleep
mvKey_NumPad0 = internal_dpg.mvKey_NumPad0
mvKey_NumPad1 = internal_dpg.mvKey_NumPad1
mvKey_NumPad2 = internal_dpg.mvKey_NumPad2
mvKey_NumPad3 = internal_dpg.mvKey_NumPad3
mvKey_NumPad4 = internal_dpg.mvKey_NumPad4
mvKey_NumPad5 = internal_dpg.mvKey_NumPad5
mvKey_NumPad6 = internal_dpg.mvKey_NumPad6
mvKey_NumPad7 = internal_dpg.mvKey_NumPad7
mvKey_NumPad8 = internal_dpg.mvKey_NumPad8
mvKey_NumPad9 = internal_dpg.mvKey_NumPad9
mvKey_Multiply = internal_dpg.mvKey_Multiply
mvKey_Add = internal_dpg.mvKey_Add
mvKey_Separator = internal_dpg.mvKey_Separator
mvKey_Subtract = internal_dpg.mvKey_Subtract
mvKey_Decimal = internal_dpg.mvKey_Decimal
mvKey_Divide = internal_dpg.mvKey_Divide
mvKey_F1 = internal_dpg.mvKey_F1
mvKey_F2 = internal_dpg.mvKey_F2
mvKey_F3 = internal_dpg.mvKey_F3
mvKey_F4 = internal_dpg.mvKey_F4
mvKey_F5 = internal_dpg.mvKey_F5
mvKey_F6 = internal_dpg.mvKey_F6
mvKey_F7 = internal_dpg.mvKey_F7
mvKey_F8 = internal_dpg.mvKey_F8
mvKey_F9 = internal_dpg.mvKey_F9
mvKey_F10 = internal_dpg.mvKey_F10
mvKey_F11 = internal_dpg.mvKey_F11
mvKey_F12 = internal_dpg.mvKey_F12
mvKey_F13 = internal_dpg.mvKey_F13
mvKey_F14 = internal_dpg.mvKey_F14
mvKey_F15 = internal_dpg.mvKey_F15
mvKey_F16 = internal_dpg.mvKey_F16
mvKey_F17 = internal_dpg.mvKey_F17
mvKey_F18 = internal_dpg.mvKey_F18
mvKey_F19 = internal_dpg.mvKey_F19
mvKey_F20 = internal_dpg.mvKey_F20
mvKey_F21 = internal_dpg.mvKey_F21
mvKey_F22 = internal_dpg.mvKey_F22
mvKey_F23 = internal_dpg.mvKey_F23
mvKey_F24 = internal_dpg.mvKey_F24
mvKey_F25 = internal_dpg.mvKey_F25
mvKey_NumLock = internal_dpg.mvKey_NumLock
mvKey_ScrollLock = internal_dpg.mvKey_ScrollLock
mvKey_LShift = internal_dpg.mvKey_LShift
mvKey_RShift = internal_dpg.mvKey_RShift
mvKey_LControl = internal_dpg.mvKey_LControl
mvKey_RControl = internal_dpg.mvKey_RControl
mvKey_LMenu = internal_dpg.mvKey_LMenu
mvKey_RMenu = internal_dpg.mvKey_RMenu
mvKey_Browser_Back = internal_dpg.mvKey_Browser_Back
mvKey_Browser_Forward = internal_dpg.mvKey_Browser_Forward
mvKey_Browser_Refresh = internal_dpg.mvKey_Browser_Refresh
mvKey_Browser_Stop = internal_dpg.mvKey_Browser_Stop
mvKey_Browser_Search = internal_dpg.mvKey_Browser_Search
mvKey_Browser_Favorites = internal_dpg.mvKey_Browser_Favorites
mvKey_Browser_Home = internal_dpg.mvKey_Browser_Home
mvKey_Volume_Mute = internal_dpg.mvKey_Volume_Mute
mvKey_Volume_Down = internal_dpg.mvKey_Volume_Down
mvKey_Volume_Up = internal_dpg.mvKey_Volume_Up
mvKey_Media_Next_Track = internal_dpg.mvKey_Media_Next_Track
mvKey_Media_Prev_Track = internal_dpg.mvKey_Media_Prev_Track
mvKey_Media_Stop = internal_dpg.mvKey_Media_Stop
mvKey_Media_Play_Pause = internal_dpg.mvKey_Media_Play_Pause
mvKey_Launch_Mail = internal_dpg.mvKey_Launch_Mail
mvKey_Launch_Media_Select = internal_dpg.mvKey_Launch_Media_Select
mvKey_Launch_App1 = internal_dpg.mvKey_Launch_App1
mvKey_Launch_App2 = internal_dpg.mvKey_Launch_App2
mvKey_Colon = internal_dpg.mvKey_Colon
mvKey_Plus = internal_dpg.mvKey_Plus
mvKey_Comma = internal_dpg.mvKey_Comma
mvKey_Minus = internal_dpg.mvKey_Minus
mvKey_Period = internal_dpg.mvKey_Period
mvKey_Slash = internal_dpg.mvKey_Slash
mvKey_Tilde = internal_dpg.mvKey_Tilde
mvKey_Open_Brace = internal_dpg.mvKey_Open_Brace
mvKey_Backslash = internal_dpg.mvKey_Backslash
mvKey_Close_Brace = internal_dpg.mvKey_Close_Brace
mvKey_Quote = internal_dpg.mvKey_Quote
mvAll = internal_dpg.mvAll
mvTool_About = internal_dpg.mvTool_About
mvTool_Debug = internal_dpg.mvTool_Debug
mvTool_Doc = internal_dpg.mvTool_Doc
mvTool_ItemRegistry = internal_dpg.mvTool_ItemRegistry
mvTool_Metrics = internal_dpg.mvTool_Metrics
mvTool_Style = internal_dpg.mvTool_Style
mvTool_Font = internal_dpg.mvTool_Font
mvFontAtlas = internal_dpg.mvFontAtlas
mvAppUUID = internal_dpg.mvAppUUID
mvInvalidUUID = internal_dpg.mvInvalidUUID
mvDir_None = internal_dpg.mvDir_None
mvDir_Left = internal_dpg.mvDir_Left
mvDir_Right = internal_dpg.mvDir_Right
mvDir_Up = internal_dpg.mvDir_Up
mvDir_Down = internal_dpg.mvDir_Down
mvComboHeight_Small = internal_dpg.mvComboHeight_Small
mvComboHeight_Regular = internal_dpg.mvComboHeight_Regular
mvComboHeight_Large = internal_dpg.mvComboHeight_Large
mvComboHeight_Largest = internal_dpg.mvComboHeight_Largest
mvPlatform_Windows = internal_dpg.mvPlatform_Windows
mvPlatform_Apple = internal_dpg.mvPlatform_Apple
mvPlatform_Linux = internal_dpg.mvPlatform_Linux
mvColorEdit_AlphaPreviewNone = internal_dpg.mvColorEdit_AlphaPreviewNone
mvColorEdit_AlphaPreview = internal_dpg.mvColorEdit_AlphaPreview
mvColorEdit_AlphaPreviewHalf = internal_dpg.mvColorEdit_AlphaPreviewHalf
mvColorEdit_uint8 = internal_dpg.mvColorEdit_uint8
mvColorEdit_float = internal_dpg.mvColorEdit_float
mvColorEdit_rgb = internal_dpg.mvColorEdit_rgb
mvColorEdit_hsv = internal_dpg.mvColorEdit_hsv
mvColorEdit_hex = internal_dpg.mvColorEdit_hex
mvColorEdit_input_rgb = internal_dpg.mvColorEdit_input_rgb
mvColorEdit_input_hsv = internal_dpg.mvColorEdit_input_hsv
mvPlotColormap_Default = internal_dpg.mvPlotColormap_Default
mvPlotColormap_Deep = internal_dpg.mvPlotColormap_Deep
mvPlotColormap_Dark = internal_dpg.mvPlotColormap_Dark
mvPlotColormap_Pastel = internal_dpg.mvPlotColormap_Pastel
mvPlotColormap_Paired = internal_dpg.mvPlotColormap_Paired
mvPlotColormap_Viridis = internal_dpg.mvPlotColormap_Viridis
mvPlotColormap_Plasma = internal_dpg.mvPlotColormap_Plasma
mvPlotColormap_Hot = internal_dpg.mvPlotColormap_Hot
mvPlotColormap_Cool = internal_dpg.mvPlotColormap_Cool
mvPlotColormap_Pink = internal_dpg.mvPlotColormap_Pink
mvPlotColormap_Jet = internal_dpg.mvPlotColormap_Jet
mvPlotColormap_Twilight = internal_dpg.mvPlotColormap_Twilight
mvPlotColormap_RdBu = internal_dpg.mvPlotColormap_RdBu
mvPlotColormap_BrBG = internal_dpg.mvPlotColormap_BrBG
mvPlotColormap_PiYG = internal_dpg.mvPlotColormap_PiYG
mvPlotColormap_Spectral = internal_dpg.mvPlotColormap_Spectral
mvPlotColormap_Greys = internal_dpg.mvPlotColormap_Greys
mvColorPicker_bar = internal_dpg.mvColorPicker_bar
mvColorPicker_wheel = internal_dpg.mvColorPicker_wheel
mvTabOrder_Reorderable = internal_dpg.mvTabOrder_Reorderable
mvTabOrder_Fixed = internal_dpg.mvTabOrder_Fixed
mvTabOrder_Leading = internal_dpg.mvTabOrder_Leading
mvTabOrder_Trailing = internal_dpg.mvTabOrder_Trailing
mvTimeUnit_Us = internal_dpg.mvTimeUnit_Us
mvTimeUnit_Ms = internal_dpg.mvTimeUnit_Ms
mvTimeUnit_S = internal_dpg.mvTimeUnit_S
mvTimeUnit_Min = internal_dpg.mvTimeUnit_Min
mvTimeUnit_Hr = internal_dpg.mvTimeUnit_Hr
mvTimeUnit_Day = internal_dpg.mvTimeUnit_Day
mvTimeUnit_Mo = internal_dpg.mvTimeUnit_Mo
mvTimeUnit_Yr = internal_dpg.mvTimeUnit_Yr
mvDatePickerLevel_Day = internal_dpg.mvDatePickerLevel_Day
mvDatePickerLevel_Month = internal_dpg.mvDatePickerLevel_Month
mvDatePickerLevel_Year = internal_dpg.mvDatePickerLevel_Year
mvCullMode_None = internal_dpg.mvCullMode_None
mvCullMode_Back = internal_dpg.mvCullMode_Back
mvCullMode_Front = internal_dpg.mvCullMode_Front
mvFontRangeHint_Default = internal_dpg.mvFontRangeHint_Default
mvFontRangeHint_Japanese = internal_dpg.mvFontRangeHint_Japanese
mvFontRangeHint_Korean = internal_dpg.mvFontRangeHint_Korean
mvFontRangeHint_Chinese_Full = internal_dpg.mvFontRangeHint_Chinese_Full
mvFontRangeHint_Chinese_Simplified_Common = internal_dpg.mvFontRangeHint_Chinese_Simplified_Common
mvFontRangeHint_Cyrillic = internal_dpg.mvFontRangeHint_Cyrillic
mvFontRangeHint_Thai = internal_dpg.mvFontRangeHint_Thai
mvFontRangeHint_Vietnamese = internal_dpg.mvFontRangeHint_Vietnamese
mvNode_PinShape_Circle = internal_dpg.mvNode_PinShape_Circle
mvNode_PinShape_CircleFilled = internal_dpg.mvNode_PinShape_CircleFilled
mvNode_PinShape_Triangle = internal_dpg.mvNode_PinShape_Triangle
mvNode_PinShape_TriangleFilled = internal_dpg.mvNode_PinShape_TriangleFilled
mvNode_PinShape_Quad = internal_dpg.mvNode_PinShape_Quad
mvNode_PinShape_QuadFilled = internal_dpg.mvNode_PinShape_QuadFilled
mvNode_Attr_Input = internal_dpg.mvNode_Attr_Input
mvNode_Attr_Output = internal_dpg.mvNode_Attr_Output
mvNode_Attr_Static = internal_dpg.mvNode_Attr_Static
mvPlotBin_Sqrt = internal_dpg.mvPlotBin_Sqrt
mvPlotBin_Sturges = internal_dpg.mvPlotBin_Sturges
mvPlotBin_Rice = internal_dpg.mvPlotBin_Rice
mvPlotBin_Scott = internal_dpg.mvPlotBin_Scott
mvXAxis = internal_dpg.mvXAxis
mvYAxis = internal_dpg.mvYAxis
mvPlotMarker_None = internal_dpg.mvPlotMarker_None
mvPlotMarker_Circle = internal_dpg.mvPlotMarker_Circle
mvPlotMarker_Square = internal_dpg.mvPlotMarker_Square
mvPlotMarker_Diamond = internal_dpg.mvPlotMarker_Diamond
mvPlotMarker_Up = internal_dpg.mvPlotMarker_Up
mvPlotMarker_Down = internal_dpg.mvPlotMarker_Down
mvPlotMarker_Left = internal_dpg.mvPlotMarker_Left
mvPlotMarker_Right = internal_dpg.mvPlotMarker_Right
mvPlotMarker_Cross = internal_dpg.mvPlotMarker_Cross
mvPlotMarker_Plus = internal_dpg.mvPlotMarker_Plus
mvPlotMarker_Asterisk = internal_dpg.mvPlotMarker_Asterisk
mvPlot_Location_Center = internal_dpg.mvPlot_Location_Center
mvPlot_Location_North = internal_dpg.mvPlot_Location_North
mvPlot_Location_South = internal_dpg.mvPlot_Location_South
mvPlot_Location_West = internal_dpg.mvPlot_Location_West
mvPlot_Location_East = internal_dpg.mvPlot_Location_East
mvPlot_Location_NorthWest = internal_dpg.mvPlot_Location_NorthWest
mvPlot_Location_NorthEast = internal_dpg.mvPlot_Location_NorthEast
mvPlot_Location_SouthWest = internal_dpg.mvPlot_Location_SouthWest
mvPlot_Location_SouthEast = internal_dpg.mvPlot_Location_SouthEast
mvNodeMiniMap_Location_BottomLeft = internal_dpg.mvNodeMiniMap_Location_BottomLeft
mvNodeMiniMap_Location_BottomRight = internal_dpg.mvNodeMiniMap_Location_BottomRight
mvNodeMiniMap_Location_TopLeft = internal_dpg.mvNodeMiniMap_Location_TopLeft
mvNodeMiniMap_Location_TopRight = internal_dpg.mvNodeMiniMap_Location_TopRight
mvTable_SizingFixedFit = internal_dpg.mvTable_SizingFixedFit
mvTable_SizingFixedSame = internal_dpg.mvTable_SizingFixedSame
mvTable_SizingStretchProp = internal_dpg.mvTable_SizingStretchProp
mvTable_SizingStretchSame = internal_dpg.mvTable_SizingStretchSame
mvFormat_Float_rgba = internal_dpg.mvFormat_Float_rgba
mvFormat_Float_rgb = internal_dpg.mvFormat_Float_rgb
mvThemeCat_Core = internal_dpg.mvThemeCat_Core
mvThemeCat_Plots = internal_dpg.mvThemeCat_Plots
mvThemeCat_Nodes = internal_dpg.mvThemeCat_Nodes
mvThemeCol_Text = internal_dpg.mvThemeCol_Text
mvThemeCol_TextDisabled = internal_dpg.mvThemeCol_TextDisabled
mvThemeCol_WindowBg = internal_dpg.mvThemeCol_WindowBg
mvThemeCol_ChildBg = internal_dpg.mvThemeCol_ChildBg
mvThemeCol_Border = internal_dpg.mvThemeCol_Border
mvThemeCol_PopupBg = internal_dpg.mvThemeCol_PopupBg
mvThemeCol_BorderShadow = internal_dpg.mvThemeCol_BorderShadow
mvThemeCol_FrameBg = internal_dpg.mvThemeCol_FrameBg
mvThemeCol_FrameBgHovered = internal_dpg.mvThemeCol_FrameBgHovered
mvThemeCol_FrameBgActive = internal_dpg.mvThemeCol_FrameBgActive
mvThemeCol_TitleBg = internal_dpg.mvThemeCol_TitleBg
mvThemeCol_TitleBgActive = internal_dpg.mvThemeCol_TitleBgActive
mvThemeCol_TitleBgCollapsed = internal_dpg.mvThemeCol_TitleBgCollapsed
mvThemeCol_MenuBarBg = internal_dpg.mvThemeCol_MenuBarBg
mvThemeCol_ScrollbarBg = internal_dpg.mvThemeCol_ScrollbarBg
mvThemeCol_ScrollbarGrab = internal_dpg.mvThemeCol_ScrollbarGrab
mvThemeCol_ScrollbarGrabHovered = internal_dpg.mvThemeCol_ScrollbarGrabHovered
mvThemeCol_ScrollbarGrabActive = internal_dpg.mvThemeCol_ScrollbarGrabActive
mvThemeCol_CheckMark = internal_dpg.mvThemeCol_CheckMark
mvThemeCol_SliderGrab = internal_dpg.mvThemeCol_SliderGrab
mvThemeCol_SliderGrabActive = internal_dpg.mvThemeCol_SliderGrabActive
mvThemeCol_Button = internal_dpg.mvThemeCol_Button
mvThemeCol_ButtonHovered = internal_dpg.mvThemeCol_ButtonHovered
mvThemeCol_ButtonActive = internal_dpg.mvThemeCol_ButtonActive
mvThemeCol_Header = internal_dpg.mvThemeCol_Header
mvThemeCol_HeaderHovered = internal_dpg.mvThemeCol_HeaderHovered
mvThemeCol_HeaderActive = internal_dpg.mvThemeCol_HeaderActive
mvThemeCol_Separator = internal_dpg.mvThemeCol_Separator
mvThemeCol_SeparatorHovered = internal_dpg.mvThemeCol_SeparatorHovered
mvThemeCol_SeparatorActive = internal_dpg.mvThemeCol_SeparatorActive
mvThemeCol_ResizeGrip = internal_dpg.mvThemeCol_ResizeGrip
mvThemeCol_ResizeGripHovered = internal_dpg.mvThemeCol_ResizeGripHovered
mvThemeCol_ResizeGripActive = internal_dpg.mvThemeCol_ResizeGripActive
mvThemeCol_Tab = internal_dpg.mvThemeCol_Tab
mvThemeCol_TabHovered = internal_dpg.mvThemeCol_TabHovered
mvThemeCol_TabActive = internal_dpg.mvThemeCol_TabActive
mvThemeCol_TabUnfocused = internal_dpg.mvThemeCol_TabUnfocused
mvThemeCol_TabUnfocusedActive = internal_dpg.mvThemeCol_TabUnfocusedActive
mvThemeCol_DockingPreview = internal_dpg.mvThemeCol_DockingPreview
mvThemeCol_DockingEmptyBg = internal_dpg.mvThemeCol_DockingEmptyBg
mvThemeCol_PlotLines = internal_dpg.mvThemeCol_PlotLines
mvThemeCol_PlotLinesHovered = internal_dpg.mvThemeCol_PlotLinesHovered
mvThemeCol_PlotHistogram = internal_dpg.mvThemeCol_PlotHistogram
mvThemeCol_PlotHistogramHovered = internal_dpg.mvThemeCol_PlotHistogramHovered
mvThemeCol_TableHeaderBg = internal_dpg.mvThemeCol_TableHeaderBg
mvThemeCol_TableBorderStrong = internal_dpg.mvThemeCol_TableBorderStrong
mvThemeCol_TableBorderLight = internal_dpg.mvThemeCol_TableBorderLight
mvThemeCol_TableRowBg = internal_dpg.mvThemeCol_TableRowBg
mvThemeCol_TableRowBgAlt = internal_dpg.mvThemeCol_TableRowBgAlt
mvThemeCol_TextSelectedBg = internal_dpg.mvThemeCol_TextSelectedBg
mvThemeCol_DragDropTarget = internal_dpg.mvThemeCol_DragDropTarget
mvThemeCol_NavHighlight = internal_dpg.mvThemeCol_NavHighlight
mvThemeCol_NavWindowingHighlight = internal_dpg.mvThemeCol_NavWindowingHighlight
mvThemeCol_NavWindowingDimBg = internal_dpg.mvThemeCol_NavWindowingDimBg
mvThemeCol_ModalWindowDimBg = internal_dpg.mvThemeCol_ModalWindowDimBg
mvPlotCol_Line = internal_dpg.mvPlotCol_Line
mvPlotCol_Fill = internal_dpg.mvPlotCol_Fill
mvPlotCol_MarkerOutline = internal_dpg.mvPlotCol_MarkerOutline
mvPlotCol_MarkerFill = internal_dpg.mvPlotCol_MarkerFill
mvPlotCol_ErrorBar = internal_dpg.mvPlotCol_ErrorBar
mvPlotCol_FrameBg = internal_dpg.mvPlotCol_FrameBg
mvPlotCol_PlotBg = internal_dpg.mvPlotCol_PlotBg
mvPlotCol_PlotBorder = internal_dpg.mvPlotCol_PlotBorder
mvPlotCol_LegendBg = internal_dpg.mvPlotCol_LegendBg
mvPlotCol_LegendBorder = internal_dpg.mvPlotCol_LegendBorder
mvPlotCol_LegendText = internal_dpg.mvPlotCol_LegendText
mvPlotCol_TitleText = internal_dpg.mvPlotCol_TitleText
mvPlotCol_InlayText = internal_dpg.mvPlotCol_InlayText
mvPlotCol_XAxis = internal_dpg.mvPlotCol_XAxis
mvPlotCol_XAxisGrid = internal_dpg.mvPlotCol_XAxisGrid
mvPlotCol_YAxis = internal_dpg.mvPlotCol_YAxis
mvPlotCol_YAxisGrid = internal_dpg.mvPlotCol_YAxisGrid
mvPlotCol_YAxis2 = internal_dpg.mvPlotCol_YAxis2
mvPlotCol_YAxisGrid2 = internal_dpg.mvPlotCol_YAxisGrid2
mvPlotCol_YAxis3 = internal_dpg.mvPlotCol_YAxis3
mvPlotCol_YAxisGrid3 = internal_dpg.mvPlotCol_YAxisGrid3
mvPlotCol_Selection = internal_dpg.mvPlotCol_Selection
mvPlotCol_Query = internal_dpg.mvPlotCol_Query
mvPlotCol_Crosshairs = internal_dpg.mvPlotCol_Crosshairs
mvNodeCol_NodeBackground = internal_dpg.mvNodeCol_NodeBackground
mvNodeCol_NodeBackgroundHovered = internal_dpg.mvNodeCol_NodeBackgroundHovered
mvNodeCol_NodeBackgroundSelected = internal_dpg.mvNodeCol_NodeBackgroundSelected
mvNodeCol_NodeOutline = internal_dpg.mvNodeCol_NodeOutline
mvNodeCol_TitleBar = internal_dpg.mvNodeCol_TitleBar
mvNodeCol_TitleBarHovered = internal_dpg.mvNodeCol_TitleBarHovered
mvNodeCol_TitleBarSelected = internal_dpg.mvNodeCol_TitleBarSelected
mvNodeCol_Link = internal_dpg.mvNodeCol_Link
mvNodeCol_LinkHovered = internal_dpg.mvNodeCol_LinkHovered
mvNodeCol_LinkSelected = internal_dpg.mvNodeCol_LinkSelected
mvNodeCol_Pin = internal_dpg.mvNodeCol_Pin
mvNodeCol_PinHovered = internal_dpg.mvNodeCol_PinHovered
mvNodeCol_BoxSelector = internal_dpg.mvNodeCol_BoxSelector
mvNodeCol_BoxSelectorOutline = internal_dpg.mvNodeCol_BoxSelectorOutline
mvNodeCol_GridBackground = internal_dpg.mvNodeCol_GridBackground
mvNodeCol_GridLine = internal_dpg.mvNodeCol_GridLine
mvNodesCol_GridLinePrimary = internal_dpg.mvNodesCol_GridLinePrimary
mvNodesCol_MiniMapBackground = internal_dpg.mvNodesCol_MiniMapBackground
mvNodesCol_MiniMapBackgroundHovered = internal_dpg.mvNodesCol_MiniMapBackgroundHovered
mvNodesCol_MiniMapOutline = internal_dpg.mvNodesCol_MiniMapOutline
mvNodesCol_MiniMapOutlineHovered = internal_dpg.mvNodesCol_MiniMapOutlineHovered
mvNodesCol_MiniMapNodeBackground = internal_dpg.mvNodesCol_MiniMapNodeBackground
mvNodesCol_MiniMapNodeBackgroundHovered = internal_dpg.mvNodesCol_MiniMapNodeBackgroundHovered
mvNodesCol_MiniMapNodeBackgroundSelected = internal_dpg.mvNodesCol_MiniMapNodeBackgroundSelected
mvNodesCol_MiniMapNodeOutline = internal_dpg.mvNodesCol_MiniMapNodeOutline
mvNodesCol_MiniMapLink = internal_dpg.mvNodesCol_MiniMapLink
mvNodesCol_MiniMapLinkSelected = internal_dpg.mvNodesCol_MiniMapLinkSelected
mvNodesCol_MiniMapCanvas = internal_dpg.mvNodesCol_MiniMapCanvas
mvNodesCol_MiniMapCanvasOutline = internal_dpg.mvNodesCol_MiniMapCanvasOutline
mvStyleVar_Alpha = internal_dpg.mvStyleVar_Alpha
mvStyleVar_WindowPadding = internal_dpg.mvStyleVar_WindowPadding
mvStyleVar_WindowRounding = internal_dpg.mvStyleVar_WindowRounding
mvStyleVar_WindowBorderSize = internal_dpg.mvStyleVar_WindowBorderSize
mvStyleVar_WindowMinSize = internal_dpg.mvStyleVar_WindowMinSize
mvStyleVar_WindowTitleAlign = internal_dpg.mvStyleVar_WindowTitleAlign
mvStyleVar_ChildRounding = internal_dpg.mvStyleVar_ChildRounding
mvStyleVar_ChildBorderSize = internal_dpg.mvStyleVar_ChildBorderSize
mvStyleVar_PopupRounding = internal_dpg.mvStyleVar_PopupRounding
mvStyleVar_PopupBorderSize = internal_dpg.mvStyleVar_PopupBorderSize
mvStyleVar_FramePadding = internal_dpg.mvStyleVar_FramePadding
mvStyleVar_FrameRounding = internal_dpg.mvStyleVar_FrameRounding
mvStyleVar_FrameBorderSize = internal_dpg.mvStyleVar_FrameBorderSize
mvStyleVar_ItemSpacing = internal_dpg.mvStyleVar_ItemSpacing
mvStyleVar_ItemInnerSpacing = internal_dpg.mvStyleVar_ItemInnerSpacing
mvStyleVar_IndentSpacing = internal_dpg.mvStyleVar_IndentSpacing
mvStyleVar_CellPadding = internal_dpg.mvStyleVar_CellPadding
mvStyleVar_ScrollbarSize = internal_dpg.mvStyleVar_ScrollbarSize
mvStyleVar_ScrollbarRounding = internal_dpg.mvStyleVar_ScrollbarRounding
mvStyleVar_GrabMinSize = internal_dpg.mvStyleVar_GrabMinSize
mvStyleVar_GrabRounding = internal_dpg.mvStyleVar_GrabRounding
mvStyleVar_TabRounding = internal_dpg.mvStyleVar_TabRounding
mvStyleVar_ButtonTextAlign = internal_dpg.mvStyleVar_ButtonTextAlign
mvStyleVar_SelectableTextAlign = internal_dpg.mvStyleVar_SelectableTextAlign
mvPlotStyleVar_LineWeight = internal_dpg.mvPlotStyleVar_LineWeight
mvPlotStyleVar_Marker = internal_dpg.mvPlotStyleVar_Marker
mvPlotStyleVar_MarkerSize = internal_dpg.mvPlotStyleVar_MarkerSize
mvPlotStyleVar_MarkerWeight = internal_dpg.mvPlotStyleVar_MarkerWeight
mvPlotStyleVar_FillAlpha = internal_dpg.mvPlotStyleVar_FillAlpha
mvPlotStyleVar_ErrorBarSize = internal_dpg.mvPlotStyleVar_ErrorBarSize
mvPlotStyleVar_ErrorBarWeight = internal_dpg.mvPlotStyleVar_ErrorBarWeight
mvPlotStyleVar_DigitalBitHeight = internal_dpg.mvPlotStyleVar_DigitalBitHeight
mvPlotStyleVar_DigitalBitGap = internal_dpg.mvPlotStyleVar_DigitalBitGap
mvPlotStyleVar_PlotBorderSize = internal_dpg.mvPlotStyleVar_PlotBorderSize
mvPlotStyleVar_MinorAlpha = internal_dpg.mvPlotStyleVar_MinorAlpha
mvPlotStyleVar_MajorTickLen = internal_dpg.mvPlotStyleVar_MajorTickLen
mvPlotStyleVar_MinorTickLen = internal_dpg.mvPlotStyleVar_MinorTickLen
mvPlotStyleVar_MajorTickSize = internal_dpg.mvPlotStyleVar_MajorTickSize
mvPlotStyleVar_MinorTickSize = internal_dpg.mvPlotStyleVar_MinorTickSize
mvPlotStyleVar_MajorGridSize = internal_dpg.mvPlotStyleVar_MajorGridSize
mvPlotStyleVar_MinorGridSize = internal_dpg.mvPlotStyleVar_MinorGridSize
mvPlotStyleVar_PlotPadding = internal_dpg.mvPlotStyleVar_PlotPadding
mvPlotStyleVar_LabelPadding = internal_dpg.mvPlotStyleVar_LabelPadding
mvPlotStyleVar_LegendPadding = internal_dpg.mvPlotStyleVar_LegendPadding
mvPlotStyleVar_LegendInnerPadding = internal_dpg.mvPlotStyleVar_LegendInnerPadding
mvPlotStyleVar_LegendSpacing = internal_dpg.mvPlotStyleVar_LegendSpacing
mvPlotStyleVar_MousePosPadding = internal_dpg.mvPlotStyleVar_MousePosPadding
mvPlotStyleVar_AnnotationPadding = internal_dpg.mvPlotStyleVar_AnnotationPadding
mvPlotStyleVar_FitPadding = internal_dpg.mvPlotStyleVar_FitPadding
mvPlotStyleVar_PlotDefaultSize = internal_dpg.mvPlotStyleVar_PlotDefaultSize
mvPlotStyleVar_PlotMinSize = internal_dpg.mvPlotStyleVar_PlotMinSize
mvNodeStyleVar_GridSpacing = internal_dpg.mvNodeStyleVar_GridSpacing
mvNodeStyleVar_NodeCornerRounding = internal_dpg.mvNodeStyleVar_NodeCornerRounding
mvNodeStyleVar_NodePadding = internal_dpg.mvNodeStyleVar_NodePadding
mvNodeStyleVar_NodeBorderThickness = internal_dpg.mvNodeStyleVar_NodeBorderThickness
mvNodeStyleVar_LinkThickness = internal_dpg.mvNodeStyleVar_LinkThickness
mvNodeStyleVar_LinkLineSegmentsPerLength = internal_dpg.mvNodeStyleVar_LinkLineSegmentsPerLength
mvNodeStyleVar_LinkHoverDistance = internal_dpg.mvNodeStyleVar_LinkHoverDistance
mvNodeStyleVar_PinCircleRadius = internal_dpg.mvNodeStyleVar_PinCircleRadius
mvNodeStyleVar_PinQuadSideLength = internal_dpg.mvNodeStyleVar_PinQuadSideLength
mvNodeStyleVar_PinTriangleSideLength = internal_dpg.mvNodeStyleVar_PinTriangleSideLength
mvNodeStyleVar_PinLineThickness = internal_dpg.mvNodeStyleVar_PinLineThickness
mvNodeStyleVar_PinHoverRadius = internal_dpg.mvNodeStyleVar_PinHoverRadius
mvNodeStyleVar_PinOffset = internal_dpg.mvNodeStyleVar_PinOffset
mvNodesStyleVar_MiniMapPadding = internal_dpg.mvNodesStyleVar_MiniMapPadding
mvNodesStyleVar_MiniMapOffset = internal_dpg.mvNodesStyleVar_MiniMapOffset
mvInputText = internal_dpg.mvInputText
mvButton = internal_dpg.mvButton
mvRadioButton = internal_dpg.mvRadioButton
mvTabBar = internal_dpg.mvTabBar
mvTab = internal_dpg.mvTab
mvImage = internal_dpg.mvImage
mvMenuBar = internal_dpg.mvMenuBar
mvViewportMenuBar = internal_dpg.mvViewportMenuBar
mvMenu = internal_dpg.mvMenu
mvMenuItem = internal_dpg.mvMenuItem
mvChildWindow = internal_dpg.mvChildWindow
mvGroup = internal_dpg.mvGroup
mvSliderFloat = internal_dpg.mvSliderFloat
mvSliderInt = internal_dpg.mvSliderInt
mvFilterSet = internal_dpg.mvFilterSet
mvDragFloat = internal_dpg.mvDragFloat
mvDragInt = internal_dpg.mvDragInt
mvInputFloat = internal_dpg.mvInputFloat
mvInputInt = internal_dpg.mvInputInt
mvColorEdit = internal_dpg.mvColorEdit
mvClipper = internal_dpg.mvClipper
mvColorPicker = internal_dpg.mvColorPicker
mvTooltip = internal_dpg.mvTooltip
mvCollapsingHeader = internal_dpg.mvCollapsingHeader
mvSeparator = internal_dpg.mvSeparator
mvCheckbox = internal_dpg.mvCheckbox
mvListbox = internal_dpg.mvListbox
mvText = internal_dpg.mvText
mvCombo = internal_dpg.mvCombo
mvPlot = internal_dpg.mvPlot
mvSimplePlot = internal_dpg.mvSimplePlot
mvDrawlist = internal_dpg.mvDrawlist
mvWindowAppItem = internal_dpg.mvWindowAppItem
mvSelectable = internal_dpg.mvSelectable
mvTreeNode = internal_dpg.mvTreeNode
mvProgressBar = internal_dpg.mvProgressBar
mvSpacer = internal_dpg.mvSpacer
mvImageButton = internal_dpg.mvImageButton
mvTimePicker = internal_dpg.mvTimePicker
mvDatePicker = internal_dpg.mvDatePicker
mvColorButton = internal_dpg.mvColorButton
mvFileDialog = internal_dpg.mvFileDialog
mvTabButton = internal_dpg.mvTabButton
mvDrawNode = internal_dpg.mvDrawNode
mvNodeEditor = internal_dpg.mvNodeEditor
mvNode = internal_dpg.mvNode
mvNodeAttribute = internal_dpg.mvNodeAttribute
mvTable = internal_dpg.mvTable
mvTableColumn = internal_dpg.mvTableColumn
mvTableRow = internal_dpg.mvTableRow
mvDrawLine = internal_dpg.mvDrawLine
mvDrawArrow = internal_dpg.mvDrawArrow
mvDrawTriangle = internal_dpg.mvDrawTriangle
mvDrawImageQuad = internal_dpg.mvDrawImageQuad
mvDrawCircle = internal_dpg.mvDrawCircle
mvDrawEllipse = internal_dpg.mvDrawEllipse
mvDrawBezierCubic = internal_dpg.mvDrawBezierCubic
mvDrawBezierQuadratic = internal_dpg.mvDrawBezierQuadratic
mvDrawQuad = internal_dpg.mvDrawQuad
mvDrawRect = internal_dpg.mvDrawRect
mvDrawText = internal_dpg.mvDrawText
mvDrawPolygon = internal_dpg.mvDrawPolygon
mvDrawPolyline = internal_dpg.mvDrawPolyline
mvDrawImage = internal_dpg.mvDrawImage
mvDragFloatMulti = internal_dpg.mvDragFloatMulti
mvDragIntMulti = internal_dpg.mvDragIntMulti
mvSliderFloatMulti = internal_dpg.mvSliderFloatMulti
mvSliderIntMulti = internal_dpg.mvSliderIntMulti
mvInputIntMulti = internal_dpg.mvInputIntMulti
mvInputFloatMulti = internal_dpg.mvInputFloatMulti
mvDragPoint = internal_dpg.mvDragPoint
mvDragLine = internal_dpg.mvDragLine
mvAnnotation = internal_dpg.mvAnnotation
mvLineSeries = internal_dpg.mvLineSeries
mvScatterSeries = internal_dpg.mvScatterSeries
mvStemSeries = internal_dpg.mvStemSeries
mvStairSeries = internal_dpg.mvStairSeries
mvBarSeries = internal_dpg.mvBarSeries
mvErrorSeries = internal_dpg.mvErrorSeries
mvVLineSeries = internal_dpg.mvVLineSeries
mvHLineSeries = internal_dpg.mvHLineSeries
mvHeatSeries = internal_dpg.mvHeatSeries
mvImageSeries = internal_dpg.mvImageSeries
mvPieSeries = internal_dpg.mvPieSeries
mvShadeSeries = internal_dpg.mvShadeSeries
mvLabelSeries = internal_dpg.mvLabelSeries
mvHistogramSeries = internal_dpg.mvHistogramSeries
mv2dHistogramSeries = internal_dpg.mv2dHistogramSeries
mvCandleSeries = internal_dpg.mvCandleSeries
mvAreaSeries = internal_dpg.mvAreaSeries
mvColorMapScale = internal_dpg.mvColorMapScale
mvSlider3D = internal_dpg.mvSlider3D
mvKnobFloat = internal_dpg.mvKnobFloat
mvLoadingIndicator = internal_dpg.mvLoadingIndicator
mvNodeLink = internal_dpg.mvNodeLink
mvTextureRegistry = internal_dpg.mvTextureRegistry
mvStaticTexture = internal_dpg.mvStaticTexture
mvDynamicTexture = internal_dpg.mvDynamicTexture
mvStage = internal_dpg.mvStage
mvDrawLayer = internal_dpg.mvDrawLayer
mvViewportDrawlist = internal_dpg.mvViewportDrawlist
mvFileExtension = internal_dpg.mvFileExtension
mvPlotLegend = internal_dpg.mvPlotLegend
mvPlotAxis = internal_dpg.mvPlotAxis
mvHandlerRegistry = internal_dpg.mvHandlerRegistry
mvKeyDownHandler = internal_dpg.mvKeyDownHandler
mvKeyPressHandler = internal_dpg.mvKeyPressHandler
mvKeyReleaseHandler = internal_dpg.mvKeyReleaseHandler
mvMouseMoveHandler = internal_dpg.mvMouseMoveHandler
mvMouseWheelHandler = internal_dpg.mvMouseWheelHandler
mvMouseClickHandler = internal_dpg.mvMouseClickHandler
mvMouseDoubleClickHandler = internal_dpg.mvMouseDoubleClickHandler
mvMouseDownHandler = internal_dpg.mvMouseDownHandler
mvMouseReleaseHandler = internal_dpg.mvMouseReleaseHandler
mvMouseDragHandler = internal_dpg.mvMouseDragHandler
mvHoverHandler = internal_dpg.mvHoverHandler
mvActiveHandler = internal_dpg.mvActiveHandler
mvFocusHandler = internal_dpg.mvFocusHandler
mvVisibleHandler = internal_dpg.mvVisibleHandler
mvEditedHandler = internal_dpg.mvEditedHandler
mvActivatedHandler = internal_dpg.mvActivatedHandler
mvDeactivatedHandler = internal_dpg.mvDeactivatedHandler
mvDeactivatedAfterEditHandler = internal_dpg.mvDeactivatedAfterEditHandler
mvToggledOpenHandler = internal_dpg.mvToggledOpenHandler
mvClickedHandler = internal_dpg.mvClickedHandler
mvDoubleClickedHandler = internal_dpg.mvDoubleClickedHandler
mvDragPayload = internal_dpg.mvDragPayload
mvResizeHandler = internal_dpg.mvResizeHandler
mvFont = internal_dpg.mvFont
mvFontRegistry = internal_dpg.mvFontRegistry
mvTheme = internal_dpg.mvTheme
mvThemeColor = internal_dpg.mvThemeColor
mvThemeStyle = internal_dpg.mvThemeStyle
mvThemeComponent = internal_dpg.mvThemeComponent
mvFontRangeHint = internal_dpg.mvFontRangeHint
mvFontRange = internal_dpg.mvFontRange
mvFontChars = internal_dpg.mvFontChars
mvCharRemap = internal_dpg.mvCharRemap
mvValueRegistry = internal_dpg.mvValueRegistry
mvIntValue = internal_dpg.mvIntValue
mvFloatValue = internal_dpg.mvFloatValue
mvFloat4Value = internal_dpg.mvFloat4Value
mvInt4Value = internal_dpg.mvInt4Value
mvBoolValue = internal_dpg.mvBoolValue
mvStringValue = internal_dpg.mvStringValue
mvDoubleValue = internal_dpg.mvDoubleValue
mvDouble4Value = internal_dpg.mvDouble4Value
mvColorValue = internal_dpg.mvColorValue
mvFloatVectValue = internal_dpg.mvFloatVectValue
mvSeriesValue = internal_dpg.mvSeriesValue
mvRawTexture = internal_dpg.mvRawTexture
mvSubPlots = internal_dpg.mvSubPlots
mvColorMap = internal_dpg.mvColorMap
mvColorMapRegistry = internal_dpg.mvColorMapRegistry
mvColorMapButton = internal_dpg.mvColorMapButton
mvColorMapSlider = internal_dpg.mvColorMapSlider
mvTemplateRegistry = internal_dpg.mvTemplateRegistry
mvTableCell = internal_dpg.mvTableCell
mvItemHandlerRegistry = internal_dpg.mvItemHandlerRegistry
mvInputDouble = internal_dpg.mvInputDouble
mvInputDoubleMulti = internal_dpg.mvInputDoubleMulti
mvDragDouble = internal_dpg.mvDragDouble
mvDragDoubleMulti = internal_dpg.mvDragDoubleMulti
mvSliderDouble = internal_dpg.mvSliderDouble
mvSliderDoubleMulti = internal_dpg.mvSliderDoubleMulti
mvCustomSeries = internal_dpg.mvCustomSeries
mvReservedUUID_0 = internal_dpg.mvReservedUUID_0
mvReservedUUID_1 = internal_dpg.mvReservedUUID_1
mvReservedUUID_2 = internal_dpg.mvReservedUUID_2
mvReservedUUID_3 = internal_dpg.mvReservedUUID_3
mvReservedUUID_4 = internal_dpg.mvReservedUUID_4
mvReservedUUID_5 = internal_dpg.mvReservedUUID_5
mvReservedUUID_6 = internal_dpg.mvReservedUUID_6
mvReservedUUID_7 = internal_dpg.mvReservedUUID_7
mvReservedUUID_8 = internal_dpg.mvReservedUUID_8
mvReservedUUID_9 = internal_dpg.mvReservedUUID_9